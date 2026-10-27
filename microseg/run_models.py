"""Run general-purpose segmentation/detection models on the digital-twin images.

Every model is reduced to the same output, a list of atom-column centres (x, y in pixels, the
dataset's convention) with scores, saved per image as JSON:

    <out>/<model>/<sample>__<dose>.json   {"centers": [[x, y], ...], "scores": [...], "seconds": t}

so `microseg.score_models` can compare them on equal terms. Nothing is tuned to the images: each model
runs with its library defaults, text-prompted models get the same prompt ("dots", as AtomGPT uses),
and the only post-processing is dropping masks/boxes larger than --max-area of the image (whole-image
or background segments), applied to every model alike.

Models: peaks (classical baseline), sam3, sam1, sam2, gdino, owlv2, florence2, cellpose, microsam,
atomgpt (reads the cached atomgpt.org replies written by evaluate_sam3.py; no GPU).

Usage:
    python -m microseg.run_models --model sam3 --data dataset --limit 50
"""
import argparse
import glob
import json
import os
import time

import numpy as np
from PIL import Image


def centroids(masks, scores, max_px):
    out_c, out_s = [], []
    for m, s in zip(masks, scores):
        m = np.asarray(m, bool)
        n = m.sum()
        if 0 < n <= max_px:
            ys, xs = np.nonzero(m)
            out_c.append([float(xs.mean()), float(ys.mean())])
            out_s.append(float(s))
    return out_c, out_s


def box_centers(boxes, scores, max_px):
    out_c, out_s = [], []
    for (x0, y0, x1, y1), s in zip(np.asarray(boxes, float).reshape(-1, 4), scores):
        if (x1 - x0) * (y1 - y0) <= max_px:
            out_c.append([(x0 + x1) / 2, (y0 + y1) / 2])
            out_s.append(float(s))
    return out_c, out_s


class Peaks:
    """Classical baseline: Gaussian smoothing + local maxima. Generic settings, not tuned to MoS2."""

    def __init__(self, args):
        from scipy.ndimage import gaussian_filter
        from skimage.feature import peak_local_max
        self.gf, self.plm = gaussian_filter, peak_local_max

    def __call__(self, img, max_px):
        a = self.gf(np.asarray(img.convert('L'), float), 2)
        pk = self.plm(a, min_distance=10, threshold_rel=0.1, exclude_border=False)
        return pk[:, ::-1].astype(float).tolist(), a[pk[:, 0], pk[:, 1]].tolist()


class Sam3:
    def __init__(self, args):
        import torch
        from transformers import Sam3Model, Sam3Processor
        self.torch = torch
        self.model = Sam3Model.from_pretrained('facebook/sam3', torch_dtype=torch.bfloat16).to('cuda').eval()
        self.proc = Sam3Processor.from_pretrained('facebook/sam3')
        self.prompt, self.thr = args.prompt, args.threshold if args.threshold is not None else 0.3

    def __call__(self, img, max_px):
        inp = self.proc(images=img.convert('RGB'), text=self.prompt, return_tensors='pt').to('cuda')
        inp = {k: (v.to(self.torch.bfloat16) if v.is_floating_point() else v) for k, v in inp.items()}
        with self.torch.inference_mode():
            out = self.model(**inp)
        r = self.proc.post_process_instance_segmentation(out, threshold=self.thr, mask_threshold=0.5,
                                                         target_sizes=[img.size[::-1]])[0]
        return centroids(r['masks'].cpu().numpy(), r['scores'].float().cpu().numpy(), max_px)


class MaskGenerator:
    """SAM 1 / SAM 2.1 'segment everything' via the transformers mask-generation pipeline (defaults)."""
    checkpoint = None

    def __init__(self, args):
        from transformers import pipeline
        self.pipe = pipeline('mask-generation', model=self.checkpoint, device=0, points_per_batch=64)

    def __call__(self, img, max_px):
        r = self.pipe(img.convert('RGB'))
        return centroids(r['masks'], np.asarray(r['scores'], float), max_px)


class Sam1(MaskGenerator):
    checkpoint = 'facebook/sam-vit-huge'


class Sam2(MaskGenerator):
    checkpoint = 'facebook/sam2.1-hiera-large'


class GroundingDino:
    def __init__(self, args):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.torch = torch
        mid = 'IDEA-Research/grounding-dino-base'
        self.proc = AutoProcessor.from_pretrained(mid)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to('cuda').eval()
        self.prompt = args.prompt.rstrip('.')  # transformers 5 takes a list of labels and adds the '.' itself
        self.thr = args.threshold if args.threshold is not None else 0.25

    def __call__(self, img, max_px):
        inp = self.proc(images=img.convert('RGB'), text=[[self.prompt]], return_tensors='pt').to('cuda')
        with self.torch.inference_mode():
            out = self.model(**inp)
        r = self.proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=self.thr,
                                                             text_threshold=self.thr,
                                                             target_sizes=[img.size[::-1]])[0]
        return box_centers(r['boxes'].cpu().numpy(), r['scores'].cpu().numpy(), max_px)


class Owlv2:
    def __init__(self, args):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.torch = torch
        mid = 'google/owlv2-base-patch16-ensemble'
        self.proc = Owlv2Processor.from_pretrained(mid)
        self.model = Owlv2ForObjectDetection.from_pretrained(mid).to('cuda').eval()
        self.prompt, self.thr = args.prompt, args.threshold if args.threshold is not None else 0.1

    def __call__(self, img, max_px):
        inp = self.proc(text=[[self.prompt]], images=img.convert('RGB'), return_tensors='pt').to('cuda')
        with self.torch.inference_mode():
            out = self.model(**inp)
        side = max(img.size)  # OWLv2 pads to a square; boxes come back in padded coordinates
        r = self.proc.post_process_grounded_object_detection(out, threshold=self.thr, target_sizes=[(side, side)])[0]
        return box_centers(r['boxes'].cpu().numpy(), r['scores'].cpu().numpy(), max_px)


class Florence2:
    def __init__(self, args):
        import torch
        from transformers import AutoProcessor, Florence2ForConditionalGeneration
        self.torch = torch
        mid = 'florence-community/Florence-2-large'
        self.proc = AutoProcessor.from_pretrained(mid)
        self.model = Florence2ForConditionalGeneration.from_pretrained(mid, torch_dtype=torch.bfloat16).to('cuda').eval()
        self.task, self.prompt = '<OPEN_VOCABULARY_DETECTION>', args.prompt

    def __call__(self, img, max_px):
        img = img.convert('RGB')
        inp = self.proc(text=self.task + self.prompt, images=img, return_tensors='pt').to('cuda', self.torch.bfloat16)
        with self.torch.inference_mode():
            ids = self.model.generate(**inp, max_new_tokens=1024, num_beams=3, do_sample=False)
        text = self.proc.batch_decode(ids, skip_special_tokens=False)[0]
        r = self.proc.post_process_generation(text, task=self.task, image_size=img.size)[self.task]
        boxes = r.get('bboxes', [])
        return box_centers(boxes, [1.0] * len(boxes), max_px)


class Cellpose:
    def __init__(self, args):
        from cellpose import models
        self.model = models.CellposeModel(gpu=True)  # Cellpose-SAM (cpsam) in cellpose >= 4

    def __call__(self, img, max_px):
        labels = self.model.eval(np.asarray(img.convert('L')))[0]
        ids = [i for i in np.unique(labels) if i]
        return centroids([labels == i for i in ids], [1.0] * len(ids), max_px)


class MicroSam:
    def __init__(self, args):
        from micro_sam.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter
        self.run = automatic_instance_segmentation
        self.predictor, self.segmenter = get_predictor_and_segmenter(model_type='vit_b_lm', segmentation_mode='ais')

    def __call__(self, img, max_px):
        labels = self.run(self.predictor, self.segmenter, np.asarray(img.convert('L')), ndim=2, verbose=False)
        ids = [i for i in np.unique(labels) if i]
        return centroids([labels == i for i in ids], [1.0] * len(ids), max_px)


MODELS = {'peaks': Peaks, 'sam3': Sam3, 'sam1': Sam1, 'sam2': Sam2, 'gdino': GroundingDino,
          'owlv2': Owlv2, 'florence2': Florence2, 'cellpose': Cellpose, 'microsam': MicroSam}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, choices=list(MODELS) + ['atomgpt'])
    ap.add_argument('--data', default='dataset')
    ap.add_argument('--out', default='model_preds')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--doses', nargs='+', help='default: all present')
    ap.add_argument('--prompt', default='dots')
    ap.add_argument('--threshold', type=float, help='score threshold (model default if unset)')
    ap.add_argument('--max-area', type=float, default=0.02, help='drop masks/boxes above this image fraction')
    args = ap.parse_args()

    jobs = []
    for s in sorted(glob.glob(os.path.join(args.data, 'sample_*')))[:args.limit]:
        for png in sorted(glob.glob(os.path.join(s, 'dose_*.png'))):
            dose = os.path.basename(png)[5:-4]
            if args.doses is None or dose in args.doses:
                jobs.append((os.path.basename(s), dose, png))
    out_dir = os.path.join(args.out, args.model)
    os.makedirs(out_dir, exist_ok=True)
    todo = [j for j in jobs if not os.path.exists(os.path.join(out_dir, f'{j[0]}__{j[1]}.json'))]
    print(f'{args.model}: {len(jobs)} images, {len(todo)} to run', flush=True)

    if args.model == 'atomgpt':
        for name, dose, png in todo:
            f = glob.glob(os.path.join(os.path.dirname(png), 'sam3', f'dose_{dose}__*_lat1.json'))
            if f:
                d = json.load(open(f[0]))
                with open(os.path.join(out_dir, f'{name}__{dose}.json'), 'w') as fh:
                    json.dump({'centers': d.get('atoms', []), 'scores': [1.0] * len(d.get('atoms', [])),
                               'seconds': (d.get('processing_time') or {}).get('total')}, fh)
        return

    model = MODELS[args.model](args)
    for k, (name, dose, png) in enumerate(todo):
        img = Image.open(png)
        max_px = args.max_area * img.size[0] * img.size[1]
        t = time.time()
        try:
            c, s = model(img, max_px)
            rec = {'centers': c, 'scores': s}
        except Exception as e:  # a model failing on one image is a result, not a crash
            rec = {'centers': [], 'scores': [], 'error': f'{type(e).__name__}: {e}'[:500]}
        rec['seconds'] = time.time() - t
        with open(os.path.join(out_dir, f'{name}__{dose}.json'), 'w') as fh:
            json.dump(rec, fh)
        if k % 20 == 0 or 'error' in rec:
            print(f'  [{k + 1}/{len(todo)}] {name} {dose}: {len(rec["centers"])} centres, '
                  f'{rec["seconds"]:.1f}s {rec.get("error", "")}', flush=True)


if __name__ == '__main__':
    main()
