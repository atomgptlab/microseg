"""Score AtomGPT's SAM3 microscopy endpoint against the digital-twin ground truth.

For every sample and dose image it calls POST {BASE}/microscopy/segment (as in the MicroscopyGPT demo
notebook), caches the JSON reply next to the sample, and scores it:

  atoms    predicted atoms vs. ground-truth atom columns, one-to-one (Hungarian) matching within --tol
  defects  a ground-truth defect counts as found if any flagged point (vacancy, off-lattice or anomaly)
           lies within --tol of it; flags far from every defect are false alarms
  sites    for matched atoms, how cleanly the solved lattice sites separate Mo from S2 columns

Points closer than --edge to the image border are ignored on both sides: the image is a periodic cell,
so columns there are cut in half.

Usage:
    export ATOMGPT_API_KEY=sk-...
    python -m microseg.evaluate_sam3 --data dataset --limit 20   # small pilot first
    python -m microseg.evaluate_sam3 --data dataset              # everything (cached replies are reused)
"""
import argparse
import csv
import glob
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from scipy.optimize import linear_sum_assignment


def segment(path, args):
    with open(path, 'rb') as f:
        img = f.read()
    for attempt in range(4):
        try:
            r = requests.post(
                f'{args.base}/microscopy/segment',
                headers={'Authorization': f'Bearer {args.key}'},
                files={'image': (os.path.basename(path), img, 'image/png')},
                data={'prompt': args.prompt, 'max_detections': args.max_detections,
                      'score_threshold': args.score_threshold, 'lattice': args.lattice},
                timeout=600,
            )
            if r.status_code in (401, 403):
                # the server reports a failed API-key DB lookup (e.g. under concurrent requests) as 401
                if attempt < 3:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise SystemExit(f'{r.status_code}: check ATOMGPT_API_KEY ({r.text[:200]})')
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            if attempt == 3:
                return {'error': str(e)}
            time.sleep(10 * (attempt + 1))


def cached_segment(path, args):
    tag = f'{args.prompt}_t{args.score_threshold}_lat{int(args.lattice)}'.replace(' ', '-')
    out = os.path.join(os.path.dirname(path), 'sam3', f'{os.path.basename(path)[:-4]}__{tag}.json')
    if os.path.exists(out):
        with open(out) as f:
            return json.load(f)
    d = segment(path, args)
    if 'error' not in d:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as f:
            json.dump(d, f)
    return d


def inside(pts, shape, margin):
    pts = np.asarray(pts, float).reshape(-1, 2)
    return pts[(pts[:, 0] >= margin) & (pts[:, 0] < shape[1] - margin) &
               (pts[:, 1] >= margin) & (pts[:, 1] < shape[0] - margin)]


def match(gt, pred, tol):
    """One-to-one matching within tol (px). Returns list of (gt_index, pred_index)."""
    if len(gt) == 0 or len(pred) == 0:
        return []
    dist = np.linalg.norm(gt[:, None, :] - pred[None, :, :], axis=2)
    gi, pi = linear_sum_assignment(np.where(dist <= tol, dist, 1e6))
    return [(g, p) for g, p in zip(gi, pi) if dist[g, p] <= tol]


def score(labels, d, args):
    shape = labels['shape_rows_cols']
    px = float(np.mean(labels['pixel_size_xy_A']))
    tol, margin = args.tol / px, args.edge / px

    gt_cols = [c for c in labels['columns'] if len(inside([c['xy_px']], shape, margin))]
    gt_xy = np.array([c['xy_px'] for c in gt_cols]).reshape(-1, 2)
    pred_all = np.asarray(d.get('atoms') or [], float).reshape(-1, 2)
    keep = np.array([len(inside([p], shape, margin)) > 0 for p in pred_all], bool)
    pred_xy = pred_all[keep] if len(pred_all) else pred_all
    m = match(gt_xy, pred_xy, tol)
    row = {'gt_atoms': len(gt_xy), 'pred_atoms': len(pred_xy), 'atoms_tp': len(m)}

    # does the lattice solve put Mo and S2 columns on different sites?
    sites = np.asarray(d.get('atom_site') or [], int)
    sites = sites[keep] if len(sites) == len(pred_all) else np.array([], int)
    if len(sites):
        table = defaultdict(lambda: defaultdict(int))
        for g, p in m:
            if gt_cols[g]['class'] in ('Mo', 'S2'):
                table[int(sites[p])][gt_cols[g]['class']] += 1
        n = sum(sum(v.values()) for v in table.values())
        row['site_purity'] = sum(max(v.values()) for v in table.values()) / n if n else np.nan

    flags = []
    for key in ('vacancies', 'off_lattice', 'anomaly_xy'):
        for p in inside(d.get(key) or [], shape, margin):
            flags.append((key, p))
    gt_def = [r for r in labels['defects'] if len(inside([r['xy_px']], shape, margin))]
    used = set()
    for r in gt_def:
        hits = [k for k, (key, p) in enumerate(flags) if np.linalg.norm(p - r['xy_px']) <= tol]
        r['found_by'] = flags[hits[0]][0] if hits else None
        used.update(hits)
    row.update({'gt_defects': len(gt_def), 'defects_found': sum(r['found_by'] is not None for r in gt_def),
                'flags': len(flags), 'false_flags': len(flags) - len(used)})
    return row, gt_def


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='dataset')
    ap.add_argument('--base', default=os.environ.get('BASE', 'https://atomgpt.org'))
    ap.add_argument('--key', default=os.environ.get('ATOMGPT_API_KEY'))
    ap.add_argument('--prompt', default='dots')
    ap.add_argument('--score-threshold', type=float, default=0.3)
    ap.add_argument('--max-detections', type=int, default=400)
    ap.add_argument('--no-lattice', dest='lattice', action='store_false', help='SAM3 detections only')
    ap.add_argument('--doses', nargs='+', help='e.g. inf 1000000 10000 (default: all present)')
    ap.add_argument('--limit', type=int, help='only the first N samples')
    ap.add_argument('--workers', type=int, default=1, help='>1 triggers spurious 401s on atomgpt.org')
    ap.add_argument('--tol', type=float, default=0.5, help='match tolerance, A')
    ap.add_argument('--edge', type=float, default=0.6, help='ignore points this close to the border, A')
    ap.add_argument('--out', default='sam3_results')
    args = ap.parse_args()
    key_file = os.path.expanduser('~/.atomgpt_key')
    if not args.key and os.path.exists(key_file):
        args.key = open(key_file).read().strip()
    if not args.key:
        raise SystemExit('Set ATOMGPT_API_KEY or put the key in ~/.atomgpt_key '
                         '(atomgpt.org > Profile > Settings > Account > API Keys)')

    samples = sorted(glob.glob(os.path.join(args.data, 'sample_*')))[:args.limit]
    jobs = []
    for s in samples:
        for png in sorted(glob.glob(os.path.join(s, 'dose_*.png'))):
            dose = os.path.basename(png)[5:-4]
            if args.doses is None or dose in args.doses:
                jobs.append((s, dose, png))
    print(f'{len(jobs)} images from {len(samples)} samples')

    with ThreadPoolExecutor(args.workers) as pool:
        replies = list(pool.map(lambda j: cached_segment(j[2], args), jobs))

    os.makedirs(args.out, exist_ok=True)
    rows, per_defect = [], []
    for (s, dose, _), d in zip(jobs, replies):
        if 'error' in d:
            print(f'{s} {dose}: {d["error"]}')
            continue
        with open(os.path.join(s, 'labels.json')) as f:
            labels = json.load(f)
        row, defs = score(labels, d, args)
        row.update({'sample': os.path.basename(s), 'dose': dose,
                    'poor_fit': (d.get('lattice_analysis') or {}).get('poor_fit')})
        rows.append(row)
        for r in defs:
            per_defect.append({'sample': row['sample'], 'dose': dose, 'type': r['type'],
                               'element': r.get('element', ''), 'layer': r.get('layer', ''),
                               'found_by': r['found_by'] or ''})

    def write(name, data):
        if data:
            with open(os.path.join(args.out, name), 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in data for k in r)))
                w.writeheader()
                w.writerows(data)

    write('per_image.csv', rows)
    write('per_defect.csv', per_defect)

    def dose_key(x):
        return float('inf') if x == 'inf' else float(x)

    summary = []
    for dose in sorted({r['dose'] for r in rows}, key=dose_key, reverse=True):
        R = [r for r in rows if r['dose'] == dose]
        tp, ng, npred = (sum(r[k] for r in R) for k in ('atoms_tp', 'gt_atoms', 'pred_atoms'))
        prec, rec = tp / max(npred, 1), tp / max(ng, 1)
        D = [p for p in per_defect if p['dose'] == dose]
        flags, false = sum(r['flags'] for r in R), sum(r['false_flags'] for r in R)
        line = {'dose': dose, 'images': len(R),
                'atom_precision': round(prec, 3), 'atom_recall': round(rec, 3),
                'atom_F1': round(2 * prec * rec / max(prec + rec, 1e-12), 3),
                'site_purity': round(float(np.nanmean([r.get('site_purity', np.nan) for r in R])), 3),
                'defect_recall': round(sum(p['found_by'] != '' for p in D) / max(len(D), 1), 3),
                'flag_precision': round((flags - false) / flags, 3) if flags else '',
                'false_flags_per_image': round(false / len(R), 2)}
        for t in sorted({p['type'] for p in D}):
            T = [p for p in D if p['type'] == t]
            line[f'recall_{t}'] = f'{sum(p["found_by"] != "" for p in T)}/{len(T)}'
        summary.append(line)
    write('summary_by_dose.csv', summary)
    for line in summary:
        print(line)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        x = [1e8 if s['dose'] == 'inf' else float(s['dose']) for s in summary]
        fig, ax = plt.subplots(figsize=(6, 4))
        for k in ('atom_F1', 'defect_recall', 'flag_precision'):
            ax.plot(x, [s[k] if s[k] != '' else np.nan for s in summary], 'o-', label=k)
        ax.set_xscale('log')
        ax.set_xlabel('dose (e/Å²); rightmost point = noise-free')
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.set_title(f'AtomGPT SAM3 ("{args.prompt}") on MoS$_2$ digital twins')
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, 'sam3_vs_dose.png'), dpi=150)
    except ImportError:
        pass


if __name__ == '__main__':
    main()
