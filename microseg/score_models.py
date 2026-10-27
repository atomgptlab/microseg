"""Score every model in model_preds/ against the digital-twin ground truth.

Tier 1, atom columns (straight from each model's centres):
    precision / recall / F1, one-to-one matching within --tol; recall split by column class
    (Mo, S2, and columns created by defects); `at_vacancy` = predictions sitting on an empty
    (V_Mo / V_S2) site, i.e. atoms the model invented where one is missing.

Tier 2, defects (each model's centres fed to the AtomGPT lattice solver, lattice_defects.analyze,
with its own peak finder switched off, so the solver only sees what the model found):
    located  = a vacancy / off-lattice / anomaly flag within --tol of the defect
    strict   = the flag is also the expected kind (EXPECTED_FLAG below)
    false flags per image = flags not near any defect
For `atomgpt` the production reply's own flags are used (its solver also runs its own peak finder).

Points within --edge of the border are ignored (the image is a periodic cell, so edge columns are cut).

Usage:
    python -m microseg.score_models --data dataset --preds model_preds --out model_results
"""
import argparse
import csv
import glob
import importlib.util
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

EMPTY_SITES = ('V_Mo', 'V_S2')
EXPECTED_FLAG = {  # what a correct lattice/defect solve should call each defect
    'V_Mo': {'vacancies'}, 'V_S2': {'vacancies'},
    'V_S': {'vacancies', 'anomaly_xy'},      # half an S2 column: dim, may fall either side of vac_frac
    'S_Mo': {'vacancies', 'anomaly_xy'},     # lone S on a Mo site is much dimmer than Mo
    'Mo_S': {'anomaly_xy'}, 'Mo_S2': {'anomaly_xy'}, 'TM_dopant': {'anomaly_xy'},
    'TM_adatom': {'off_lattice'}, 'Mo_i': {'off_lattice'}, 'S_i': {'off_lattice'},
}
FLAG_KEYS = ('vacancies', 'off_lattice', 'anomaly_xy')


def load_solver(path):
    spec = importlib.util.spec_from_file_location('lattice_defects', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def inside(xy, shape, margin):
    xy = np.asarray(xy, float).reshape(-1, 2)
    return (xy[:, 0] >= margin) & (xy[:, 0] < shape[1] - margin) & (xy[:, 1] >= margin) & (xy[:, 1] < shape[0] - margin)


def match(gt, pred, tol):
    if len(gt) == 0 or len(pred) == 0:
        return []
    d = np.linalg.norm(gt[:, None] - pred[None], axis=2)
    gi, pi = linear_sum_assignment(np.where(d <= tol, d, 1e6))
    return [(g, p) for g, p in zip(gi, pi) if d[g, p] <= tol]


def solve(solver, png, centers, cache):
    """Flags from the lattice solver given only `centers`; cached because it is the slow part."""
    if os.path.exists(cache):
        return json.load(open(cache))
    out = {k: [] for k in FLAG_KEYS}
    if len(centers) >= 10:
        try:
            img = np.asarray(Image.open(png).convert('L'), float)
            r = solver.analyze(img, extra_centers=centers, peak_rel=1e9)  # 1e9: solver's own peaks off
            a, off = r.pop('_arrays'), np.asarray(r['offset'], float)
            out = {'vacancies': (np.asarray(a['vac']).reshape(-1, 2) + off).tolist(),
                   'off_lattice': (np.asarray(a['inter']).reshape(-1, 2) + off).tolist(),
                   'anomaly_xy': (np.asarray(a['atoms'])[a['anom']].reshape(-1, 2) + off).tolist(),
                   'K': r['K'], 'poor_fit': r['poor_fit']}
        except Exception as e:
            out['error'] = f'{type(e).__name__}: {e}'[:300]
    else:
        out['error'] = 'fewer than 10 centres'
    with open(cache, 'w') as f:
        json.dump(out, f)
    return out


def score_image(labels, pred, flags, args):
    shape = labels['shape_rows_cols']
    tol = args.tol / float(np.mean(labels['pixel_size_xy_A']))
    margin = args.edge / float(np.mean(labels['pixel_size_xy_A']))

    cols = [c for c in labels['columns'] if inside(c['xy_px'], shape, margin)[0]]
    gt = np.array([c['xy_px'] for c in cols]).reshape(-1, 2)
    p = np.asarray(pred['centers'], float).reshape(-1, 2)
    p = p[inside(p, shape, margin)] if len(p) else p
    m = match(gt, p, tol)
    hit = np.zeros(len(cols), bool)
    hit[[g for g, _ in m]] = True
    row = {'gt': len(gt), 'pred': len(p), 'tp': len(m), 'seconds': pred.get('seconds'), 'error': int('error' in pred)}
    for cls in ('Mo', 'S2', 'defect_col'):
        sel = [k for k, c in enumerate(cols) if (c['class'] == cls if cls != 'defect_col' else c['class'] not in ('Mo', 'S2'))]
        row[f'{cls}_n'], row[f'{cls}_hit'] = len(sel), int(hit[sel].sum())
    empty = np.array([d['xy_px'] for d in labels['defects'] if d['type'] in EMPTY_SITES]).reshape(-1, 2)
    row['at_vacancy'] = int(sum(np.any(np.linalg.norm(p - e, axis=1) <= tol) for e in empty)) if len(p) else 0
    row['vacancy_sites'] = len(empty)

    fl = [(k, np.asarray(xy, float)) for k in FLAG_KEYS for xy in flags.get(k, [])]
    fl = [(k, xy) for k, xy in fl if inside(xy, shape, margin)[0]]
    used, per_def = set(), []
    for d in labels['defects']:
        if not inside(d['xy_px'], shape, margin)[0]:
            continue
        near = [i for i, (k, xy) in enumerate(fl) if np.linalg.norm(xy - d['xy_px']) <= tol]
        used.update(near)
        per_def.append({'type': d['type'], 'located': bool(near),
                        'strict': any(fl[i][0] in EXPECTED_FLAG[d['type']] for i in near)})
    row.update({'defects': len(per_def), 'located': sum(d['located'] for d in per_def),
                'strict': sum(d['strict'] for d in per_def), 'flags': len(fl), 'false_flags': len(fl) - len(used),
                'solver_error': int('error' in flags)})
    return row, per_def


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='dataset')
    ap.add_argument('--preds', default='model_preds')
    ap.add_argument('--out', default='model_results')
    ap.add_argument('--solver', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lattice_defects.py'))
    ap.add_argument('--tol', type=float, default=0.5, help='A')
    ap.add_argument('--edge', type=float, default=0.6, help='A')
    args = ap.parse_args()
    solver = load_solver(args.solver)
    os.makedirs(args.out, exist_ok=True)

    rows, defs = [], []
    for mdir in sorted(glob.glob(os.path.join(args.preds, '*'))):
        model = os.path.basename(mdir)
        os.makedirs(os.path.join(mdir, 'solver'), exist_ok=True)
        for f in sorted(glob.glob(os.path.join(mdir, 'sample_*.json'))):
            sample, dose = os.path.basename(f)[:-5].split('__')
            labels = json.load(open(os.path.join(args.data, sample, 'labels.json')))
            pred = json.load(open(f))
            png = os.path.join(args.data, sample, f'dose_{dose}.png')
            if model == 'atomgpt':
                api = glob.glob(os.path.join(args.data, sample, 'sam3', f'dose_{dose}__*_lat1.json'))
                flags = json.load(open(api[0])) if api else {'error': 'no reply'}
            else:
                flags = solve(solver, png, pred['centers'], os.path.join(mdir, 'solver', os.path.basename(f)))
            row, pd_ = score_image(labels, pred, flags, args)
            row.update({'model': model, 'sample': sample, 'dose': dose})
            rows.append(row)
            defs += [dict(d, model=model, sample=sample, dose=dose) for d in pd_]
        print(f'{model}: {sum(r["model"] == model for r in rows)} images', flush=True)

    def write(name, data):
        with open(os.path.join(args.out, name), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in data for k in r)))
            w.writeheader()
            w.writerows(data)

    write('per_image.csv', rows)
    write('per_defect.csv', defs)

    dose_val = lambda d: float('inf') if d == 'inf' else float(d)
    summary, by_type = [], defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for model in sorted({r['model'] for r in rows}):
        for dose in sorted({r['dose'] for r in rows if r['model'] == model}, key=dose_val, reverse=True):
            R = [r for r in rows if r['model'] == model and r['dose'] == dose]
            S = lambda k: sum(r[k] for r in R)
            prec, rec = S('tp') / max(S('pred'), 1), S('tp') / max(S('gt'), 1)
            secs = [r['seconds'] for r in R if r['seconds'] is not None]
            summary.append({
                'model': model, 'dose': dose, 'images': len(R),
                'precision': round(prec, 3), 'recall': round(rec, 3), 'F1': round(2 * prec * rec / max(prec + rec, 1e-9), 3),
                'recall_Mo': round(S('Mo_hit') / max(S('Mo_n'), 1), 3), 'recall_S2': round(S('S2_hit') / max(S('S2_n'), 1), 3),
                'recall_defect_cols': round(S('defect_col_hit') / max(S('defect_col_n'), 1), 3),
                'atoms_at_vacancies': f"{S('at_vacancy')}/{S('vacancy_sites')}",
                'defects_located': round(S('located') / max(S('defects'), 1), 3),
                'defects_strict': round(S('strict') / max(S('defects'), 1), 3),
                'false_flags_per_img': round(S('false_flags') / len(R), 2),
                'solver_failed': S('solver_error'), 'model_errors': S('error'),
                'sec_per_img': round(float(np.median(secs)), 2) if secs else '',
            })
            for d in defs:
                if d['model'] == model and d['dose'] == dose:
                    by_type[(model, dose)][d['type']][0] += d['strict']
                    by_type[(model, dose)][d['type']][1] += 1
    write('summary.csv', summary)
    write('strict_recall_by_type.csv', [dict({'model': m, 'dose': d}, **{t: f'{a}/{n}' for t, (a, n) in sorted(v.items())})
                                        for (m, d), v in by_type.items()])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    for model in sorted({s['model'] for s in summary}):
        S = [s for s in summary if s['model'] == model]
        x = [1e8 if s['dose'] == 'inf' else float(s['dose']) for s in S]
        for ax, k in zip(axs, ('F1', 'recall_S2', 'defects_strict')):
            ax.plot(x, [s[k] for s in S], 'o-', label=model)
    for ax, t in zip(axs, ('atom-column F1', 'S$_2$ column recall', 'defects found (strict)')):
        ax.set_xscale('log')
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(t)
        ax.set_xlabel('dose (e/Å²); 1e8 = noise-free')
    axs[-1].legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'models_vs_dose.png'), dpi=150)

    for s in summary:
        if s['dose'] in ('inf', '150000', '10000'):
            print({k: s[k] for k in ('model', 'dose', 'precision', 'recall', 'recall_S2', 'defects_located',
                                     'defects_strict', 'false_flags_per_img', 'sec_per_img')})


if __name__ == '__main__':
    main()
