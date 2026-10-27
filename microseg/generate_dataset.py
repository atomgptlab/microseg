"""Generate labelled MoS2 STEM digital twins (abTEM multislice) for benchmarking segmentation models.

Follows the MoS2 conditions of Fuhr et al. (Appl. Phys. Lett., DMX digital twins): 60 kV, 30 mrad probe,
HAADF 80-200 mrad, 0.5 A Gaussian filter, Poisson noise at several electron doses. Defect classes are the
paper's nine (adapted to MoS2) plus a double anion vacancy.

Unlike the original notebook, each image is a scan of the *whole* periodic cell (no scan-and-tile), so every
defect appears exactly once and its pixel position is known.

Output layout (one folder per sample):
    sample_0000/
        structure.vasp          defected structure that was simulated
        clean.npy               float32 HAADF image, no blur / noise, shape (rows=y, cols=x)
        dose_inf.png            blurred, noise-free, 8-bit (min-max normalised)
        dose_<D>.png            blurred + Poisson noise at D e/A^2, 8-bit
        mask.png                uint8 per-pixel class index (see classes.json)
        labels.json             columns, defects, pixel size, simulation and noise parameters
    classes.json, manifest.csv

Pixel convention: col = x / pixel_size_x, row = y / pixel_size_y, row 0 at the top of the PNG.

Usage:
    python -m microseg.generate_dataset --vasp mos2-B_Defect-Free.vasp --n 200 --out dataset
    (on WSL prefix NUMBA_CUDA_DRIVER=/usr/lib/wsl/lib/libcuda.so.1 so numba finds the GPU)
"""
import argparse
import csv
import json
import os
import time

import numpy as np

# abTEM 1.0.0b32 predates NumPy 2.x / SciPy 1.17 (same shims as the notebook).
for _name, _alias in {'float': float, 'int': int, 'complex': complex, 'product': np.prod}.items():
    if not hasattr(np, _name):
        setattr(np, _name, _alias)
import scipy.special
if not hasattr(scipy.special, 'sph_harm'):
    scipy.special.sph_harm = lambda m, n, theta, phi: scipy.special.sph_harm_y(n, m, phi, theta)

from ase import Atoms
from ase.io import read, write
from PIL import Image
from abtem import Potential, Probe, GridScan, FlexibleAnnularDetector, FrozenPhonons
from abtem.noise import poisson_noise
from abtem.structures import orthogonalize_cell

CLASSES = ['background', 'Mo', 'S2',
           'V_S', 'V_S2', 'V_Mo', 'Mo_S', 'Mo_S2', 'S_Mo', 'TM_dopant', 'TM_adatom', 'Mo_i', 'S_i']
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
DEFECT_WEIGHTS = {'V_S': 3, 'V_S2': 2, 'V_Mo': 1, 'Mo_S': 1, 'Mo_S2': 1, 'S_Mo': 1,
                  'TM_dopant': 1, 'TM_adatom': 1, 'Mo_i': 1, 'S_i': 1}
DOPANTS = ['W', 'Re', 'Nb', 'V']      # Nb (Z=41) is nearly invisible next to Mo (Z=42), on purpose
ADATOMS = ['Mo', 'W']
MIN_DEFECT_SEPARATION = 5.0           # A, minimum-image distance between defect sites
MASK_RADIUS = 0.6                     # A, disk radius per column in mask.png


def min_image(d, L):
    return d - L * np.round(d / L)


def build_pristine(vasp):
    atoms = read(vasp, format='vasp')
    atoms = orthogonalize_cell(atoms, max_repetitions=2) * (2, 1, 1)
    atoms.wrap()
    return atoms


def find_sites(atoms):
    """Mo columns, S2 columns and hollow sites (xy) of a pristine 2H-MX2 orthogonal cell."""
    L = np.diag(atoms.cell)[:2]
    pos = atoms.positions
    sym = np.array(atoms.get_chemical_symbols())
    mo = np.where(sym == 'Mo')[0]
    s = np.where(sym == 'S')[0]

    s_cols, used = [], set()
    for i in s:
        if i in used:
            continue
        d = np.linalg.norm(min_image(pos[s, :2] - pos[i, :2], L), axis=1)
        pair = [int(j) for j in s[d < 0.3]]
        used.update(pair)
        s_cols.append(pair)

    hollows = []
    for i in mo:
        d = min_image(pos[s, :2] - pos[i, :2], L)
        nearest = d[np.argsort(np.linalg.norm(d, axis=1))[0]]
        h = np.mod(pos[i, :2] - nearest, L)
        if all(np.linalg.norm(min_image(h - o, L)) > 0.3 for o in hollows):
            hollows.append(h)
    return mo, s_cols, np.array(hollows)


def add_defects(atoms, rng, n_defects):
    """Return (defected Atoms, defect records). Sites are kept MIN_DEFECT_SEPARATION apart."""
    L = np.diag(atoms.cell)[:2]
    pos = atoms.positions.copy()
    sym = list(atoms.get_chemical_symbols())
    mo, s_cols, hollows = find_sites(atoms)
    z_mo = pos[mo, 2].mean()
    z_top = max(pos[j, 2] for c in s_cols for j in c)

    names = list(DEFECT_WEIGHTS)
    p = np.array([DEFECT_WEIGHTS[k] for k in names], float)
    remove, add, records, taken = set(), [], [], []

    def free(xy):
        return all(np.linalg.norm(min_image(xy - t, L)) >= MIN_DEFECT_SEPARATION for t in taken)

    for _ in range(n_defects):
        kind = rng.choice(names, p=p / p.sum())
        for _attempt in range(200):
            if kind in ('V_Mo', 'S_Mo', 'TM_dopant'):
                i = int(rng.choice(mo))
                xy = pos[i, :2]
            elif kind in ('TM_adatom', 'Mo_i', 'S_i'):
                xy = hollows[rng.integers(len(hollows))]
            else:
                col = s_cols[rng.integers(len(s_cols))]
                xy = pos[col, :2].mean(axis=0)
            if free(xy):
                break
        else:
            continue
        taken.append(xy)
        rec = {'type': kind, 'xy_A': [float(xy[0]), float(xy[1])]}

        if kind == 'V_Mo':
            remove.add(i)
        elif kind == 'S_Mo':
            sym[i] = 'S'
        elif kind == 'TM_dopant':
            sym[i] = rec['element'] = str(rng.choice(DOPANTS))
        elif kind in ('V_S', 'Mo_S'):
            j = int(rng.choice(col))
            rec['layer'] = 'top' if pos[j, 2] > z_mo else 'bottom'
            if kind == 'V_S':
                remove.add(j)
            else:
                sym[j] = 'Mo'
        elif kind == 'V_S2':
            remove.update(col)
        elif kind == 'Mo_S2':
            remove.update(col)
            add.append(('Mo', [xy[0], xy[1], z_mo]))
        elif kind == 'TM_adatom':
            rec['element'] = str(rng.choice(ADATOMS))
            add.append((rec['element'], [xy[0], xy[1], z_top + 1.6]))
        elif kind == 'Mo_i':
            add.append(('Mo', [xy[0], xy[1], z_mo]))
        elif kind == 'S_i':
            add.append(('S', [xy[0], xy[1], z_mo]))
        records.append(rec)

    keep = [k for k in range(len(sym)) if k not in remove]
    new = Atoms([sym[k] for k in keep] + [a[0] for a in add],
                positions=np.vstack([pos[keep]] + [np.array([a[1] for a in add])] if add else [pos[keep]]),
                cell=atoms.cell, pbc=True)
    new.center(vacuum=2, axis=2)
    return new, records


def columns(atoms, tol=0.3):
    """Group atoms into projected columns: [{'xy_A', 'species': {el: n}}]."""
    L = np.diag(atoms.cell)[:2]
    pos = atoms.positions
    sym = atoms.get_chemical_symbols()
    cols = []
    for p, s in zip(pos, sym):
        for c in cols:
            if np.linalg.norm(min_image(p[:2] - c['xy_A'], L)) < tol:
                c['species'][s] = c['species'].get(s, 0) + 1
                break
        else:
            cols.append({'xy_A': p[:2].copy(), 'species': {s: 1}})
    for c in cols:
        c['xy_A'] = [float(c['xy_A'][0]), float(c['xy_A'][1])]
    return cols


def label_columns(cols, defects, L, tol=0.3):
    """Class of each column: the defect sitting on it, else Mo / S2 by composition.

    Composition alone is not enough: a Mo_S2 antisite or Mo interstitial is a lone-Mo column too.
    """
    for c in cols:
        on = [d['type'] for d in defects
              if np.linalg.norm(min_image(np.subtract(c['xy_A'], d['xy_A']), L)) < tol]
        if on:
            c['class'] = on[0]
        else:
            c['class'] = {(('Mo', 1),): 'Mo', (('S', 2),): 'S2'}.get(tuple(sorted(c['species'].items())), 'unknown')


def make_mask(shape, px, cols, defects):
    rows, ccols = np.mgrid[0:shape[0], 0:shape[1]]
    mask = np.zeros(shape, np.uint8)

    def disk(xy, cid):
        dx = min_image(ccols - xy[0] / px[0], shape[1]) * px[0]
        dy = min_image(rows - xy[1] / px[1], shape[0]) * px[1]
        mask[dx ** 2 + dy ** 2 <= MASK_RADIUS ** 2] = cid

    for c in cols:
        if c['class'] in ('Mo', 'S2'):
            disk(c['xy_A'], CLASS_ID[c['class']])
    for d in defects:  # includes vacancies, which have no column
        disk(d['xy_A'], CLASS_ID[d['type']])
    return mask


def simulate(atoms, args):
    target = FrozenPhonons(atoms, args.phonons, sigmas=0.1) if args.phonons else atoms
    extent = np.diag(atoms.cell)[:2]
    gpts = [int(np.ceil(e / args.potential_sampling / 32) * 32) for e in extent]
    potential = Potential(target, gpts=gpts, projection='finite', slice_thickness=1,
                          parametrization='kirkland', device=args.device).build(pbar=False)
    probe = Probe(energy=60e3, semiangle_cutoff=30, Cs=0, defocus=0, device=args.device)
    probe.grid.match(potential)
    scan = GridScan(start=[0, 0], end=potential.extent, sampling=probe.ctf.nyquist_sampling * .9)
    measurement = probe.scan(scan, FlexibleAnnularDetector(), potential, pbar=False)
    return measurement.integrate(80, 200).interpolate(args.pixel_size)


def to_png(array, path):
    a = array.T  # abTEM arrays are (x, y); images are (rows=y, cols=x)
    a = (a - a.min()) / max(a.max() - a.min(), 1e-12)
    Image.fromarray((a * 255).round().astype(np.uint8)).save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--vasp', default='mos2-B_Defect-Free.vasp', help='defect-free MoS2 cell (see README)')
    ap.add_argument('--out', default='dataset')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-defects', type=int, default=6)
    ap.add_argument('--pristine-fraction', type=float, default=0.1)
    ap.add_argument('--doses', type=float, nargs='+', default=[1e7, 1e6, 1.5e5, 5e4, 1e4], help='e/A^2')
    ap.add_argument('--phonons', type=int, default=0, help='frozen phonon configs (0 = off; cost x N)')
    ap.add_argument('--blur', type=float, default=0.5, help='Gaussian filter sigma, A')
    ap.add_argument('--pixel-size', type=float, default=0.04, help='output image pixel size, A')
    ap.add_argument('--potential-sampling', type=float, default=0.05, help='A; 0.05 keeps 200 mrad unaliased')
    ap.add_argument('--device', default='gpu', choices=['gpu', 'cpu'])
    args = ap.parse_args()

    if not os.path.exists(args.vasp):
        raise SystemExit(f'{args.vasp} not found; see the README for where to get the MoS2 structure')
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'classes.json'), 'w') as f:
        json.dump({'classes': CLASSES, 'mask_radius_A': MASK_RADIUS}, f, indent=1)
    pristine = build_pristine(args.vasp)
    manifest_path = os.path.join(args.out, 'manifest.csv')
    rows = list(csv.DictReader(open(manifest_path))) if os.path.exists(manifest_path) else []
    done = {r['sample'] for r in rows}

    for n in range(args.n):
        name = f'sample_{n:04d}'
        if name in done:
            continue
        t0 = time.time()
        rng = np.random.default_rng([args.seed, n])
        n_def = 0 if rng.random() < args.pristine_fraction else int(rng.integers(1, args.max_defects + 1))
        atoms, defects = add_defects(pristine, rng, n_def)
        cols = columns(atoms)
        label_columns(cols, defects, np.diag(atoms.cell)[:2])
        assert not any(c['class'] == 'unknown' for c in cols), cols

        clean = simulate(atoms, args)
        px = [float(c.sampling) for c in clean.calibrations]  # (x, y); rounding makes them differ slightly
        assert all(c.offset == 0 for c in clean.calibrations)
        blurred = clean.gaussian_filter(args.blur) if args.blur else clean

        d = os.path.join(args.out, name)
        os.makedirs(d, exist_ok=True)
        write(os.path.join(d, 'structure.vasp'), atoms, format='vasp', direct=True, sort=True)
        np.save(os.path.join(d, 'clean.npy'), clean.array.T.astype(np.float32))
        to_png(blurred.array, os.path.join(d, 'dose_inf.png'))
        noise_seeds = {}
        for k, dose in enumerate(args.doses):
            noise_seeds[f'{dose:.0f}'] = seed = args.seed * 1_000_000 + n * 100 + k
            np.random.seed(seed)  # poisson_noise draws from the global NumPy RNG
            to_png(poisson_noise(blurred, dose).array, os.path.join(d, f'dose_{dose:.0f}.png'))

        shape = clean.array.T.shape
        Image.fromarray(make_mask(shape, px, cols, defects)).save(os.path.join(d, 'mask.png'))
        for c in cols:
            c['xy_px'] = [c['xy_A'][0] / px[0], c['xy_A'][1] / px[1]]
        for r in defects:
            r['xy_px'] = [r['xy_A'][0] / px[0], r['xy_A'][1] / px[1]]
        labels = {
            'sample': name, 'pixel_size_xy_A': px, 'shape_rows_cols': list(shape),
            'pixel_convention': 'col = x / pixel_size_x, row = y / pixel_size_y, row 0 at top of PNG',
            'periodic': True, 'columns': cols, 'defects': defects,
            'simulation': {'code': 'abTEM 1.0.0b32 multislice', 'energy_eV': 60e3, 'semiangle_mrad': 30,
                           'Cs_A': 0, 'defocus_A': 0, 'detector_mrad': [80, 200], 'slice_thickness_A': 1,
                           'potential_sampling_A': args.potential_sampling, 'phonon_configs': args.phonons,
                           'gaussian_sigma_A': args.blur},
            'doses_e_per_A2': args.doses, 'noise_seeds': noise_seeds,
        }
        with open(os.path.join(d, 'labels.json'), 'w') as f:
            json.dump(labels, f, indent=1)

        rows.append({'sample': name, 'n_defects': len(defects),
                     'defect_types': ';'.join(r['type'] for r in defects),
                     'n_atoms': len(atoms), 'rows': shape[0], 'cols': shape[1], 'pixel_size_x_A': f'{px[0]:.5f}', 'pixel_size_y_A': f'{px[1]:.5f}'})
        with open(manifest_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f'{name}: {len(defects)} defects ({rows[-1]["defect_types"] or "pristine"}) '
              f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
