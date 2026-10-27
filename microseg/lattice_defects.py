"""
lattice_defects.py - lattice and point-defect analysis of an atomic-resolution image.

Nothing about the structure is assumed. Everything is learned from the
image itself:

  atoms      top-hat peak finding, which keeps dim columns a plain
             threshold would lose, optionally merged with SAM3 detections
  lattice    a1, a2 = the shortest translations that map atoms onto atoms
             of SIMILAR INTENSITY, so Te->Te counts and Te->Fe does not
  basis      atoms folded into one cell and clustered -> K basis sites
  refinement least squares on a1, a2 and every basis position at once
  motif      the image averaged over all unit cells. This is the step that
             matters: a sublattice too faint for single-atom peak finding
             emerges once N cells are averaged, since noise falls as
             sqrt(N). It is why Fe shows up in FeTe, where per-atom peak
             finding sees only Te.
  defects    site intensity vs its sublattice median -> vacancies; peaks
             on no site -> off-lattice; robust z-score -> substitutions

Single grain, approximately uniform lattice.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

DEFAULTS = dict(
    sigma=2.0,            # smoothing, px
    peak_rel=0.15,        # peak height vs the brightest atom; lower finds dimmer
    match_tol=0.30,       # match tolerance as a fraction of nearest-neighbour
    int_tol=0.30,         # "same kind" if intensities differ by less than this
    basis_min_frac=0.30,  # a basis site needs this share of the largest one
    z_anom=3.5,           # robust z for an intensity anomaly
    vac_frac=0.5,         # vacant below this fraction of the sublattice median
    motif_rel=0.05,       # faint motif peaks must clear this share of the max
    nb=48,                # unit-cell averaging grid
)


def _edge_run(mask):
    n = 0
    while n < len(mask) and mask[n]:
        n += 1
    return n


def trim_frame(a):
    """Strip uniform padding / plot furniture, which otherwise makes fake peaks."""
    g = a.std()
    if g <= 0:
        return a, np.array([0.0, 0.0])
    rs, cs = a.std(1) < 0.1 * g, a.std(0) < 0.1 * g
    y0 = _edge_run(rs) + (_edge_run(rs) > 0)
    y1 = len(rs) - _edge_run(rs[::-1]) - (_edge_run(rs[::-1]) > 0)
    x0 = _edge_run(cs) + (_edge_run(cs) > 0)
    x1 = len(cs) - _edge_run(cs[::-1]) - (_edge_run(cs[::-1]) > 0)
    if y1 - y0 < 32 or x1 - x0 < 32:
        return a, np.array([0.0, 0.0])
    return a[y0:y1, x0:x1], np.array([float(x0), float(y0)])


def autocorr_scale(x, sigma):
    """Characteristic repeat distance, from the first autocorrelation ring."""
    x = x - x.mean()
    F = np.fft.fft2(x)
    ac = np.fft.fftshift(np.real(np.fft.ifft2(F * np.conj(F))))
    ac /= ac.max()
    cy, cx = np.array(ac.shape) // 2
    pk = peak_local_max(ac, min_distance=3, threshold_rel=0.1)
    r = np.hypot(pk[:, 0] - cy, pk[:, 1] - cx)
    r = r[r > 2 * sigma + 2]
    return float(r.min()) if len(r) else min(ac.shape) / 10.0


def _dedupe(p, r):
    keep = np.ones(len(p), bool)
    for i, j in sorted(cKDTree(p).query_pairs(r)):
        if keep[i] and keep[j]:
            keep[j] = False
    return p[keep]


def analyze(image, extra_centers=None, **kw):
    """Full analysis. `image` is a 2-D float array; returns a dict."""
    p = dict(DEFAULTS)
    p.update({k: v for k, v in kw.items() if v is not None})

    im, offset = trim_frame(np.asarray(image, dtype=float))
    H, W = im.shape
    sm = gaussian_filter(im, p["sigma"])
    s0 = autocorr_scale(sm, p["sigma"])

    # top-hat removes the slow background so dim and bright columns compete
    # on equal terms
    win = max(5, int(1.5 * s0)) | 1
    tophat = sm - minimum_filter(sm, size=win)
    # s0 is the lattice period, but in a honeycomb projection (2H-MX2) the nearest
    # columns are only s0/sqrt(3) apart; 0.25*s0 let the bright metal columns
    # suppress every chalcogen peak, so the window is scaled to that spacing.
    pk = peak_local_max(tophat, min_distance=max(3, int(0.25 * s0 / np.sqrt(3))),
                        threshold_abs=0.05 * tophat.max(), exclude_border=2)
    if len(pk) == 0:
        raise ValueError("No peaks found; try lowering peak_rel.")
    h = tophat[pk[:, 0], pk[:, 1]]
    pk = pk[h >= p["peak_rel"] * np.percentile(h, 95)]
    atoms = pk[:, ::-1].astype(float)
    source = "peaks"

    if extra_centers is not None and len(extra_centers):
        api = np.asarray(extra_centers, dtype=float) - offset
        atoms = _dedupe(np.vstack([api, atoms]), 0.3 * s0)
        source = f"SAM3({len(api)}) + peaks"

    N = len(atoms)
    if N < 10:
        raise ValueError(f"Only {N} atoms found; lower peak_rel.")
    yi = np.clip(np.round(atoms[:, 1]).astype(int), 0, H - 1)
    xi = np.clip(np.round(atoms[:, 0]).astype(int), 0, W - 1)
    I = tophat[yi, xi]
    t_at = cKDTree(atoms)
    dmin = float(np.median(t_at.query(atoms, k=2)[0][:, 1]))
    tol = p["match_tol"] * dmin

    # --- primitive vectors: cluster inter-atomic difference vectors -------
    pairs = t_at.query_pairs(3.2 * dmin, output_type="ndarray")
    if len(pairs) == 0:
        raise ValueError("Atoms too sparse to infer a lattice.")
    dv = atoms[pairs[:, 1]] - atoms[pairs[:, 0]]
    dv = np.vstack([dv, -dv])
    dv = dv[np.linalg.norm(dv, axis=1) > 0.5 * dmin]
    tv = cKDTree(dv)
    cnt = tv.query_ball_point(dv, r=tol, return_length=True)
    used = np.zeros(len(dv), bool)
    cands = []
    for i in np.argsort(-cnt):
        if used[i]:
            continue
        mem = [j for j in tv.query_ball_point(dv[i], r=tol) if not used[j]]
        used[mem] = True
        if len(mem) >= 3:
            cands.append(dv[mem].mean(0))
    cands = np.array(cands)
    if len(cands) < 2:
        raise ValueError("Could not infer lattice vectors.")

    def tscore(v):
        """Share of atoms that land on an atom of similar intensity."""
        q = atoms + v
        inb = ((q[:, 0] >= 0) & (q[:, 0] < W) &
               (q[:, 1] >= 0) & (q[:, 1] < H))
        if inb.sum() < 5:
            return 0.0
        dd, j = t_at.query(q[inb])
        same = np.abs(I[inb] - I[j]) < p["int_tol"] * np.maximum(I[inb], I[j])
        return float(np.mean((dd < tol) & same))

    scores = np.array([tscore(v) for v in cands])
    good = cands[scores >= 0.75 * scores.max()]
    good = good[np.argsort(np.linalg.norm(good, axis=1))]
    a1 = good[0]
    a2 = None
    for v in good[1:]:
        cross = abs(a1[0] * v[1] - a1[1] * v[0])
        if cross / (np.linalg.norm(a1) * np.linalg.norm(v)) > 0.3:
            a2 = v
            break
    if a2 is None:
        raise ValueError("Could not find a second, non-collinear lattice vector.")
    M = np.column_stack([a1, a2])

    # --- basis ------------------------------------------------------------
    r0 = atoms[np.argmin(np.linalg.norm(atoms - [W / 2, H / 2], axis=1))]
    Fr = np.linalg.solve(M, (atoms - r0).T).T % 1.0
    df = Fr[:, None, :] - Fr[None, :, :]
    df -= np.round(df)
    near = np.linalg.norm(df @ M.T, axis=2) < tol
    unassigned = np.ones(N, bool)
    basis_frac, first = [], None
    while unassigned.any():
        c = (near & unassigned[None, :]).sum(1)
        c[~unassigned] = -1
        i = int(c.argmax())
        first = first or c[i]
        if c[i] < p["basis_min_frac"] * first:
            break
        mem = near[i] & unassigned
        w = Fr[mem] - Fr[i]
        w -= np.round(w)
        basis_frac.append((Fr[i] + w.mean(0)) % 1.0)
        unassigned[mem] = False
    K = len(basis_frac)
    P = r0 + np.array(basis_frac) @ M.T

    def ideal_sites(a1, a2, P):
        Mx = np.column_stack([a1, a2])
        corners = np.array([[0, 0], [W, 0], [0, H], [W, H]], float)
        g = np.linalg.solve(Mx, (corners - P[0]).T).T
        n = np.arange(np.floor(g[:, 0].min()) - 2, np.ceil(g[:, 0].max()) + 3)
        m = np.arange(np.floor(g[:, 1].min()) - 2, np.ceil(g[:, 1].max()) + 3)
        nn_, mm_ = [x.ravel() for x in np.meshgrid(n, m)]
        lab_nm = np.tile(np.c_[nn_, mm_], (len(P), 1))
        lab_s = np.repeat(np.arange(len(P)), len(nn_))
        pos = P[lab_s] + lab_nm[:, :1] * a1 + lab_nm[:, 1:2] * a2
        k = ((pos[:, 0] > -dmin) & (pos[:, 0] < W + dmin) &
             (pos[:, 1] > -dmin) & (pos[:, 1] < H + dmin))
        return pos[k], lab_nm[k], lab_s[k]

    def match(pos):
        dist, idx = cKDTree(pos).query(atoms)
        ok = np.zeros(N, bool)
        taken = np.zeros(len(pos), bool)
        for i in np.argsort(dist):
            if dist[i] < tol and not taken[idx[i]]:
                ok[i] = True
                taken[idx[i]] = True
        return ok, idx

    for _ in range(3):
        pos, lab_nm, lab_s = ideal_sites(a1, a2, P)
        ok, idx = match(pos)
        A = np.c_[lab_nm[idx[ok]], np.eye(K)[lab_s[idx[ok]]]]
        coef, *_ = np.linalg.lstsq(A, atoms[ok], rcond=None)
        a1, a2, P = coef[0], coef[1], coef[2:]

    pos, lab_nm, lab_s = ideal_sites(a1, a2, P)
    ok, idx = match(pos)
    rms = float(np.sqrt(np.mean(np.sum((atoms[ok] - pos[idx[ok]]) ** 2, axis=1))))
    M = np.column_stack([a1, a2])

    # --- unit-cell average: recover sublattices peak finding missed -------
    nb = int(p["nb"])
    yy, xx = np.mgrid[:H, :W]
    Fp = np.linalg.solve(M, (np.c_[xx.ravel(), yy.ravel()] - P[0]).T).T % 1.0
    ij = np.minimum((Fp * nb).astype(int), nb - 1)
    flat = ij[:, 0] * nb + ij[:, 1]
    cnt_b = np.bincount(flat, minlength=nb * nb)
    motif = (np.bincount(flat, weights=tophat.ravel(), minlength=nb * nb)
             / np.maximum(cnt_b, 1)).reshape(nb, nb)
    motif = gaussian_filter(motif, 1, mode="wrap")
    pad = nb // 4
    cell_px = float(np.sqrt(abs(np.linalg.det(M))))
    mpk = peak_local_max(np.pad(motif, pad, mode="wrap"),
                         min_distance=max(1, int(0.25 * dmin / cell_px * nb)),
                         threshold_abs=p["motif_rel"] * motif.max()) - pad
    mpk = mpk[(mpk >= 0).all(1) & (mpk < nb).all(1)]

    def pdist_px(f1, f2):
        d = np.asarray(f1) - np.asarray(f2)
        d -= np.round(d)
        return float(np.linalg.norm(M @ d))

    basis_frac = [np.linalg.solve(M, P[s] - P[0]) % 1.0 for s in range(K)]
    K0 = K
    for i, j in sorted(mpk.tolist(), key=lambda q: -motif[q[0], q[1]]):
        f = np.array([(i + 0.5) / nb, (j + 0.5) / nb])
        if all(pdist_px(f, fb) > 0.5 * dmin for fb in basis_frac):
            basis_frac.append(f)
            P = np.vstack([P, P[0] + M @ f])
    K = len(P)

    ang = float(np.degrees(np.arccos(
        np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2)))))

    # --- occupancy of every ideal site ------------------------------------
    pos, lab_nm, lab_s = ideal_sites(a1, a2, P)
    dsite = float(np.median(cKDTree(pos).query(pos, k=2)[0][:, 1]))
    inview = ((pos[:, 0] >= 0) & (pos[:, 0] <= W - 1) &
              (pos[:, 1] >= 0) & (pos[:, 1] <= H - 1))
    pos, lab_nm, lab_s = pos[inview], lab_nm[inview], lab_s[inview]
    rloc = max(2, int(0.15 * dsite))

    site_xy = pos.copy()
    site_I = np.zeros(len(pos))
    for k, (x, y) in enumerate(pos):
        x0_, y0_ = int(round(x)), int(round(y))
        ys = slice(max(y0_ - rloc, 0), min(y0_ + rloc + 1, H))
        xs = slice(max(x0_ - rloc, 0), min(x0_ + rloc + 1, W))
        w_ = tophat[ys, xs]
        if w_.size == 0:
            continue
        site_I[k] = w_.max()
        ww = w_ - w_.min()
        if ww.sum() > 0:
            Y, X = np.mgrid[ys, xs]
            site_xy[k] = [(ww * X).sum() / ww.sum(), (ww * Y).sum() / ww.sum()]

    margin = 0.5 * dsite
    inside = ((pos[:, 0] > margin) & (pos[:, 0] < W - margin) &
              (pos[:, 1] > margin) & (pos[:, 1] < H - margin))
    occ = np.zeros(len(pos), bool)
    site_med = np.zeros(K)
    for s in range(K):
        sel = lab_s == s
        ref = site_I[sel & inside] if (sel & inside).sum() >= 5 else site_I[sel]
        site_med[s] = np.median(ref) if len(ref) else 0.0
        occ[sel] = site_I[sel] >= p["vac_frac"] * site_med[s]

    atoms_all = site_xy[occ]
    site_of_atom = lab_s[occ]
    I_all = site_I[occ]
    vmask = inside & ~occ
    vac, vac_s = pos[vmask], lab_s[vmask]

    dist_off, _ = cKDTree(pos).query(atoms)
    inter = atoms[dist_off > p["match_tol"] * dsite]

    zscore = np.zeros(len(atoms_all))
    in_all = inside[occ]
    for s in range(K):
        sel = (site_of_atom == s) & in_all
        if sel.sum() < 5:
            continue
        med = np.median(I_all[sel])
        mad = max(np.median(np.abs(I_all[sel] - med)), 0.05 * abs(med), 1e-9)
        zscore[sel] = 0.6745 * (I_all[sel] - med) / mad
    anom = np.abs(zscore) > p["z_anom"]

    if len(vac):
        pr = cKDTree(vac).query_pairs(1.2 * dsite, output_type="ndarray")
        G = (coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])),
                        shape=(len(vac),) * 2) if len(pr)
             else coo_matrix((len(vac), len(vac))))
        ncl, vcl = connected_components(G, directed=False)
        hist = {int(s): int(c)
                for s, c in enumerate(np.bincount(np.bincount(vcl))) if c and s}
    else:
        ncl, vcl, hist = 0, np.array([], int), {}

    sites = []
    for s in range(K):
        n_at = int(((site_of_atom == s) & in_all).sum())
        n_vac = int((vac_s == s).sum())
        sites.append(dict(
            site=s,
            frac=[round(float(basis_frac[s][0]), 4),
                  round(float(basis_frac[s][1]), 4)],
            atoms=n_at, vacancies=n_vac,
            occupancy=round(100.0 * n_at / max(n_at + n_vac, 1), 2),
            median_intensity=round(float(site_med[s]), 2),
            from_motif=bool(s >= K0),
        ))

    return dict(
        image_shape=[int(H), int(W)], offset=offset.tolist(), source=source,
        n_peaks=int(N), nn_distance_px=round(dmin, 3),
        autocorr_px=round(s0, 3),
        a1=[round(float(a1[0]), 4), round(float(a1[1]), 4)],
        a2=[round(float(a2[0]), 4), round(float(a2[1]), 4)],
        a1_len_px=round(float(np.linalg.norm(a1)), 3),
        a2_len_px=round(float(np.linalg.norm(a2)), 3),
        angle_deg=round(ang, 3),
        K=int(K), K_from_peaks=int(K0), K_from_motif=int(K - K0),
        fit_on_lattice=int(ok.sum()), fit_rms_px=round(rms, 3),
        fit_rms_frac_nn=round(rms / dmin, 4),
        sites=sites,
        n_vacancies=int(len(vac)), vacancy_clusters=int(ncl),
        vacancy_cluster_sizes=hist,
        n_off_lattice=int(len(inter)), n_anomalies=int(anom.sum()),
        poor_fit=bool(ok.mean() < 0.8 or rms > 0.15 * dmin),
        _arrays=dict(atoms=atoms_all, site_of_atom=site_of_atom,
                     ideal=pos, vac=vac, vac_s=vac_s, inter=inter,
                     anom=anom, zscore=zscore, image=im, motif=motif),
    )
