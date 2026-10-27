"""Paper figures for the segmentation-model benchmark.

    python -m microseg.make_figures --data dataset --preds model_preds --results model_results --out figures
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image

MODEL_LABEL = {'peaks': 'Peak finder', 'atomgpt': 'AtomGPT (SAM3 + solver)', 'sam3': 'SAM3', 'microsam': r'$\mu$SAM',
               'sam1': 'SAM', 'cellpose': 'Cellpose-SAM', 'sam2': 'SAM2.1', 'gdino': 'Grounding DINO',
               'owlv2': 'OWLv2', 'florence2': 'Florence-2'}
ORDER = ['peaks', 'atomgpt', 'sam3', 'microsam', 'sam1', 'cellpose', 'sam2', 'gdino', 'owlv2', 'florence2']
DEFECT_MARK = {'V_S': 'V$_\\mathrm{S}$', 'V_S2': 'V$_\\mathrm{S2}$', 'V_Mo': 'V$_\\mathrm{Mo}$', 'Mo_S': 'Mo$_\\mathrm{S}$',
               'Mo_S2': 'Mo$_\\mathrm{S2}$', 'S_Mo': 'S$_\\mathrm{Mo}$', 'TM_dopant': 'dopant', 'TM_adatom': 'adatom',
               'Mo_i': 'Mo$_i$', 'S_i': 'S$_i$'}


def crop(ax, img, box):
    x0, x1, y0, y1 = box
    ax.imshow(img, cmap='gray', interpolation='nearest')
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])


def fig_dataset(args, sample):
    d = os.path.join(args.data, sample)
    L = json.load(open(os.path.join(d, 'labels.json')))
    classes = json.load(open(os.path.join(args.data, 'classes.json')))['classes']
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.9))
    crop(axs[0], Image.open(os.path.join(d, 'dose_inf.png')), (0, 637, 0, 551))
    for c in L['columns']:
        if c['class'] in ('Mo', 'S2'):
            axs[0].plot(*c['xy_px'], 'o', ms=4, mfc='none', mew=0.7, mec='#22d3ee' if c['class'] == 'Mo' else '#facc15')
    for r in L['defects']:
        axs[0].plot(*r['xy_px'], 's', ms=11, mfc='none', mew=1.4, mec='#ef4444')
        axs[0].annotate(DEFECT_MARK[r['type']], r['xy_px'], xytext=(7, -9), textcoords='offset points',
                        color='#ef4444', fontsize=8, weight='bold')
    axs[0].set_title('(a) noise-free + labels', fontsize=10)
    cmap = ListedColormap(['#111111', '#22d3ee', '#facc15'] + list(plt.cm.tab10.colors[:len(classes) - 3]))
    axs[1].imshow(np.array(Image.open(os.path.join(d, 'mask.png'))), cmap=cmap, vmin=0, vmax=len(classes) - 1,
                  interpolation='nearest')
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_title('(b) per-pixel class mask', fontsize=10)
    for ax, dose, t in ((axs[2], '150000', '(c) 1.5$\\times10^5$ e/Å$^2$'), (axs[3], '10000', '(d) 1$\\times10^4$ e/Å$^2$')):
        crop(ax, Image.open(os.path.join(d, f'dose_{dose}.png')), (0, 637, 0, 551))
        ax.set_title(t, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'Fig_seg_dataset.png'), dpi=200)
    plt.close(fig)


def fig_detections(args, sample, models):
    d = os.path.join(args.data, sample)
    L = json.load(open(os.path.join(d, 'labels.json')))
    img = Image.open(os.path.join(d, 'dose_inf.png'))
    fig, axs = plt.subplots(2, len(models) // 2, figsize=(3.4 * len(models) // 2, 6.6))
    for ax, m in zip(axs.ravel(), models):
        crop(ax, img, (0, 637, 0, 551))
        for c in L['columns']:
            ax.plot(*c['xy_px'], 'o', ms=3, mfc='none', mew=0.5, mec='#22d3ee' if c['class'] == 'Mo' else '#facc15')
        p = json.load(open(os.path.join(args.preds, m, f'{sample}__inf.json')))
        xy = np.asarray(p['centers'], float).reshape(-1, 2)
        if len(xy):
            ax.plot(xy[:, 0], xy[:, 1], '+', color='#ef4444', ms=6, mew=1.2)
        ax.set_title(f'{MODEL_LABEL[m]} ({len(xy)})', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'Fig_seg_detections.png'), dpi=200)
    plt.close(fig)


def fig_dose(args):
    rows = list(csv.DictReader(open(os.path.join(args.results, 'summary.csv'))))
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    styles = {m: (c, mk) for m, c, mk in zip(ORDER, plt.cm.tab10.colors, 'osD^vP*Xhp')}
    for m in ORDER:
        R = [r for r in rows if r['model'] == m]
        if not R:
            continue
        x = [3e7 if r['dose'] == 'inf' else float(r['dose']) for r in R]
        o = np.argsort(x)
        for ax, k in zip(axs, ('F1', 'recall_S2', 'defects_strict')):
            y = np.array([float(r[k]) for r in R])
            ax.plot(np.array(x)[o], y[o], marker=styles[m][1], color=styles[m][0], ms=5, lw=1.4, label=MODEL_LABEL[m])
    for ax, t in zip(axs, ('(a) atom-column F1', '(b) S$_2$-column recall', '(c) defects identified (strict)')):
        ax.set_xscale('log')
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks([1e4, 1e5, 1e6, 1e7, 3e7])
        ax.set_xticklabels(['$10^4$', '$10^5$', '$10^6$', '$10^7$', '$\\infty$'])
        ax.set_xlabel('electron dose (e/Å$^2$)')
        ax.set_title(t, fontsize=10)
        ax.grid(alpha=0.3)
    axs[2].legend(fontsize=7.5, loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'Fig_seg_dose.pdf'))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='dataset')
    ap.add_argument('--preds', default='model_preds')
    ap.add_argument('--results', default='model_results')
    ap.add_argument('--out', default='figures')
    ap.add_argument('--sample', default='sample_0008')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig_dataset(args, args.sample)
    fig_detections(args, args.sample, ['peaks', 'sam3', 'microsam', 'sam1', 'cellpose', 'sam2', 'gdino', 'owlv2'])
    fig_dose(args)


if __name__ == '__main__':
    main()
