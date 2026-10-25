"""Compare observed correlations with stored within-relation permutation nulls."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results_bqp/core_experiment/single_model_runs.csv'
OUT = ROOT / 'results_bqp/figures/bqp_appendix_a2_relation_permutation'
SEEDS = [7, 11, 19, 23, 31]
DATASETS = [('FB15k237', 'FB15k-237'), ('WN18RR', 'WN18RR'),
            ('NELL995', 'NELL-995 (derived)')]
MODELS = [('complex', 'ComplEx'), ('distmult', 'DistMult')]


def main():
    with SOURCE.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    lookup = {(r['dataset'], r['model'], int(r['seed'])): r for r in rows}
    expected = {(d, m, s) for d, _ in DATASETS for m, _ in MODELS for s in SEEDS}
    assert len(rows) == len(lookup) == 30 and set(lookup) == expected
    for r in rows:
        lo, mid, hi = [float(r[k]) for k in
                       ['relation_null_q025', 'relation_null_mean', 'relation_null_q975']]
        assert -1 <= lo <= mid <= hi <= 1
        assert -1 <= float(r['head_tail_correlation']) <= 1

    plt.rcParams.update({'font.family': 'serif',
                         'font.serif': ['Times New Roman', 'DejaVu Serif'],
                         'font.size': 8, 'axes.titlesize': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.linewidth': 0.7})
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.75), sharex=True, sharey=True)
    positions = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10])
    labels = [f'{name} / {seed}' for _, name in MODELS for seed in SEEDS]
    for panel, (ax, (dataset, title)) in enumerate(zip(axes, DATASETS)):
        ax.axvline(0, color='#888888', lw=0.7, ls=(0, (3, 3)), zorder=0)
        ax.axhspan(5, 10.6, color='#F4F4F4', zorder=0)
        for y, (model, seed) in zip(positions, [(m, s) for m, _ in MODELS for s in SEEDS]):
            r = lookup[dataset, model, seed]
            observed = float(r['head_tail_correlation'])
            mean = float(r['relation_null_mean'])
            lo, hi = float(r['relation_null_q025']), float(r['relation_null_q975'])
            ax.hlines(y, lo, hi, color='#0072B2', lw=2.3, zorder=2)
            ax.plot([lo, hi], [y, y], '|', color='#0072B2', ms=5, zorder=2)
            ax.scatter(mean, y, marker='s', s=18, facecolor='white',
                       edgecolor='#0072B2', lw=0.9, zorder=3)
            ax.scatter(observed, y, marker='o', s=20, color='#D55E00', zorder=4)
        ax.set_title(f'({chr(97 + panel)}) {title}', pad=8)
        ax.set_xlim(-0.14, 0.91)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.set_ylim(10.6, -0.6)
        ax.set_yticks(positions, labels)
        ax.set_xlabel('Head-tail correlation')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=2.5)
    handles = [Line2D([], [], color='#D55E00', marker='o', ls='None', ms=4, label='Observed'),
               Line2D([], [], color='#0072B2', marker='s', mfc='white', ls='None', ms=4, label='Null mean'),
               Line2D([], [], color='#0072B2', lw=2.3, label='Null 2.5-97.5% range')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.58, 0.99),
               ncol=3, frameon=False, columnspacing=1.2, handlelength=1.5)
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.15, top=0.80, wspace=0.12)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(OUT.with_suffix('.' + ext), dpi=600, facecolor='white')
    plt.close(fig)
    print('Validated 30 observed correlations and stored permutation summaries.')
    print(OUT)


if __name__ == '__main__':
    main()
