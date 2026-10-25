"""Plot all 30 core checkpoints; no averaging or additional resampling."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results_bqp/core_experiment/single_model_runs.csv'
OUT = ROOT / 'results_bqp/figures/bqp_appendix_a1_seed_se_ratio'
SEEDS = [7, 11, 19, 23, 31]
DATASETS = [('FB15k237', 'FB15k-237'), ('WN18RR', 'WN18RR'),
            ('NELL995', 'NELL-995 (derived)')]
MODELS = [('complex', 'ComplEx'), ('distmult', 'DistMult')]


def main():
    with SOURCE.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    lookup = {(r['dataset'], r['model'], int(r['seed'])): r for r in rows}
    expected = {(d, m, s) for d, _ in DATASETS for m, _ in MODELS for s in SEEDS}
    assert len(rows) == len(lookup) == 30 and set(lookup) == expected
    for r in rows:
        ratio = float(r['analytic_se_ratio'])
        assert np.isclose(ratio, float(r['triple_cluster_se']) /
                          float(r['query_independent_se']), rtol=1e-10)

    plt.rcParams.update({'font.family': 'serif',
                         'font.serif': ['Times New Roman', 'DejaVu Serif'],
                         'font.size': 9, 'axes.titlesize': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.linewidth': 0.7})
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.9), sharey=True)
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#6A5135']
    markers = ['o', 's', '^', 'D', 'P']
    offsets = np.linspace(-0.22, 0.22, len(SEEDS))
    for panel, (ax, (dataset, title)) in enumerate(zip(axes, DATASETS)):
        ax.axhline(1, color='#555555', linewidth=0.9, linestyle=(0, (4, 3)))
        for j, seed in enumerate(SEEDS):
            values = [float(lookup[dataset, model, seed]['analytic_se_ratio'])
                      for model, _ in MODELS]
            ax.scatter(np.arange(2) + offsets[j], values, s=32,
                       marker=markers[j], color=colors[j],
                       edgecolors='white', linewidths=0.35, zorder=3,
                       label=f'Seed {seed}' if panel == 0 else None)
        ax.set_title(f'({chr(97 + panel)}) {title}', pad=9)
        ax.set_xticks([0, 1], [name for _, name in MODELS])
        ax.set_xlim(-0.48, 1.48)
        ax.set_ylim(0.98, 1.39)
        ax.set_yticks(np.arange(1, 1.401, 0.1))
        ax.grid(axis='y', color='#E5E5E5', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=3)
    axes[0].set_ylabel(r'Standard error ratio $R_{\mathrm{SE}}$')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(0.53, 1.0), ncol=5, frameon=False,
               handletextpad=0.3, columnspacing=1.2)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.16, top=0.77, wspace=0.14)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(OUT.with_suffix('.' + ext), dpi=600, facecolor='white')
    plt.close(fig)
    print('Validated and plotted 30 checkpoints; shared y-axis; analytic SE ratios.')
    print(OUT)


if __name__ == '__main__':
    main()
