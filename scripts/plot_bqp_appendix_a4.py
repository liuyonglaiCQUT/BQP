"""Plot paired interval coverage for all 65 same-seed model comparisons."""
import csv
import itertools
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results_bqp/coverage_experiment/comparison_coverage.csv'
OUT = ROOT / 'results_bqp/figures/bqp_appendix_a4_comparison_coverage'
MODELS = ['complex', 'distmult', 'cp', 'rotate']
NAMES = {'complex': 'ComplEx', 'distmult': 'DistMult', 'cp': 'CP', 'rotate': 'RotatE'}
DATASETS = [('FB15k237', 'FB15k-237'), ('WN18RR', 'WN18RR'),
            ('NELL995', 'NELL-995 (derived)')]
SEEDS = [7, 11, 19, 23, 31]


def main():
    with SOURCE.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    pairs = list(itertools.combinations(MODELS, 2))
    expected = {(d, a, b, s) for d, _ in DATASETS
                for a, b in (pairs if d != 'NELL995' else pairs[:1]) for s in SEEDS}
    actual = {(r['dataset'], r['model_a'], r['model_b'], int(r['seed'])) for r in rows}
    assert len(rows) == len(actual) == 65 and actual == expected
    for r in rows:
        assert int(r['sample_size']) == 1000 and int(r['repeats']) == 3000
        for prefix in ['query_independent', 'triple_cluster']:
            assert np.isclose(float(r[prefix + '_coverage']),
                              int(r[prefix + '_hits']) / 3000, atol=1e-12)
    plt.rcParams.update({'font.family': 'serif',
                         'font.serif': ['Times New Roman', 'DejaVu Serif'],
                         'font.size': 8, 'axes.titlesize': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#8C6D31', '#555555']
    markers = ['o', 's', '^', 'D', 'P', 'v']
    values = [100 * float(r[k]) for r in rows for k in
              ['query_independent_coverage', 'triple_cluster_coverage']]
    low = 2 * np.floor((min(values) - 0.6) / 2)
    high = max(96, np.ceil(max(values) + 0.6))
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.4), sharex=True, sharey=True)
    for ax, (dataset, title), letter in zip(axes, DATASETS, 'abc'):
        ax.plot([low, high], [low, high], color='#BBBBBB', lw=0.8, zorder=0)
        ax.axhline(95, color='#555555', lw=0.8, ls=(0, (4, 3)), zorder=0)
        ax.axvline(95, color='#555555', lw=0.8, ls=(0, (4, 3)), zorder=0)
        group = [r for r in rows if r['dataset'] == dataset]
        for i, (a, b) in enumerate(pairs):
            subset = [r for r in group if (r['model_a'], r['model_b']) == (a, b)]
            if not subset:
                continue
            x = [100 * float(r['query_independent_coverage']) for r in subset]
            y = [100 * float(r['triple_cluster_coverage']) for r in subset]
            ax.scatter(x, y, marker=markers[i], c=colors[i], s=30,
                       edgecolors='white', linewidths=0.4, alpha=0.9, zorder=3)
        ax.set_title(f'({letter}) {title}', pad=9)
        ax.text(0.04, 0.06, f'{len(group)} comparisons', transform=ax.transAxes, fontsize=8)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect('equal')
        ax.set_xticks(np.arange(np.ceil(low / 5) * 5, high + 0.1, 5))
        ax.set_yticks(np.arange(np.ceil(low / 5) * 5, high + 0.1, 5))
        ax.set_xlabel('Query-level coverage (%)')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=3)
    axes[0].set_ylabel('Source-triple-level coverage (%)')
    handles = [Line2D([], [], marker=markers[i], color=colors[i], ls='None', ms=5,
                      label=f'{NAMES[a]} - {NAMES[b]}') for i, (a, b) in enumerate(pairs)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.54, 1.0),
               ncol=3, frameon=False, columnspacing=1.7, handletextpad=0.4)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.15, top=0.74, wspace=0.16)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(OUT.with_suffix('.' + ext), dpi=600, facecolor='white')
    plt.close(fig)
    print('Validated 65 comparisons against hit counts; n=1000; repeats=3000.')
    print(OUT)


if __name__ == '__main__':
    main()
