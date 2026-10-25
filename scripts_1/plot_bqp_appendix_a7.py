"""Plot analytic SE ratios for all 60 same-seed architecture comparisons."""
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results_bqp/architecture_extension/architecture_extension_results.json'
OUT = ROOT / 'results_bqp/figures/bqp_appendix_a7_architecture_se_ratio'
SEEDS = [7, 11, 19, 23, 31]
MODELS = ['complex', 'distmult', 'cp', 'rotate']
NAMES = {'complex': 'ComplEx', 'distmult': 'DistMult', 'cp': 'CP', 'rotate': 'RotatE'}
DATASETS = [('FB15k237', 'FB15k-237'), ('WN18RR', 'WN18RR')]


def main():
    data = json.loads(SOURCE.read_text(encoding='utf-8'))
    rows = data['same_seed_comparisons']
    pairs = list(itertools.combinations(MODELS, 2))
    lookup = {(r['dataset'], r['model_a'], r['model_b'], r['seed']): r for r in rows}
    expected = {(d, a, b, s) for d, _ in DATASETS for a, b in pairs for s in SEEDS}
    assert len(rows) == len(lookup) == 60 and set(lookup) == expected
    confirmed = [r for r in rows if r.get('confirmation', {}).get('confirmed_flip', False)]
    assert len(confirmed) == 1
    for r in rows:
        assert np.isclose(r['se_ratio'], r['triple_cluster_se'] /
                          r['query_independent_se'], rtol=1e-10)
    plt.rcParams.update({'font.family': 'serif',
                         'font.serif': ['Times New Roman', 'DejaVu Serif'],
                         'font.size': 8, 'axes.titlesize': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#6A5135']
    markers = ['o', 's', '^', 'D', 'P']
    offsets = np.linspace(-0.22, 0.22, 5)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.6), sharex=True, sharey=True)
    low = min(0.99, min(r['se_ratio'] for r in rows) - 0.01)
    high = max(r['se_ratio'] for r in rows) + 0.025
    for panel, (ax, (dataset, title)) in enumerate(zip(axes, DATASETS)):
        ax.axvline(1, color='#666666', linestyle=(0, (4, 3)), lw=0.9)
        for i, (a, b) in enumerate(pairs):
            if i % 2 == 1:
                ax.axhspan(i - 0.5, i + 0.5, color='#F5F5F5', zorder=0)
            for j, seed in enumerate(SEEDS):
                r = lookup[dataset, a, b, seed]
                x, y = r['se_ratio'], i + offsets[j]
                ax.scatter(x, y, s=24, color=colors[j], marker=markers[j],
                           edgecolors='white', linewidths=0.35, zorder=3)
                if r.get('confirmation', {}).get('confirmed_flip', False):
                    ax.scatter(x, y, s=90, facecolors='none', edgecolors='black', lw=1, zorder=4)
                    ax.annotate('Confirmed flip (seed 11)', xy=(x, y),
                                xytext=(1.008, y - 0.55), fontsize=7,
                                arrowprops={'arrowstyle': '-', 'color': '#333333', 'lw': 0.7})
        ax.set_title(f'({chr(97 + panel)}) {title}', pad=9)
        ax.set_yticks(range(6), [f'{NAMES[a]} - {NAMES[b]}' for a, b in pairs])
        ax.set_ylim(5.5, -0.6)
        ax.set_xlim(low, high)
        ax.set_xlabel(r'Standard error ratio $R_{\mathrm{SE}}$')
        ax.grid(axis='x', color='#E6E6E6', lw=0.5)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=3)
    handles = [Line2D([], [], color=colors[j], marker=markers[j], ls='None', ms=4,
                      label=f'Seed {s}') for j, s in enumerate(SEEDS)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.58, 0.99),
               ncol=5, frameon=False, handletextpad=0.3, columnspacing=1.1)
    fig.subplots_adjust(left=0.21, right=0.99, bottom=0.16, top=0.79, wspace=0.13)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(OUT.with_suffix('.' + ext), dpi=600, facecolor='white')
    plt.close(fig)
    print(f'Validated {len(rows)} comparisons and {len(confirmed)} confirmed flip.')
    print(f'SE ratio range: {min(r["se_ratio"] for r in rows):.6f} to {max(r["se_ratio"] for r in rows):.6f}')


if __name__ == '__main__':
    main()
