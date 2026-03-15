import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
from matplotlib.patches import Patch

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 180,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

df = pd.read_csv('../2_classification_signs/csv/global_grid_search_summary.csv')
df['model_name'] = df['model_name'].str.replace('ViT_Tiny_Patch16', 'ViT Tiny', regex=False)
df['model_name'] = df['model_name'].str.replace('DeiT_Tiny',        'DeiT Tiny', regex=False)

pre  = df[df['execution_phase'] == 'Pre-Pruning']
post = df[df['execution_phase'] == 'Post-Pruning']

key = ['model_name', 'aggregation_algorithm', 'pruning_threshold',
       'models_percentage', 'learning_rate']

merged = post.merge(
    pre[key + ['best_f1']],
    on=key, suffixes=('_post', '_pre')
)

# ── Paired calculation: Δ per run BEFORE aggregating ──────────────────────────
merged['f1_delta'] = merged['best_f1_post'] - merged['best_f1_pre']

ALGO_COLORS = {
    'FedAvg':  '#1565C0',
    'FedLC':   '#6A1B9A',
    'FedProx': '#2E7D32',
}
HATCH_PATTERN = 'xxx'

thresholds = [0.5, 1.0]
algorithms = ['FedAvg', 'FedLC', 'FedProx']
thr_labels = {0.5: 'Pruning Threshold = 0.5', 1.0: 'Pruning Threshold = 1.0'}

fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=False)

for row_idx, thr in enumerate(thresholds):
    for col_idx, alg in enumerate(algorithms):
        ax = axes[row_idx][col_idx]

        ax.grid(False)

        sub = merged[
            (merged['pruning_threshold']     == thr) &
            (merged['aggregation_algorithm'] == alg)
        ]

        # Correct aggregation: median per model of quantities already computed per run
        f1_agg = sub.groupby('model_name').agg(
            f1_pre_median  = ('best_f1_pre',  'median'),
            f1_post_median = ('best_f1_post', 'median'),
            f1_delta_median= ('f1_delta',     'median'),  # ← paired
        )

        # Sort by ascending pre-pruning F1
        f1_agg = f1_agg.sort_values('f1_pre_median', ascending=True)
        models_sorted = f1_agg.index.tolist()

        x = np.arange(len(models_sorted)) * 1.1
        w = 0.30
        color = ALGO_COLORS[alg]

        ax.bar(x - w / 2 - 0.05, f1_agg['f1_pre_median'],  w,
               color=color, alpha=0.5, label='Pre-Pruning',
               zorder=3, edgecolor='black', linewidth=1)

        ax.bar(x + w / 2 + 0.05, f1_agg['f1_post_median'], w,
               color=color, alpha=0.9, label='Post-Pruning',
               hatch=HATCH_PATTERN, zorder=3, edgecolor='black', linewidth=1)

        # Annotation based on paired median(Δ)
        y_vals = np.concatenate([f1_agg['f1_pre_median'].values,
                                  f1_agg['f1_post_median'].values])
        y_min  = y_vals.min()
        y_max  = y_vals.max()
        y_span = max(y_max - y_min, 1e-4)

        for i, model in enumerate(models_sorted):
            delta_pct = f1_agg.loc[model, 'f1_delta_median'] * 100  # median(Δ) in %

            if abs(delta_pct) < 0.005:
                continue

            sign  = '▲' if delta_pct >= 0 else '▼'
            clr   = '#2E7D32' if delta_pct >= 0 else '#B71C1C'
            y_pos = max(f1_agg.loc[model, 'f1_pre_median'],
                        f1_agg.loc[model, 'f1_post_median']) + y_span * 0.04
            ax.annotate(f'{sign}{abs(delta_pct):.2f}%',
                        xy=(x[i], y_pos), ha='center', va='bottom',
                        fontsize=8.5, color=clr, fontweight='bold')

        ax.set_title(f'{alg} — {thr_labels[thr]}', fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ') for m in models_sorted],
                           rotation=30, ha='right', fontsize=9.5)
        ax.set_xlim(-0.8, x[-1] + 0.8)
        ax.set_ylim(max(0, y_min - y_span * 0.15),
                    y_max + y_span * 0.22)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.3f}'))

        if col_idx == 0:
            ax.set_ylabel('Median F1 Score')

# ── Shared legend at the bottom center ────────────────────────────────────────
legend_elements = [
    Patch(facecolor='grey', alpha=0.5, edgecolor='black', linewidth=1,
          label='Pre-Pruning'),
    Patch(facecolor='grey', alpha=0.9, edgecolor='black', linewidth=1,
          hatch=HATCH_PATTERN, label='Post-Pruning'),
]
fig.legend(handles=legend_elements,
           loc='lower center',
           ncol=2,
           fontsize=11,
           framealpha=0.9,
           bbox_to_anchor=(0.5, -0.03))

fig.suptitle('Pre- and Post-Pruning F1 Score by Model, Aggregation Algorithm and Pruning Threshold\n'
             r'$\Delta$ = median of paired differences (Post $-$ Pre)',
             fontsize=14, fontweight='bold', y=1.01)

fig.tight_layout(w_pad=3, h_pad=4)
fig.subplots_adjust(bottom=0.10)

fig.savefig('f1_by_alg_threshold.png', bbox_inches='tight')
print("Saved!")