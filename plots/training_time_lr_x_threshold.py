import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
from matplotlib.ticker import MultipleLocator

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

key = ['model_name', 'aggregation_algorithm', 'pruning_threshold', 'models_percentage', 'learning_rate']
merged = post.merge(
    pre[key + ['total_duration_sec']],
    on=key, suffixes=('_post', '_pre')
)

learning_rates = sorted(merged['learning_rate'].unique())
thresholds     = sorted(merged['pruning_threshold'].unique())
n_lr  = len(learning_rates)
n_thr = len(thresholds)

COLOR_PRE  = '#1565C0'
COLOR_POST = '#EF6C00'
FILL_ALPHA = 0.12

# rows = thresholds, cols = learning rates
fig, axes = plt.subplots(n_thr, n_lr,
                         figsize=(7 * n_lr, 6 * n_thr),
                         sharey=False)

# Normalise axes to always be 2-D array
axes = np.array(axes).reshape(n_thr, n_lr)

for row_idx, thr in enumerate(thresholds):
    for col_idx, lr in enumerate(learning_rates):
        ax = axes[row_idx][col_idx]

        sub = merged[
            (merged['pruning_threshold'] == thr) &
            (merged['learning_rate']     == lr)
        ]

        if sub.empty:
            ax.set_visible(False)
            continue

        time_agg = (sub.groupby('model_name')[['total_duration_sec_pre', 'total_duration_sec_post']]
                    .median() / 60)

        time_agg      = time_agg.sort_values('total_duration_sec_pre', ascending=True)
        models_sorted = time_agg.index.tolist()
        x             = np.arange(len(models_sorted))

        t_pre  = time_agg['total_duration_sec_pre'].values
        t_post = time_agg['total_duration_sec_post'].values

        # Main lines
        ax.plot(x, t_pre,  color=COLOR_PRE,  linewidth=2.2, marker='o',
                markersize=7, label='Pre-Pruning',  zorder=4)
        ax.plot(x, t_post, color=COLOR_POST, linewidth=2.2, marker='s',
                markersize=7, label='Post-Pruning', zorder=4, linestyle='--')

        # Shaded area where post < pre
        ax.fill_between(x, t_pre, t_post,
                        where=(t_post <= t_pre),
                        interpolate=True,
                        color='#2E7D32', alpha=FILL_ALPHA, label='Time saved')

        # Percentage saving annotations
        y_min_val = min(t_pre.min(), t_post.min())
        y_max_val = max(t_pre.max(), t_post.max())
        y_span    = max(y_max_val - y_min_val, 1e-4)

        for i in range(len(models_sorted)):
            saving = (t_pre[i] - t_post[i]) / t_pre[i] * 100
            if abs(saving) < 0.05:
                continue
            color = '#2E7D32' if saving > 0 else '#B71C1C'
            sign  = '▼' if saving > 0 else '▲'
            y_pos = max(t_pre[i], t_post[i]) + y_span * 0.04
            ax.annotate(f'{sign}{abs(saving):.1f}%',
                        xy=(x[i], y_pos), ha='center', va='bottom',
                        fontsize=9, color=color, fontweight='bold')

        # Vertical dotted connectors
        for i in range(len(models_sorted)):
            ax.plot([x[i], x[i]], [t_pre[i], t_post[i]],
                    color='grey', linewidth=0.8, linestyle=':', zorder=2, alpha=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ') for m in models_sorted],
                           rotation=30, ha='right', fontsize=10)
        ax.set_xlim(-0.5, x[-1] + 0.5)
        ax.set_ylim(max(0, y_min_val - y_span * 0.15),
                    y_max_val + y_span * 0.30)
        ax.yaxis.set_major_locator(MultipleLocator(1))

        ax.set_title(f'LR = {lr}  |  Threshold = {thr}', fontweight='bold', pad=12)
        ax.set_ylabel('Median Training Duration (min)')

        # Legend only on first subplot
        if row_idx == 0 and col_idx == 0:
            ax.legend(framealpha=0.9, loc='upper left', ncol=1)

# Add row labels on the left margin
for row_idx, thr in enumerate(thresholds):
    axes[row_idx][0].annotate(
        f'Threshold = {thr}',
        xy=(-0.22, 0.5), xycoords='axes fraction',
        fontsize=12, fontweight='bold', rotation=90,
        va='center', ha='center'
    )

# Add column labels on the top margin
for col_idx, lr in enumerate(learning_rates):
    axes[0][col_idx].annotate(
        f'Learning Rate = {lr}',
        xy=(0.5, 1.12), xycoords='axes fraction',
        fontsize=12, fontweight='bold',
        va='bottom', ha='center'
    )

fig.suptitle('Pre- and Post-Pruning Training Time by Model\nSplit by Learning Rate × Pruning Threshold',
             fontsize=14, fontweight='bold', y=1.03)
fig.tight_layout(w_pad=4, h_pad=5)

fig.savefig('training_time_lr_x_threshold.png', bbox_inches='tight')
print("Saved!")