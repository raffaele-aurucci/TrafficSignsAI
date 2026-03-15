import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

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
df['model_name'] = df['model_name'].str.replace('DeiT_Tiny', 'DeiT Tiny', regex=False)

pre  = df[df['execution_phase'] == 'Pre-Pruning']
post = df[df['execution_phase'] == 'Post-Pruning']

key = ['model_name', 'aggregation_algorithm', 'pruning_threshold', 'models_percentage', 'learning_rate']
merged = post.merge(
    pre[key + ['total_duration_sec']],
    on=key, suffixes=('_post', '_pre')
)

thresholds = [0.5, 1.0]
thr_titles = {0.5: 'Pruning Threshold = 0.5', 1.0: 'Pruning Threshold = 1.0'}

COLOR_PRE  = '#1565C0'   # blue  — pre-pruning
COLOR_POST = '#EF6C00'   # orange — post-pruning
FILL_ALPHA = 0.12

fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)

for ax, thr in zip(axes, thresholds):
    sub = merged[merged['pruning_threshold'] == thr]

    time_agg = (sub.groupby('model_name')[['total_duration_sec_pre', 'total_duration_sec_post']]
                .median() / 60)

    # Sort by ascending pre-pruning time
    time_agg = time_agg.sort_values('total_duration_sec_pre', ascending=True)
    models_sorted = time_agg.index.tolist()
    x = np.arange(len(models_sorted))

    t_pre  = time_agg['total_duration_sec_pre'].values
    t_post = time_agg['total_duration_sec_post'].values

    # Main lines
    ax.plot(x, t_pre,  color=COLOR_PRE,  linewidth=2.2, marker='o',
            markersize=7, label='Pre-Pruning',  zorder=4)
    ax.plot(x, t_post, color=COLOR_POST, linewidth=2.2, marker='s',
            markersize=7, label='Post-Pruning', zorder=4, linestyle='--')

    # Area between the two curves — green if post < pre, red otherwise
    ax.fill_between(x, t_pre, t_post,
                    where=(t_post <= t_pre),
                    interpolate=True,
                    color='#2E7D32', alpha=FILL_ALPHA, label='Time saved')

    # Percentage saving annotations on each point
    y_max  = max(t_pre.max(), t_post.max())
    y_span = y_max - min(t_pre.min(), t_post.min())

    for i, model in enumerate(models_sorted):
        saving = (t_pre[i] - t_post[i]) / t_pre[i] * 100
        if abs(saving) < 0.05:
            continue
        color = '#2E7D32' if saving > 0 else '#B71C1C'
        sign  = '▼' if saving > 0 else '▲'
        y_pos = max(t_pre[i], t_post[i]) + y_span * 0.04
        ax.annotate(f'{sign}{abs(saving):.1f}%',
                    xy=(x[i], y_pos), ha='center', va='bottom',
                    fontsize=9, color=color, fontweight='bold')

    # Vertical dashed segments connecting pre and post for each model
    for i in range(len(models_sorted)):
        ax.plot([x[i], x[i]], [t_pre[i], t_post[i]],
                color='grey', linewidth=0.8, linestyle=':', zorder=2, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ') for m in models_sorted],
                       rotation=30, ha='right', fontsize=10)
    ax.set_xlim(-0.5, x[-1] + 0.5)
    y_min_plot = min(t_pre.min(), t_post.min())
    ax.set_ylim(max(0, y_min_plot - y_span * 0.15), 15)

    from matplotlib.ticker import MultipleLocator
    # inside the for loop, after ax.set_ylim(...)
    ax.yaxis.set_major_locator(MultipleLocator(1))

    ax.set_title(thr_titles[thr], fontweight='bold', pad=12)
    ax.set_ylabel('Median Training Duration (min)')
    ax.legend(framealpha=0.9, loc='upper left', ncol=1)

fig.suptitle('Pre- and Post-Pruning Training Time by Model, Split by Pruning Threshold',
             fontsize=13, fontweight='bold', y=1.02)
fig.tight_layout(w_pad=4)

fig.savefig('training_time_timeline.png', bbox_inches='tight')
print("Saved!")