import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
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

models  = ['YOLO11n', 'YOLO11s', 'YOLO26n', 'YOLO26s']
metrics = ['mAP@50', 'F1', 'Precision', 'Recall']
fase1 = {
    'mAP@50':    [0.5068, 0.5721, 0.5228, 0.6085],
    'F1':        [0.5597, 0.6116, 0.5630, 0.6326],
    'Precision': [0.7213, 0.7573, 0.7075, 0.7647],
    'Recall':    [0.4572, 0.5129, 0.4675, 0.5391],
}
fase2 = {
    'mAP@50':    [0.6641, 0.7293, 0.6730, 0.7450],
    'F1':        [0.6648, 0.7161, 0.6690, 0.7214],
    'Precision': [0.7398, 0.7817, 0.7351, 0.7946],
    'Recall':    [0.6036, 0.6602, 0.6136, 0.6606],
}

color_f1  = '#4C8BB5'   # blue   (Phase 1)
color_f2  = '#E07B00'   # orange (Phase 2)
delta_pos = '#2E7D32'   # dark green
delta_neg = '#B71C1C'   # dark red

bar_w = 0.30
gap   = 0.10

fig, axes = plt.subplots(2, 2, figsize=(12, 8.25), )
fig.patch.set_facecolor('white')
axes = axes.flatten()

for ax, metric in zip(axes, metrics):
    ax.grid(False)
    order         = np.argsort(fase2[metric])
    sorted_models = [models[i] for i in order]
    sorted_v1     = np.array([fase1[metric][i] for i in order])
    sorted_v2     = np.array([fase2[metric][i] for i in order])

    x = np.arange(len(sorted_models)) * 1.1

    # bars
    ax.bar(x - bar_w / 2 - gap / 2, sorted_v1, bar_w,
           color=color_f1, alpha=0.5, label='Phase 1 — Full Dataset',
           zorder=3, edgecolor='black', linewidth=1)

    ax.bar(x + bar_w / 2 + gap / 2, sorted_v2, bar_w,
           color=color_f2, alpha=0.9, label='Phase 2 — Filtered Dataset',
           zorder=3, edgecolor='black', linewidth=1)

    # axis limits
    y_vals = np.concatenate([sorted_v1, sorted_v2])
    y_min  = y_vals.min()
    y_max  = y_vals.max()
    y_span = max(y_max - y_min, 1e-4)

    ax.set_ylim(max(0, y_min - y_span * 0.15),
                y_max + y_span * 0.38)

    # value labels (horizontal, black)
    for xi, (v1, v2) in enumerate(zip(sorted_v1, sorted_v2)):
        ax.text(x[xi] - bar_w / 2 - gap / 2, v1 + y_span * 0.012,
                f'{v1:.3f}', ha='center', va='bottom',
                fontsize=10, color='black', fontweight='normal')

        ax.text(x[xi] + bar_w / 2 + gap / 2, v2 + y_span * 0.012,
                f'{v2:.3f}', ha='center', va='bottom',
                fontsize=10, color='black', fontweight='normal')

        # delta annotation
        delta_pct = (v2 - v1) * 100
        if abs(delta_pct) < 0.005:
            continue
        sign = '▲' if delta_pct >= 0 else '▼'
        clr  = delta_pos if delta_pct >= 0 else delta_neg
        y_pos = max(v1, v2) + y_span * 0.16
        ax.annotate(f'{sign}{abs(delta_pct):.1f}%',
                    xy=(x[xi], y_pos), ha='center', va='bottom',
                    fontsize=10, color=clr, fontweight='bold')

    ax.set_title(metric, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_models, rotation=30, ha='right', fontsize=10)
    ax.set_xlim(-0.8, x[-1] + 0.8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.3f}'))
    ax.set_ylabel('Score')
fig.suptitle('Pre/Post Filtering Comparison — Precise Configuration \n($\Delta$ = Phase 2 $-$ Phase 1)',
             fontsize=14, fontweight='bold', y=1.01)

p1 = mpatches.Patch(color=color_f1, alpha=0.5, label='Phase 1 — Full Dataset', edgecolor='black', linewidth=1)
p2 = mpatches.Patch(color=color_f2, alpha=0.9, label='Phase 2 — Filtered Dataset', edgecolor='black', linewidth=1)
fig.legend(handles=[p1, p2], loc='lower center', ncol=2, fontsize=11,
           framealpha=0.9, edgecolor='#aaaaaa', bbox_to_anchor=(0.5, -0.03))

fig.tight_layout(w_pad=3, h_pad=4)
fig.savefig('yolo_comparison.png', bbox_inches='tight', facecolor='white')
plt.close()
print("Saved!")