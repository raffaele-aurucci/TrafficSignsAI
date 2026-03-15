import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 180,
})

df = pd.read_csv('../2_classification_signs/csv/global_grid_search_summary.csv')

rename_map = {
    'ViT_Tiny_Patch16':   'ViT Tiny',
    'DeiT_Tiny':          'DeiT Tiny',
    'EdgeNeXt_Small':     'EdgeNeXt Small',
    'MobileViT_Small':    'MobileViT Small',
    'EfficientNet_B0':    'EfficientNet B0',
    'ConvNeXt_Atto':      'ConvNeXt Atto',
    'EfficientFormer_L1': 'EfficientFormer L1',
    'ResNet18':           'ResNet18',
}
df['model_name'] = df['model_name'].replace(rename_map)

post = df[df['execution_phase'] == 'Post-Pruning'].copy()

model_order = sorted(post['model_name'].unique())

# Global color scale
all_vals = (post.groupby(['model_name', 'pruning_threshold', 'models_percentage'])
            ['pruning_avg_train_reduction_pct'].median())
vmin = 0
vmax = all_vals.max()

fig, axes = plt.subplots(2, 4, figsize=(20, 9))

# Leave space on the right for the shared colorbar
fig.subplots_adjust(right=0.88, wspace=0.15, hspace=0.25)

# Shared colorbar axis
cbar_ax = fig.add_axes([0.91, 0.15, 0.018, 0.7])

for idx, model in enumerate(model_order):
    ax = axes[idx // 4][idx % 4]

    sub = post[post['model_name'] == model]

    pivot = (sub.groupby(['pruning_threshold', 'models_percentage'])
             ['pruning_avg_train_reduction_pct']
             .mean()
             .unstack())

    pivot.index   = [f'thr={v}' for v in pivot.index]
    pivot.columns = [f'{int(v * 100)}% selection' for v in pivot.columns]

    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        vmin=vmin,
        vmax=vmax,
        linewidths=1.5,
        linecolor='white',
        cbar=idx == 0,
        cbar_ax=cbar_ax if idx == 0 else None,
        annot_kws={'size': 13, 'fontweight': 'bold'},
        square=True,
    )

    ax.set_title(model, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', rotation=0,  labelsize=10)
    ax.tick_params(axis='y', rotation=90, labelsize=10, pad=3)
    ax.yaxis.set_tick_params(labelrotation=90)

    # Vertically center Y tick labels
    for label in ax.get_yticklabels():
        label.set_verticalalignment('center')

cbar_ax.set_ylabel('Mean Dataset Reduction (%)', fontsize=11, labelpad=10)
cbar_ax.tick_params(labelsize=10)

fig.suptitle('Mean Training Dataset Reduction (%) per Threshold × Percentage Model Selection\n'
             '(aggregated across aggregation algorithms and learning rates)',
             fontsize=13, fontweight='bold', y=1.02)

fig.savefig('reduction_dataset.png', bbox_inches='tight')
print("Saved!")