import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

df = pd.read_csv('../2_classification_signs/csv/global_grid_search_summary.csv')
pre  = df[df['execution_phase'] == 'Pre-Pruning'].copy()
post = df[df['execution_phase'] == 'Post-Pruning'].copy()

keys = ['model_name', 'aggregation_algorithm', 'pruning_threshold',
        'learning_rate', 'models_percentage', 'worker_id', 'num_clients']

merged = pre.merge(post, on=keys, suffixes=('_pre', '_post'))
merged['f1_drop_pct']    = (merged['best_f1_pre'] - merged['best_f1_post']) * 100
merged['data_reduction'] = merged['pruning_avg_train_reduction_pct_post']

MODELS = [
    'ResNet18', 'MobileViT_Small', 'EfficientNet_B0', 'DeiT_Tiny',
    'EdgeNeXt_Small', 'EfficientFormer_L1', 'ConvNeXt_Atto', 'ViT_Tiny_Patch16',
]
LABEL_MAP = {
    'ResNet18':           'ResNet18',
    'MobileViT_Small':    'MobileViT Small',
    'DeiT_Tiny':          'DeiT Tiny',
    'EfficientNet_B0':    'EfficientNet B0',
    'EdgeNeXt_Small':     'EdgeNeXt Small',
    'ViT_Tiny_Patch16':   'ViT Tiny',
    'ConvNeXt_Atto':      'ConvNeXt Atto',
    'EfficientFormer_L1': 'EfficientFormer L1',
}

COLOR_05 = '#E63946'   # threshold 0.5
COLOR_10 = '#457B9D'   # threshold 1.0

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.patch.set_facecolor('white')

axes_flat = axes.flatten()

for idx, model in enumerate(MODELS):
    ax = axes_flat[idx]
    ax.set_facecolor('white')

    sub = merged[merged['model_name'] == model]

    for thr, col, lab, mk in [
        (0.5, COLOR_05, 'Threshold 0.5', 'o'),
        (1.0, COLOR_10, 'Threshold 1.0', 's'),
    ]:
        s = sub[sub['pruning_threshold'] == thr]
        ax.scatter(s['data_reduction'], s['f1_drop_pct'],
                   color=col, marker=mk, s=55, alpha=0.75,
                   edgecolors='white', linewidths=0.6, label=lab, zorder=3)



    # Green zone
    ax.axhspan(-0.5, 2, color='#D4EDDA', alpha=0.4, zorder=0)
    ax.axhline(2, color='#28A745', lw=0.8, ls='--', alpha=0.5, zorder=1)

    ax.set_title(LABEL_MAP[model], fontsize=11, fontweight='bold', pad=6)
    ax.set_xlabel('Data Removed (%)', fontsize=8.5)
    ax.set_ylabel('F1 Drop (pp)', fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle='--', alpha=0.35, linewidth=0.7)
    ax.spines[['top', 'right']].set_visible(False)

    # Axis limits: auto but always show 0
    ax.set_xlim(left=-0.5)
    ax.set_ylim(bottom=-0.3)

# Shared legend
handles = [
    mpatches.Patch(color=COLOR_05, label='Threshold = 0.5'),
    mpatches.Patch(color=COLOR_10, label='Threshold = 1.0'),
    mpatches.Patch(color='#D4EDDA', alpha=0.7, label='F1 drop < 2pp  (acceptable)'),
]
fig.legend(handles=handles, loc='lower center', ncol=3,
           fontsize=9.5, framealpha=0.9, edgecolor='#ccc',
           bbox_to_anchor=(0.5, 0.01))

fig.suptitle('Pruning Trade-off: Data Removed vs F1 Drop — per Model',
             fontsize=14, fontweight='bold', y=1.01)

plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.savefig('tradeoff_reduction_f1.png',
            dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Done.")