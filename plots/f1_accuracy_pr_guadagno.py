import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('../2_classification_signs/csv/global_grid_search_summary.csv')
df['model_name'] = df['model_name'].str.replace('ViT_Tiny_Patch16', 'ViT Tiny', regex=False)
df['model_name'] = df['model_name'].str.replace('DeiT_Tiny',        'DeiT Tiny', regex=False)

pre  = df[df['execution_phase'] == 'Pre-Pruning']
post = df[df['execution_phase'] == 'Post-Pruning']

key = ['model_name', 'aggregation_algorithm', 'pruning_threshold',
       'models_percentage', 'learning_rate']

metrics = ['best_f1', 'best_acc', 'best_prec', 'best_recall']

# Paired merge: each row is a run with pre and post matched
merged = post.merge(
    pre[key + metrics],
    on=key, suffixes=('_post', '_pre')
)

# Compute the difference PER RUN before aggregating
for m in metrics:
    merged[f'{m}_delta'] = merged[f'{m}_post'] - merged[f'{m}_pre']

thresholds = sorted(merged['pruning_threshold'].unique())
models     = sorted(merged['model_name'].unique())

records = []
for model in models:
    for thr in thresholds:
        sub = merged[
            (merged['model_name']        == model) &
            (merged['pruning_threshold'] == thr)
        ]
        if sub.empty:
            continue

        row = {'Model': model, 'Threshold': thr, 'N runs': len(sub)}

        for m in metrics:
            deltas    = sub[f'{m}_delta']
            post_vals = sub[f'{m}_post']
            row[f'{m}_post_median'] = round(post_vals.median(), 4)
            row[f'{m}_delta_median']= round(deltas.median(),    4)
            row[f'{m}_delta_mean']  = round(deltas.mean(),      4)
            row[f'{m}_delta_std']   = round(deltas.std(),       4)
            row[f'{m}_n_negative']  = int((deltas < 0).sum())   # runs that degraded
            row[f'{m}_n_zero']      = int((deltas == 0).sum())  # runs with no change
            row[f'{m}_n_positive']  = int((deltas > 0).sum())   # runs that improved

        records.append(row)

table = pd.DataFrame(records)

pd.set_option('display.max_rows', 300)
pd.set_option('display.width', 220)
pd.set_option('display.float_format', '{:.4f}'.format)

for m in metrics:
    print(f"\n{'='*100}")
    print(f"  {m.upper()}  —  Paired Δ = Post − Pre")
    print(f"{'='*100}")
    cols = ['Model', 'Threshold', 'N runs',
            f'{m}_post_median',
            f'{m}_delta_median',
            f'{m}_delta_mean',
            f'{m}_delta_std',
            f'{m}_n_negative',
            f'{m}_n_positive',
            f'{m}_n_zero']

    sub_table = table[cols].rename(columns={
        f'{m}_post_median':  'Post (median)',
        f'{m}_delta_median': 'Δ median',
        f'{m}_delta_mean':   'Δ mean',
        f'{m}_delta_std':    'Δ std',
        f'{m}_n_negative':   'N runs↓',
        f'{m}_n_positive':   'N runs↑',
        f'{m}_n_zero':       'N runs(0)'
    })
    print(sub_table.to_string(index=False))


# ==========================================================================================================
#   BEST_F1  —  Paired Δ = Post − Pre
# ==========================================================================================================
#              Model  Threshold  N runs  Post (median)  Δ median  Δ mean  Δ std  N runs↓  N runs↑  N runs(0)
#      ConvNeXt_Atto     0.5000      18         1.0000    0.0000 -0.0082 0.0128        7        0         11
#      ConvNeXt_Atto     1.0000      18         1.0000    0.0000 -0.0064 0.0094        7        0         11
#          DeiT Tiny     0.5000      18         0.9973   -0.0027 -0.0118 0.0174        9        0          9
#          DeiT Tiny     1.0000      18         0.9973   -0.0027 -0.0085 0.0106        9        0          9
#     EdgeNeXt_Small     0.5000      18         0.9945    0.0000 -0.0110 0.0148        7        0         11
#     EdgeNeXt_Small     1.0000      18         0.9945    0.0000 -0.0070 0.0111        6        0         12
# EfficientFormer_L1     0.5000      18         1.0000    0.0000 -0.0122 0.0178        8        0         10
# EfficientFormer_L1     1.0000      18         1.0000    0.0000 -0.0073 0.0119        8        0         10
#    EfficientNet_B0     0.5000      18         0.9946   -0.0027 -0.0163 0.0255        9        2          7
#    EfficientNet_B0     1.0000      18         0.9946   -0.0027 -0.0094 0.0169        9        2          7
#    MobileViT_Small     0.5000      18         0.9945   -0.0027 -0.0187 0.0263        9        1          8
#    MobileViT_Small     1.0000      18         0.9891   -0.0054 -0.0142 0.0183       10        2          6
#           ResNet18     0.5000      18         0.9372   -0.0356 -0.0390 0.0234       18        0          0
#           ResNet18     1.0000      18         0.9430   -0.0330 -0.0353 0.0254       18        0          0
#           ViT Tiny     0.5000      18         0.9891   -0.0000 -0.0064 0.0115       12        5          1
#           ViT Tiny     1.0000      18         0.9891   -0.0000 -0.0036 0.0070        9        6          3
#
# ==========================================================================================================
#   BEST_ACC  —  Paired Δ = Post − Pre
# ==========================================================================================================
#              Model  Threshold  N runs  Post (median)  Δ median  Δ mean  Δ std  N runs↓  N runs↑  N runs(0)
#      ConvNeXt_Atto     0.5000      18         1.0000    0.0000 -0.0082 0.0127        7        0         11
#      ConvNeXt_Atto     1.0000      18         1.0000    0.0000 -0.0063 0.0094        7        0         11
#          DeiT Tiny     0.5000      18         0.9973   -0.0027 -0.0118 0.0173        9        0          9
#          DeiT Tiny     1.0000      18         0.9973   -0.0027 -0.0085 0.0106        9        0          9
#     EdgeNeXt_Small     0.5000      18         0.9946    0.0000 -0.0109 0.0147        7        0         11
#     EdgeNeXt_Small     1.0000      18         0.9946    0.0000 -0.0069 0.0110        6        0         12
# EfficientFormer_L1     0.5000      18         1.0000    0.0000 -0.0121 0.0175        8        0         10
# EfficientFormer_L1     1.0000      18         1.0000    0.0000 -0.0072 0.0118        8        0         10
#    EfficientNet_B0     0.5000      18         0.9946   -0.0027 -0.0163 0.0255        9        2          7
#    EfficientNet_B0     1.0000      18         0.9946   -0.0027 -0.0094 0.0169        9        2          7
#    MobileViT_Small     0.5000      18         0.9946   -0.0027 -0.0187 0.0263        9        1          8
#    MobileViT_Small     1.0000      18         0.9891   -0.0054 -0.0142 0.0183       10        2          6
#           ResNet18     0.5000      18         0.9375   -0.0353 -0.0386 0.0231       18        0          0
#           ResNet18     1.0000      18         0.9429   -0.0326 -0.0350 0.0252       18        0          0
#           ViT Tiny     0.5000      18         0.9891    0.0000 -0.0063 0.0114        7        3          8
#           ViT Tiny     1.0000      18         0.9891    0.0000 -0.0036 0.0070        7        2          9
#
# ==========================================================================================================
#   BEST_PREC  —  Paired Δ = Post − Pre
# ==========================================================================================================
#              Model  Threshold  N runs  Post (median)  Δ median  Δ mean  Δ std  N runs↓  N runs↑  N runs(0)
#      ConvNeXt_Atto     0.5000      18         1.0000    0.0000 -0.0076 0.0120        7        0         11
#      ConvNeXt_Atto     1.0000      18         1.0000    0.0000 -0.0059 0.0088        7        0         11
#          DeiT Tiny     0.5000      18         0.9974   -0.0026 -0.0115 0.0170        9        0          9
#          DeiT Tiny     1.0000      18         0.9974   -0.0026 -0.0080 0.0101        9        0          9
#     EdgeNeXt_Small     0.5000      18         0.9948    0.0000 -0.0104 0.0140        7        0         11
#     EdgeNeXt_Small     1.0000      18         0.9948    0.0000 -0.0066 0.0105        6        0         12
# EfficientFormer_L1     0.5000      18         1.0000    0.0000 -0.0114 0.0164        8        0         10
# EfficientFormer_L1     1.0000      18         1.0000    0.0000 -0.0070 0.0112        8        0         10
#    EfficientNet_B0     0.5000      18         0.9948   -0.0026 -0.0158 0.0248        9        2          7
#    EfficientNet_B0     1.0000      18         0.9948   -0.0026 -0.0089 0.0162        9        2          7
#    MobileViT_Small     0.5000      18         0.9948   -0.0026 -0.0184 0.0260        9        1          8
#    MobileViT_Small     1.0000      18         0.9896   -0.0052 -0.0142 0.0184       10        2          6
#           ResNet18     0.5000      18         0.9398   -0.0348 -0.0370 0.0223       18        0          0
#           ResNet18     1.0000      18         0.9441   -0.0296 -0.0340 0.0245       18        0          0
#           ViT Tiny     0.5000      18         0.9895   -0.0001 -0.0061 0.0110       11        6          1
#           ViT Tiny     1.0000      18         0.9893    0.0000 -0.0035 0.0067        8        5          5
#
# ==========================================================================================================
#   BEST_RECALL  —  Paired Δ = Post − Pre
# ==========================================================================================================
#              Model  Threshold  N runs  Post (median)  Δ median  Δ mean  Δ std  N runs↓  N runs↑  N runs(0)
#      ConvNeXt_Atto     0.5000      18         1.0000    0.0000 -0.0082 0.0127        7        0         11
#      ConvNeXt_Atto     1.0000      18         1.0000    0.0000 -0.0063 0.0094        7        0         11
#          DeiT Tiny     0.5000      18         0.9973   -0.0027 -0.0118 0.0173        9        0          9
#          DeiT Tiny     1.0000      18         0.9973   -0.0027 -0.0085 0.0106        9        0          9
#     EdgeNeXt_Small     0.5000      18         0.9946    0.0000 -0.0109 0.0147        7        0         11
#     EdgeNeXt_Small     1.0000      18         0.9946    0.0000 -0.0069 0.0110        6        0         12
# EfficientFormer_L1     0.5000      18         1.0000    0.0000 -0.0121 0.0175        8        0         10
# EfficientFormer_L1     1.0000      18         1.0000    0.0000 -0.0072 0.0118        8        0         10
#    EfficientNet_B0     0.5000      18         0.9946   -0.0027 -0.0163 0.0255        9        2          7
#    EfficientNet_B0     1.0000      18         0.9946   -0.0027 -0.0094 0.0169        9        2          7
#    MobileViT_Small     0.5000      18         0.9946   -0.0027 -0.0187 0.0263        9        1          8
#    MobileViT_Small     1.0000      18         0.9891   -0.0054 -0.0142 0.0183       10        2          6
#           ResNet18     0.5000      18         0.9375   -0.0353 -0.0386 0.0231       18        0          0
#           ResNet18     1.0000      18         0.9429   -0.0326 -0.0350 0.0252       18        0          0
#           ViT Tiny     0.5000      18         0.9891    0.0000 -0.0063 0.0114        7        3          8
#           ViT Tiny     1.0000      18         0.9891    0.0000 -0.0036 0.0070        7        2          9