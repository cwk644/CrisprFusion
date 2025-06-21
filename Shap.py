# shap_analysis.py
import os, numpy as np, pandas as pd, tensorflow as tf, shap, seaborn as sns
import matplotlib.pyplot as plt

from CrisprFusion import *
from load_data_func import *
from utils            import *                 # 若 prepare_5branches_from_df 定义在 utils
from pathlib          import Path

tf.compat.v1.disable_eager_execution()
tf.keras.backend.set_learning_phase(False)   # 避免 bn/dropout 训练态
########################################
# 0. 全局配置
########################################
DATASETS   = ['xCas','eSp','HF1','SpCas9','CRISPRON','HT_Cas9','HypaCas9']
DATASETS = ['xCas']
MODEL_DIR  = Path('./model')
N_BG       = 100        # baseline 样本数
N_EXPLAIN  = 200        # 要解释的测试样本数
OUT_DIR    = Path('./shap_figs'); OUT_DIR.mkdir(exist_ok=True)

########################################
# 1. 工具函数
########################################
def build_inputs(df):
    """返回 [x1,x2,x3,x4] four-branch numpy arrays."""
    return prepare_5branches_from_df(df)   # 你脚本里已有

def plot_base_bar(shap_seq, dataset):
    """
    shap_seq : (N,23,4) One-Hot SHAP; 画柱状图并保存
    """
    mean_abs = np.mean(np.abs(shap_seq), axis=0)      # (23,4)
    df = pd.DataFrame(mean_abs, columns=list('ACGT'))
    df['pos'] = np.arange(1,24)
    df_m = df.melt(id_vars='pos', var_name='Base', value_name='Importance')

    plt.figure(figsize=(10,4))
    for i, b in enumerate('ACGT'):
        sub = df_m[df_m.Base==b]
        plt.bar(sub['pos']+(i-1.5)*0.18, sub['Importance'], width=0.17, label=b)
    plt.xticks(range(1,24)); plt.xlabel('Position'); plt.ylabel('|SHAP|')
    plt.title(f'{dataset}: per-base importance')
    plt.legend(ncol=4); plt.tight_layout()
    plt.savefig(OUT_DIR/f'{dataset}_bar.png', dpi=300); plt.close()

def plot_summary(shap_values, feat_names, dataset):
    shap.summary_plot(
        shap_values, features=None, feature_names=feat_names,
        max_display=20, show=False)
    plt.title(f'{dataset}: SHAP summary (top-40)')
    plt.tight_layout()
    plt.savefig(OUT_DIR/f'{dataset}_summary.png', dpi=300); plt.close()

########################################
# 2. 主循环
########################################
import re
for ds in DATASETS:
    print(f'\n=== {ds} ===')

    # 2-1 载入 best.fold0 模型
    model_path = MODEL_DIR/f'{ds}_0_.best.fold0'
    if not model_path.exists():
        print(f'  ! 跳过，未找到 {model_path}')
        continue
    model = tf.keras.models.load_model(model_path, compile=False)
    print('  ✓ model loaded')

    # 2-2 准备 test inputs / labels
    test_df   = load_data_func_final(ds, 'test')
    x1,x2,x3,x4 = build_inputs(test_df)
    y_test   = np.array(load_data_final(ds)[3], dtype='float32')

    # 2-3 baseline (background) & explain batch
    bg_idx = np.random.choice(len(x1), N_BG, replace=False)
    background = [x1[bg_idx], x2[bg_idx], x3[bg_idx], x4[bg_idx]]

    explain_batch = [x1[:N_EXPLAIN], x2[:N_EXPLAIN],
                     x3[:N_EXPLAIN], x4[:N_EXPLAIN]]

    # 2-4 DeepExplainer
    explainer   = shap.DeepExplainer(model, background)
    shap_vals   = explainer.shap_values(explain_batch,
                                        ranked_outputs=1,
                                        check_additivity=False)   # list length=4
    print('  ✓ SHAP computed')

    # 2-5 (a) 23×4 柱状图
    shap_vals = shap_vals[0]
    print(len(shap_vals[0]))
    shap_seq = shap_vals[0][0]          # (N,23,4)
    shap_seq = np.reshape(shap_seq,newshape=(-1,23,4))
    plot_base_bar(shap_seq, ds)
    print('    • base bar saved')

    # 2-6 (b) summary beeswarm
    #   把序列 flatten，+ shape + misc
    sv_seq  = shap_seq.reshape(N_EXPLAIN, -1)
    sv_shape= shap_vals[0][2].reshape(N_EXPLAIN, -1)
    sv_misc = shap_vals[0][3]                  # (N,16)
    full_sv = np.concatenate([sv_seq, sv_shape, sv_misc], axis=1)

    names_seq  = [f'{b}{i+1}' for i in range(23) for b in 'ACGT']
    names_shape= [f'shape{i+1}_{k}' for i in range(23)
                                   for k in ['MGW','HelT','ProT','Roll','Rise']]
    names_misc = [f'misc{i+1}' for i in range(sv_misc.shape[1])]
    feat_names = names_seq + names_shape + names_misc

    plot_summary(full_sv, feat_names, ds)
    print('    • summary plot saved')
    
    sv_misc   = shap_vals[0][3]            # ndarray
    mean_abs  = np.mean(np.abs(sv_misc), axis=0)   # (16,)

    #feat_names = [f'misc{i+1}' for i in range(mean_abs.shape[0])]
    feat_names = test_df['misc_names'].iloc[0] 
    feat_names = [n.replace('_feature1', '') for n in feat_names]

    plt.figure(figsize=(8,4))
    plt.bar(feat_names, mean_abs, color='steelblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('|SHAP|')
    plt.title(f'{ds}: misc feature importance')
    plt.tight_layout()
    plt.savefig(OUT_DIR/f'{ds}_misc_bar.png', dpi=300)
    plt.close()
    print('    • misc bar saved')
print('\nAll done — figures are in ./shap_figs/')
