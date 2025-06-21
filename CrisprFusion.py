import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from scipy import stats
from spektral.layers import GATConv
from tensorflow.keras.layers import (
    Input, Conv1D, Conv2D, Dense, Dropout, Flatten, concatenate,
    GlobalAveragePooling2D, MultiHeadAttention, LayerNormalization, LeakyReLU, Reshape,
    Multiply,Add,LayerNormalization,GlobalAveragePooling1D,BatchNormalization,ReLU,LeakyReLU
)
from tensorflow.keras.models import Model
from utils import *
from read import *
from load_data_func import *
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
#####################
#  1) 数据准备部分  #
#####################
def ATCG_df(seq):
    """
    自定义的转换函数: 将 sgRNA 序列(23nt) 转为 (23,4,1) 的 one-hot。
    假设A->[1,0,0,0], T->[0,1,0,0], G->[0,0,1,0], C->[0,0,0,1].
    """
    mapping = {'A':0, 'C':1, 'G':2, 'T':3}
    arr = np.zeros((23,4), dtype=np.float32)
    seq = seq.upper()
    for i, ch in enumerate(seq[:23]):
        if ch in mapping:
            arr[i, mapping[ch]] = 1.0
        else:
            # 遇到未知碱基时，可以自己决定如何处理。这里示意全0
            pass
    
    arr = np.reshape(arr, newshape=(23,4,1))
    return arr

def prepare_5branches_from_df(df):
    """
    将 DataFrame df 中的列拆分为五路输入:
      Branch1: (N, 23,4,1)      -> onehot
      Branch2: (N, 23,23)       -> Sc
      Branch3: (N, 23,5)        -> DNA_shape
      Branch4: (N, D+1)         -> misc + binding_free_energy
    """
    # 1) Branch1: one-hot
    if 'onehot' not in df.columns:
        df['onehot'] = df['sgRNA'].apply(ATCG_df)  # shape (23,4,1)
    x1_list = df['onehot'].values  # object array
    x1 = np.stack(x1_list, axis=0)  # (N,23,4,1)

    # 2) Branch2: Sc
    x2_list = df['Sc'].values  # each is (23,23)
    x2 = np.stack(x2_list, axis=0) # (N,23,23)

    # 3) Branch3: DNA_shape
    x3_list = df['DNA_shape'].values
    x3 = np.stack(x3_list, axis=0) # (N,23,5)

    # 4) Branch4: misc + binding_free_energy => (D+1,)
    misc_list = df['misc'].values  # shape (N,) each (D,)
    misc_arr = np.stack(misc_list, axis=0) # (N,D)
    
    '''
    if 'binding_free_energy' in df.columns:
        bf = df['binding_free_energy'].values.reshape(-1,1)
        x4 = np.concatenate([misc_arr, bf], axis=1) # (N, D+1)
    else:
        x4 = misc_arr
    '''
    x4 = misc_arr
    # 5) Branch5: node_features for GNN (one-hot + DNA_shape)
    # x1: (N,23,4,1) -> reshape to (N,23,4)
    x1_reshaped = np.reshape(x1, (x1.shape[0], x1.shape[1], x1.shape[2]))  # (N,23,4)

    return x1,x2,x3,x4


def normalize_adjacency(A):
    """
    归一化邻接矩阵: \hat{A} = D^{-0.5} * A * D^{-0.5}
    
    参数:
    A: Tensor，形状为 (batch_size, N, N)
    
    返回:
    归一化后的邻接矩阵，形状为 (batch_size, N, N)
    """
    degrees = tf.reduce_sum(A, axis=-1)  # 计算每个节点的度，形状为 (batch_size, N)
    degrees_inv_sqrt = tf.math.pow(degrees, -0.5)  # D^{-0.5}
    degrees_inv_sqrt = tf.where(tf.math.is_inf(degrees_inv_sqrt), tf.zeros_like(degrees_inv_sqrt), degrees_inv_sqrt)  # 处理度为0的情况

    # 构建 D^{-0.5} 的对角矩阵
    D_inv_sqrt = tf.linalg.diag(degrees_inv_sqrt)  # 形状为 (batch_size, N, N)

    # 计算 \hat{A} = D^{-0.5} * A * D^{-0.5}
    A_normalized = tf.matmul(tf.matmul(D_inv_sqrt, A), D_inv_sqrt)  # 形状为 (batch_size, N, N)

    return A_normalized

#####################
#  2) 模型部分      #
#####################

def hcaf_single(seq_tok, F_list, d_model=128, num_heads=4, tf25=True):
    """
    tf25=True → 仅返回融合向量 Z（无权重）；False → TF≥2.6 可同时返回 W_att
    """
    Q  = tf.expand_dims(seq_tok, 1)          # (B,1,128)
    KV = tf.stack(F_list, axis=1)            # (B,3,128)

    mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads,
                             output_shape=d_model)
    Z = mha(Q, KV, KV)                   # 只会返回 (B,1,128)
    Z = tf.squeeze(Z, 1)                 # -> (B,128)
    return Z, None                       # 第二个返回值占位


def CrisprFusion():
    """
    五分支模型的实现，包含：
      - Branch1: (23,4,1) 序列 One-Hot 编码 + CNN
      - Branch2: (23,23) 矩阵特征 + 空间注意力 + 多尺度 CNN
      - Branch3: (23,5) DNA shape + Transformer
      - Branch4: (16,) 标量序列 + Dense layers
    
    最后通过直接拼接五个分支的特征来进行特征融合，输出预测结果。
    """
    
    # --- Branch 1: One-Hot CNN (23,4,1)
    input1 = Input(shape=(23,4,1), name="branch1_onehot")
    conv1_1 = Conv2D(128, (1,1), padding='same', activation='relu', name="branch1_conv1_1")(input1)
    conv1_2 = Conv2D(128, (2,4), padding='same', activation='relu', name="branch1_conv1_2")(conv1_1)
    conv1_3 = Conv2D(128, (3,4), padding='same', activation='relu', name="branch1_conv1_3")(conv1_2)
    merge1 = concatenate([conv1_1, conv1_2, conv1_3], axis=-1, name="branch1_merge")
    
    conv1_4 = Conv2D(128, (3,4), padding='same', name="branch1_conv1_4")(merge1)
    conv1_4 = LeakyReLU(alpha=0.1, name="branch1_leakyrelu1_4")(conv1_4)
    
    conv1_5 = Conv2D(128, (3,4), padding='same', name="branch1_conv1_5")(conv1_4)
    conv1_5 = LeakyReLU(alpha=0.1, name="branch1_leakyrelu1_5")(conv1_5)
    
    merge2 = concatenate([conv1_4, conv1_5], axis=-1, name="branch1_merge2")
    pool1 = Flatten()(merge2)
    branch1_output = Dense(128, activation='sigmoid', name="branch1_dense1")(pool1)  # 128
    
    # --- Branch 2: 2D Conv Flow (23,23)
    input2 = Input(shape=(23, 23), name="branch2_sc")
    input2_reshaped = Reshape((23, 23, 1), name="branch2_reshape")(input2)
    
    def spatial_attention(x):
        attn_weights = tf.nn.softmax(x, axis=1, name='spatial_attention')
        attn_output = x * attn_weights
        return attn_output

    attn2 = spatial_attention(input2_reshaped)
    conv2_1 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', name="branch2_conv2_1")(attn2)
    conv2_2 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same', name="branch2_conv2_2")(conv2_1)
    conv2_3 = Conv2D(128, kernel_size=(5,5), activation='relu', padding='same', name="branch2_conv2_3")(conv2_1)
    merge2_b2 = concatenate([conv2_2, conv2_3], axis=-1, name="branch2_merge")
    
    pool2 = GlobalAveragePooling2D()(merge2_b2)  # (192,)
    branch2_output = Dense(128, activation='sigmoid', name="branch2_dense2")(pool2)  # 128
    
    # --- Branch 3: 1D CNN + Transformer (23,5)
    input3 = Input(shape=(23, 5), name="branch3_dna_shape")
    conv3_qk = Conv1D(128, kernel_size=3, activation='relu', padding='same', name="branch3_conv3_qk")(input3)
    conv3_v = Conv1D(128, kernel_size=3, activation='relu', padding='same', name="branch3_conv3_v")(input3)
    
    attn_output = MultiHeadAttention(num_heads=4, key_dim=32, name="branch3_mha")(conv3_qk, conv3_v, conv3_qk)
    attn_output = LayerNormalization(epsilon=1e-6, name="branch3_layernorm")(attn_output + conv3_qk)
    
    pool3 = tf.keras.layers.GlobalAveragePooling1D(name="branch3_pool3")(attn_output)
    #flat3 = Flatten()(attn_output)
    branch3_output = Dense(128, activation='sigmoid', name="branch3_dense3")(pool3)  # 128

    # --- Branch 4: misc + binding_free_energy Dense (17,)
    input4 = Input(shape=(16,), name="branch4_misc")
    
    branch4_output = Dense(128, activation='sigmoid', name="branch4_dense6")(input4)  # 128


    # --- 获得三个分支的 pooled 向量 ---
    b1_tok = branch1_output         # (B,128)
    b2_vec = branch2_output         # (B,128)
    b3_vec = branch3_output
    b4_vec = branch4_output         # 
    
    Z_cross, W_att = hcaf_single(b1_tok, [b2_vec, b3_vec, b4_vec])

    # 动态门控 α_i
    gate_logits = Dense(3)(concatenate([b1_tok, b2_vec, b3_vec, b4_vec]))
    alpha = tf.nn.softmax(gate_logits, axis=-1)      # (B,3)
    Z_gate = tf.reduce_sum(alpha[:, :, None] * tf.stack([b2_vec, b3_vec, b4_vec],1), axis=1)
    
    # 拼进原 fusion
    fusion = Add()([b1_tok,Z_cross, Z_gate])
    
    # 更细粒度的fusion
    # ─── Branch-1  : merge2 (B,23,1,256) → 64d ───────────
    b1_tok = Conv2D(64, (1,4),padding='valid', activation='sigmoid', name='b1_tok_conv')(merge2)
    b1_tok = tf.squeeze(b1_tok, 2)               # (B,23,64)

    # ─── Branch-2  : merge2_b2 (B,23,23,192) → 压列 → 64d ─
    b2_mid = Conv2D(64, (1,23), padding='valid',activation='sigmoid', name='b2_col_conv')(merge2_b2)
    b2_tok = tf.squeeze(b2_mid, 2)               # (B,23,64)

    # ─── Branch-3  : input3 (B,23,5) → 映射 64d ───────────
    b3_tok = Conv1D(64, 1, activation='sigmoid', name='b3_tok_conv')(attn_output)  # (B,23,64)

    # ─── Branch-4  : misc (B,16) → (B,23,64) 通过 broadcast ─
    b4_base = Dense(64, activation='sigmoid', name='misc64')(input4)          # (B,64)
    b4_tok  = tf.tile(b4_base[:,None,:], [1,23,1], name='b4_tile')         # (B,23,64)

    xatt_seq_shape = MultiHeadAttention(8, 8, name='xatt_seq_shape')(b1_tok, b3_tok, b3_tok)
    xatt_seq_shape = LayerNormalization(epsilon=1e-6)(b1_tok + xatt_seq_shape)  # (B,23,64)

    xatt_seq_sc = MultiHeadAttention(8, 8, name='xatt_seq_sc')(b1_tok, b2_tok, b2_tok)
    xatt_seq_sc = LayerNormalization(epsilon=1e-6)(b1_tok + xatt_seq_sc)        # (B,23,64)

    # 逐位加总得到精细融合 token
    tok_fused = Add(name='token_add')([xatt_seq_shape, xatt_seq_sc, b4_tok])    # (B,23,64)

    fine_vec = GlobalAveragePooling1D(name='fine_pool')(tok_fused)              # (B,64)

    
    fusion = concatenate([fusion,fine_vec])
    merged_features = Flatten()(fusion)
    
    #merged_features = concatenate([branch1_output,branch2_output,branch3_output,branch4_output])
    # 通过全连接层进行特征融合
    x = Dense(256, activation='relu', name="dense7")(merged_features)  # 640 -> 256
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu', name="dense8")(x)  # 256 -> 128

    output = Dense(1, name="output")(x)  # 128 -> 1
    
    model = Model(inputs=[input1, input2, input3, input4], outputs=output, name="CrisprFusion_Fusion")
    return model


#####################
#  3) 5折训练部分   #
#####################
def get_spearman(pred, true):
    return stats.spearmanr(pred, true)[0]

def get_spearmanr(pred, true):
    return stats.spearmanr(pred, true)[0]

import tensorflow_addons as tfa

def train_5fold(train_df, test_df, dataset,ep):
    """
    假设 train_df, test_df 中含有 'sgRNA','Sc','DNA_shape','misc','binding_free_energy','label' 等列.
    做5折CV, 训练后在 test_df 上做预测, 最终spearman.
    """
    # 准备train
    x1_train, x2_train, x3_train, x4_train = prepare_5branches_from_df(train_df)
    # 准备test
    x1_test, x2_test, x3_test, x4_test = prepare_5branches_from_df(test_df)
    
    # 从 load_data_final 获取 y_train 和 y_test
    train_x, y_train, test_x, y_test = load_data_final(dataset)
    y_train = np.array(y_train, dtype='float64')
    y_test = np.array(y_test, dtype='float64')

    # KFold 交叉验证
    isKF = True # 是否使用 KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_preds = []
    model_save_prefix = f"./model/{dataset}_{i}_"

    print(f"Branch4 shape: {x4_train.shape}")
    model1 = CrisprFusion()
    model1.summary()
    for fold, (tr_idx, val_idx) in enumerate(kf.split(x1_train)):
        if (fold > 0 and isKF == False):
            break
        print(f"\n开始训练第 {fold + 1} 个fold...")
        X1_tr, X1_val = x1_train[tr_idx], x1_train[val_idx]
        X2_tr, X2_val = x2_train[tr_idx], x2_train[val_idx]
        X3_tr, X3_val = x3_train[tr_idx], x3_train[val_idx]
        X4_tr, X4_val = x4_train[tr_idx], x4_train[val_idx]
        y_tr, y_val   = y_train[tr_idx], y_train[val_idx]
        
        # 定义和编译模型
        model = CrisprFusion()
        #optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        optimizer = tfa.optimizers.AdamW(
              learning_rate=3e-4, weight_decay=1e-4, beta_1=0.9, beta_2=0.999)
        #optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9, nesterov=True)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mse','mae'])
        #model.summary()
        
        # 定义回调函数
        es = tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=25,  mode='min', 
                                              restore_best_weights=True, verbose=1,min_delta=1e-4)                  
        mc = tf.keras.callbacks.ModelCheckpoint(
            model_save_prefix + f'.best.fold{fold}',
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1
        )

        # 训练模型
        model.fit(
            [X1_tr, X2_tr, X3_tr, X4_tr],
            y_tr,
            validation_data=([X1_val, X2_val, X3_val, X4_val], y_val),
            epochs=200,
            batch_size=256,
            verbose=2,
            callbacks=[es, mc]
        )

        # 加载最佳模型并预测
        best_model = tf.keras.models.load_model(
            model_save_prefix + f'.best.fold{fold}', 
            compile=False
        )
        test_pred_fold = best_model.predict([x1_test, x2_test, x3_test, x4_test]).reshape(-1)
        fold_preds.append(test_pred_fold)

    fold_preds = np.array(fold_preds)  # shape (5, test_size)
    mean_pred = fold_preds.mean(axis=0)
    spearman_score = get_spearman(mean_pred, y_test)
    return spearman_score


#####################
#  4) 主脚本入口    #
#####################

if __name__ == "__main__":
    # 指定数据集
    datasets = ['xCas','eSp','HF1','SpCas9','CRISPRON','HT_Cas9','HypaCas9']  # 或者其他
    #datasets = ['xCas']  # 仅示例一个数据集
    #datasets= ['eSp','HF1','SpCas9','CRISPRON','HT_Cas9','HypaCas9']
    for dataset in datasets:
        iftrain = False
        ifshap = False  # 移除 Permutation Feature Importance
        if iftrain:
            res = [222222]
            train_df = load_data_func_final(dataset, 'train')
            test_df  = load_data_func_final(dataset, 'test')
            for i in range(5):
                print(f"\n处理数据集: {dataset}")
                print("训练集样本数:", train_df.shape[0])
                print("测试集样本数:", test_df.shape[0])
                score = train_5fold(train_df, test_df, dataset,i)
                print(f"Dataset {dataset} 5fold Spearman: {score:.4f}")
                restmp = "model/" + dataset + ".npy"
                res.append(score)
                c = np.array(res)
                np.save(restmp, c)
                print(f"Spearman分数已保存到 {restmp}")
        

