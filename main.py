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
      Branch5_nodes: (N, 23,9)  -> node_features for GNN (one-hot + DNA_shape)
      Branch5_adj: (N,23,23)    -> adjacency matrices for GNN (Sc + sequential connections)
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
    # x3: (N,23,5)
    # Concatenate along last axis
    x5_nodes = np.concatenate([x1_reshaped, x3], axis=-1)  # (N,23,9)

    # Branch5_adj: adjacency matrices for GNN
    # 添加顺序连接 (1-2, 2-3, ...,22-23) 到 Sc 矩阵
    x5_adj = []
    for adj in x2:
        # 创建顺序连接的邻接矩阵
        sequential_adj = np.zeros_like(adj)
        for i in range(22):
            sequential_adj[i, i+1] = 1
            sequential_adj[i+1, i] = 1
        # 合并 Sc 矩阵和顺序连接
        combined_adj = adj + sequential_adj
        # 确保邻接矩阵是二值的（如果有多条边）
        combined_adj = np.where(combined_adj > 0, 1, 0)
        x5_adj.append(combined_adj)
    x5_adj = np.stack(x5_adj, axis=0)  # (N,23,23)

    return x1, x2, x3, x4, x5_nodes, x5_adj

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

def cross_attention(x,y,num_splits=4,encode_dim=128):
    x = tf.reshape(x, (-1, num_splits, encode_dim // num_splits))  
    y = tf.reshape(y, (-1, num_splits, encode_dim // num_splits))  
    # x=branch1 y=branch2~4
    attn = MultiHeadAttention(num_heads=2, key_dim=16)
    y_attn = attn(query=x, value=y, key=y)
    y_attn = LayerNormalization()(y_attn + y)
    y_conv = Conv1D(filters=32, kernel_size=4, padding='valid', activation=LeakyReLU(alpha=0.01))(y_attn) 
    # 4,32 -> 1,32
    return y_conv

def broadcast_1d(ch_up, inputs):
    """
    ch_up:  (B, 1, C)
    inputs: (B, L, C)
    返回 (B, L, C)，沿中间维 L 复制 ch_up。
    """
    # 从 inputs 里获取序列长度 L
    L = tf.shape(inputs)[1]
    # 在维度1上 tile L 倍
    # [1, L, 1] 表示: batch 不变, 序列复制L次, 通道不变
    ch_broadcast = tf.tile(ch_up, [1, L, 1])  
    return ch_broadcast

def ms_cam_1d_block(inputs, reduction=16, name=None):
    """
    MS-CAM (1D版本):
      inputs: 形状 (B, L, C)
      reduction: 通道降维比率r
      返回: X' = X * M, 同样形状 (B, L, C)
    """

    # ============ 2) 空间分支 (spatial branch) ============
    # 在 (B,L,C) 上 kernel_size=1 => 不改变 L，只是对通道做变换
    sp_down = Conv1D(filters=C//reduction, kernel_size=1, padding='valid',
                            use_bias=False,activation=LeakyReLU(alpha=0.01))(inputs)  # (B,L,C/r)
    sp_down = ReLU()(sp_down)

    sp_up = Conv1D(filters=C, kernel_size=1, padding='valid',
                          use_bias=False,activation=LeakyReLU(alpha=0.01))(sp_down)  # (B,L,C)

    # ============ 3) 合并 & 激活 (sigmoid) ============
    #merged = Add()([ch_branch, sp_up])    # (B,L,C)
    merged = sp_up
    #merged = LayerNormalization()(merged) # 可选
    att_map = tf.keras.layers.Activation('sigmoid')(merged)  # (B,L,C)

    # ============ 4) X' = X * M ============
    x_out = Multiply()([inputs, att_map]) # (B,L,C)
    return x_out

def AFF_block_1d(x, y, reduction=16, name=None):
    """
    AFF (1D):
      x, y: (B,L,C)
    """
    # X' = MS-CAM(X)
    x_att = ms_cam_1d_block(x, reduction=reduction)
    # Y' = MS-CAM(Y)
    y_att = ms_cam_1d_block(y, reduction=reduction)

    # Z = X' + Y'
    z = Add()([x_att, y_att])
    return z  # (B,L,C)

def CrisprFusion():
    """
    五分支模型的实现，包含：
      - Branch1: (23,4,1) 序列 One-Hot 编码 + CNN
      - Branch2: (23,23) 矩阵特征 + 空间注意力 + 多尺度 CNN
      - Branch3: (23,5) DNA shape + Transformer
      - Branch4: (16,) 标量序列 + Dense layers
      - Branch5: (23,9) node_features + (23,23) adjacency for GNN
    
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
    branch1_output = Dense(128, activation='relu', name="branch1_dense1")(pool1)  # 128
    
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
    branch2_output = Dense(128, activation='relu', name="branch2_dense2")(pool2)  # 128
    
    # --- Branch 3: 1D CNN + Transformer (23,5)
    input3 = Input(shape=(23, 5), name="branch3_dna_shape")
    conv3_qk = Conv1D(128, kernel_size=3, activation='relu', padding='same', name="branch3_conv3_qk")(input3)
    conv3_v = Conv1D(128, kernel_size=3, activation='relu', padding='same', name="branch3_conv3_v")(input3)
    
    attn_output = MultiHeadAttention(num_heads=4, key_dim=32, name="branch3_mha")(conv3_qk, conv3_v, conv3_qk)
    attn_output = LayerNormalization(epsilon=1e-6, name="branch3_layernorm")(attn_output + conv3_qk)
    
    pool3 = tf.keras.layers.GlobalAveragePooling1D(name="branch3_pool3")(attn_output)
    #flat3 = Flatten()(attn_output)
    branch3_output = Dense(128, activation='relu', name="branch3_dense3")(pool3)  # 128

    # --- Branch 4: misc + binding_free_energy Dense (17,)
    input4 = Input(shape=(16,), name="branch4_misc")
    b4 = Dense(128, activation='relu', name="branch4_dense4")(input4)
    b4 = Dense(64, activation='relu', name="branch4_dense5")(b4)
    branch4_output = Dense(128, activation='relu', name="branch4_dense6")(b4)  # 128
    
    # --- Branch 5: Graph Neural Network (GNN)
    # 新增两个输入: node_features 和 adjacency
    # node_features: (23,9) one-hot + DNA_shape
    input5_nodes = Input(shape=(23,9), name="branch5_node_features")
    input5_adj = Input(shape=(23,23), name="branch5_adjacency")
    
    # 归一化邻接矩阵
    adjacency_norm = normalize_adjacency(input5_adj)  # (batch_size,23,23)
    
    # GNN 层：使用两层 GATConv
    gat1 = GATConv(channels=64, 
                  activation='relu',
                  kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                  dropout_rate=0.05,
                  attn_heads=4,
                  concat_heads=False,
                  name="branch5_gat1")([input5_nodes, adjacency_norm])
    gat2 = GATConv(channels=64, 
                  activation='relu',
                  kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                  dropout_rate=0.05,
                  attn_heads=4,
                  concat_heads=False,
                  name="branch5_gat2")([gat1, adjacency_norm])
    # 输出形状: (batch_size,23,64)
    
    # 池化图特征，可以使用 GlobalAveragePooling
    graph_pool = tf.reduce_mean(gat2, axis=1)  # (batch_size,64)
    
    # Dense 层
    branch5_output = Dense(128, activation='relu', name="branch5_dense1")(graph_pool)  # 64 -> 128
    
    
    b1_self_att = cross_attention(branch1_output,branch1_output)
    b2_att = cross_attention(branch1_output,branch2_output)
    b3_att = cross_attention(branch1_output,branch3_output)
    b4_att = cross_attention(branch1_output,branch4_output)
    
    att_conc = concatenate([b1_self_att,b2_att,b3_att,b4_att],axis=1)
    
    b1_reshape = tf.reshape(branch1_output,(-1,4,32))
    
    feature_fusion = AFF_block_1d(b1_reshape,att_conc)
    feature_fusion_flatten = Flatten()(feature_fusion)
    
    #att_conc_f = Flatten()(att_conc)
    merged_features = concatenate([feature_fusion_flatten,branch5_output], axis=-1)  # (batch_size, 640)
    #merged_features = concatenate([branch1_output,branch2_output,branch3_output,branch4_output,branch5_output], axis=-1)
    # 通过全连接层进行特征融合
    x = Dense(256, activation='relu', name="dense7")(merged_features)  # 640 -> 256
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu', name="dense8")(x)  # 256 -> 128

    output = Dense(1, name="output")(x)  # 128 -> 1
    
    model = Model(inputs=[input1, input2, input3, input4, input5_nodes, input5_adj], outputs=output, name="CrisprFusion_Fusion_with_GNN")
    return model

#####################
#  3) 5折训练部分   #
#####################
def get_spearman(pred, true):
    return stats.spearmanr(pred, true)[0]

def get_spearmanr(pred, true):
    return stats.spearmanr(pred, true)[0]

def train_5fold(train_df, test_df, dataset):
    """
    假设 train_df, test_df 中含有 'sgRNA','Sc','DNA_shape','misc','binding_free_energy','label' 等列.
    做5折CV, 训练后在 test_df 上做预测, 最终spearman.
    """
    # 准备train
    x1_train, x2_train, x3_train, x4_train, x5_nodes_train, x5_adj_train = prepare_5branches_from_df(train_df)
    # 准备test
    x1_test, x2_test, x3_test, x4_test, x5_nodes_test, x5_adj_test = prepare_5branches_from_df(test_df)
    
    # 从 load_data_final 获取 y_train 和 y_test
    train_x, y_train, test_x, y_test = load_data_final(dataset)
    y_train = np.array(y_train, dtype='float64')
    y_test = np.array(y_test, dtype='float64')

    # KFold 交叉验证
    isKF = True # 是否使用 KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_preds = []
    model_save_prefix = f"./model/{dataset}"
    print(f"Branch4 shape: {x4_train.shape}")
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(x1_train)):
        if (fold > 0 and isKF == False):
            break
        print(f"\n开始训练第 {fold + 1} 个fold...")
        X1_tr, X1_val = x1_train[tr_idx], x1_train[val_idx]
        X2_tr, X2_val = x2_train[tr_idx], x2_train[val_idx]
        X3_tr, X3_val = x3_train[tr_idx], x3_train[val_idx]
        X4_tr, X4_val = x4_train[tr_idx], x4_train[val_idx]
        X5_nodes_tr, X5_nodes_val = x5_nodes_train[tr_idx], x5_nodes_train[val_idx]
        X5_adj_tr, X5_adj_val = x5_adj_train[tr_idx], x5_adj_train[val_idx]
        y_tr, y_val   = y_train[tr_idx], y_train[val_idx]
        
        # 定义和编译模型
        model = CrisprFusion()
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        #optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9, nesterov=True)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mse','mae'])
        model.summary()
        
        # 定义回调函数
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=35, verbose=1)
        mc = tf.keras.callbacks.ModelCheckpoint(
            model_save_prefix + f'.best.fold{fold}',
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1
        )

        # 训练模型
        model.fit(
            [X1_tr, X2_tr, X3_tr, X4_tr, X5_nodes_tr, X5_adj_tr],
            y_tr,
            validation_data=([X1_val, X2_val, X3_val, X4_val, X5_nodes_val, X5_adj_val], y_val),
            epochs=120,
            batch_size=100,
            verbose=2,
            callbacks=[es, mc]
        )

        # 加载最佳模型并预测
        best_model = tf.keras.models.load_model(
            model_save_prefix + f'.best.fold{fold}', 
            compile=False, 
            custom_objects={'GATConv': GATConv}
        )
        test_pred_fold = best_model.predict([x1_test, x2_test, x3_test, x4_test, x5_nodes_test, x5_adj_test]).reshape(-1)
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
    #datasets = ['HF1','SpCas9','CRISPRON','HT_Cas9','HypaCas9']
    #datasets = ['xCas','eSp']
    datasets=['xCas']
    for dataset in datasets:
        
        iftrain = False
        ifshap = False  # 移除 Permutation Feature Importance
        isplot = True
        if iftrain:
            train_df = load_data_func_final(dataset, 'train')
            test_df  = load_data_func_final(dataset, 'test')
            print(f"\n处理数据集: {dataset}")
            print("训练集样本数:", train_df.shape[0])
            print("测试集样本数:", test_df.shape[0])
            
            score = train_5fold(train_df, test_df, dataset)
            print(f"Dataset {dataset} 5fold Spearman: {score:.4f}")

            restmp = "model/" + dataset + ".npy"
            res = [222222]
            res.append(score)
            c = np.array(res)
            np.save(restmp, c)
            print(f"Spearman分数已保存到 {restmp}")


