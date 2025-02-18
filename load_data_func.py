import os
import numpy as np
import pandas as pd
from utils import load_data

def load_npy_features(sgRNA_df, feature_name, dataset_type, dataset):
    """
    加载 .npy 文件并添加到 sgRNA_df 中的同名列里。
    假设 features 的顺序与 sgRNA_df 行顺序严格对应。
    """
    file_path = f'./features/{dataset}_{feature_name}_{dataset_type}.npy'
    if not os.path.exists(file_path):
        print(f"警告: {file_path} 不存在，跳过加载 {feature_name}。")
        return sgRNA_df

    features = np.load(file_path)
    if len(features) != len(sgRNA_df):
        print(f"警告: {feature_name} 的行数({len(features)})与 sgRNA_df({len(sgRNA_df)}) 不匹配。")

    sgRNA_df[feature_name] = features
    return sgRNA_df

def load_Sc_as_array_column(sgRNA_df, dataset, dataset_type):
    """
    加载 Sc 矩阵 (N x 23 x 23)，存到 df['Sc'] 列，每行一个 (23,23) 数组。
    """
    file_path = f'./features/{dataset}_Sc_{dataset_type}.npy'
    if not os.path.exists(file_path):
        print(f"警告: Sc 文件 {file_path} 不存在，跳过。")
        return sgRNA_df

    sc_matrix = np.load(file_path)  # shape (N, 23, 23)
    if len(sc_matrix) != len(sgRNA_df):
        print(f"警告: Sc 的行数({len(sc_matrix)}) 与 sgRNA_df ({len(sgRNA_df)}) 不匹配。")

    # 将每个 (23,23) 存为一行
    sgRNA_df['Sc'] = list(sc_matrix)
    return sgRNA_df

def load_bed_features(sgRNA_df, bed_file, feature_prefix):
    """
    加载bed文件特征并添加到 sgRNA_df 中，使用 'sgRNA' 列进行匹配。
    """
    if not os.path.exists(bed_file):
        print(f"警告: Bed 文件 {bed_file} 不存在，跳过 {feature_prefix}。")
        return sgRNA_df

    bed_df = pd.read_csv(bed_file, sep='\t', header=None)
    print(f"加载的 Bed 文件: {bed_file}")
    #print(bed_df.head())

    # bed 文件格式: chrom, start, end, sgRNA, score, strand, ...
    bed_df.columns = (
        ['chrom', 'start', 'end', 'sgRNA', 'score', 'strand']
        + [f'{feature_prefix}_feature{i}' for i in range(1, bed_df.shape[1]-6+1)]
    )

    feature_cols = ['sgRNA'] + [col for col in bed_df.columns if col.startswith(feature_prefix)]
    bed_features = bed_df[feature_cols]

    sgRNA_df = sgRNA_df.merge(bed_features, on='sgRNA', how='left')
    return sgRNA_df

def load_shape_features_inorder(
    sgRNA_df,
    dataset,
    dataset_type,
    feature_name,
    lines_per_sgRNA=1
):
    """
    不做 merge, 按行顺序将形状值放进 DataFrame。
    若某些特征(如 HelT, Roll) 只有22值, 我们自动在前面补0凑到23。
    """
    file_path = f'./bedfeatures/{dataset}/{dataset_type}/output_{feature_name}.fasta'
    if not os.path.exists(file_path):
        print(f"警告: shape 文件 {file_path} 不存在，跳过加载 {feature_name}.")
        return sgRNA_df
    
    print(f"加载的 fasta 文件: {file_path}")
    shape_data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        current_values = []
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('>'):
                # 标识符，忽略
                pass
            else:
                val_line = [float(x) for x in line.split(',') if x.strip() != '']
                current_values.extend(val_line)

                if len(current_values) // len(val_line) == lines_per_sgRNA:
                    # lines_per_sgRNA行已读完
                    # 如果不足23个值，就在前面补0
                    if len(current_values) < 23:
                        diff = 23 - len(current_values)
                        # 在前面补
                        current_values = [0.0]*diff + current_values
                        #print(f"注意: {feature_name} 只有 {23-diff} 个值, 补了 {diff} 个0在前。")
                    shape_data_list.append(current_values)
                    current_values = []

    if len(shape_data_list) != len(sgRNA_df):
        print(f"警告: {file_path} 中 {len(shape_data_list)} 条, df有 {len(sgRNA_df)} 行。")

    if not shape_data_list:
        print(f"警告: 文件 {file_path} 中没有解析到任何 shape 数据。")
        return sgRNA_df

    # 构造列
    feat_count = len(shape_data_list[0])
    col_names = [f'{feature_name}_feat{i}' for i in range(1, feat_count+1)]
    shape_vals_df = pd.DataFrame(shape_data_list, columns=col_names)

    for col in col_names:
        sgRNA_df[col] = shape_vals_df[col].values

    return sgRNA_df

def load_binding_free_energy_inorder(sgRNA_df, dataset, dataset_type):
    """
    从 'bedfeatures/{dataset}/{dataset_type}/binding_free_energy_only.tsv' 中
    顺序加载 binding_free_energy (CRISPRoff_score)，第一行是标题 (CRISPRoff_score),
    后面的每行对应一个数值。

    前提: 行数(除去首行标题) == len(sgRNA_df) 。
    加载后在 df 中新建列 'binding_free_energy'，每行一个 float 值。
    """
    file_path = f'./bedfeatures/{dataset}/{dataset_type}/binding_free_energy_only.tsv'
    if not os.path.exists(file_path):
        print(f"警告: binding_free_energy文件 {file_path} 不存在，跳过。")
        return sgRNA_df

    vals = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # 去除空行和末尾换行
        lines = [ln.strip() for ln in lines if ln.strip()]

    if not lines:
        print(f"警告: 文件 {file_path} 为空或只有空行。跳过。")
        return sgRNA_df

    # 第一行是 "CRISPRoff_score" 标题
    header = lines[0]
    if header.lower() != "crisproff_score":
        print(f"注意: {file_path} 首行为 '{header}', 并非 'CRISPRoff_score'. 依然继续读取。")

    # 从第二行开始解析float
    data_lines = lines[1:]  # 跳过第一行标题
    for ln in data_lines:
        try:
            val = float(ln)
            vals.append(val)
        except ValueError:
            print(f"警告: 解析 {ln} 失败，跳过。")

    # 检查行数是否与 sgRNA_df 对齐
    if len(vals) != len(sgRNA_df):
        print(f"警告: binding_free_energy条数 ({len(vals)}) 与 sgRNA_df行数 ({len(sgRNA_df)}) 不一致。")
        # 你可以决定 raise Error 或是用 min(...) 长度来合并
        # 这里先只打印警告
        # raise ValueError("行数不匹配，无法按顺序对齐。")

    # 若一行都没加载到, 返回
    if not vals:
        print(f"警告: {file_path} 未解析到任何 binding_free_energy 数值.")
        return sgRNA_df

    # 将其写入一列
    sgRNA_df['binding_free_energy'] = vals

    return sgRNA_df

def unify_DNA_shape_columns(df, shape_features):
    """
    将 df 中各 shape_features 的 feat1..feat23 列(有的或许被补了0)合并成 (23,5)。
    如果某特征实则 23 长度, 就正常stack; 
    如果是补0的, 也已经在 load_shape_features_inorder 那里补齐到 23.
    """
    print("正在整合DNA_shape特征")
    def row_to_23xN(row):
        # shape_features = ['EP','HelT','MGW','ProT','Roll']  => 5
        arrays = []
        for feat in shape_features:
            col_names = [f"{feat}_feat{i}" for i in range(1,24)]
            arr_23 = row[col_names].to_numpy(dtype=float)  # (23,)
            arrays.append(arr_23)
        # stack => (23, len(shape_features)) => (23,5)
        shape_23x5 = np.stack(arrays, axis=1)
        return shape_23x5

    # 逐行处理
    shape_col = []
    for i in range(len(df)):
        shape_23x5 = row_to_23xN(df.iloc[i])
        shape_col.append(shape_23x5)
    df['DNA_shape'] = shape_col
    # 假设 shape_features = ['EP','HelT','MGW','ProT','Roll']
    for feat in shape_features:
        col_names = [f"{feat}_feat{i}" for i in range(1, 24)]
        df.drop(columns=col_names, inplace=True, errors='ignore')
        
    return df

def unify_misc_features(df, ignore_cols=('sgRNA','Sc','DNA_shape')):
    """
    将除 ignore_cols 外的所有标量列合并成1D数组 => df['misc']。
    """
    print("正在整合其余特征")
    all_cols = set(df.columns)
    exclude = set(ignore_cols)
    used_cols = sorted(list(all_cols - exclude))

    def row_to_1d(row):
        return row[used_cols].to_numpy(dtype=float)

    df['misc'] = df.apply(row_to_1d, axis=1)
    return df

def load_data_func_final(dataset,k):
    train_x, test_x, train_y, test_y= load_data(dataset)
    if k == "train":
        p = train_x
    else:
        p = test_x
    
    sgRNA_df = pd.DataFrame(p, columns=['sgRNA'])

    # 1. 加载 .npy 特征
    features_to_load = ['DNA_Enthalpy','Ensemble_free_energy','MFE','Tm']
    for feature in features_to_load:
        sgRNA_df = load_npy_features(sgRNA_df, feature, k, dataset)

    # 2. 加载 Sc => (N, 23, 23)
    sgRNA_df = load_Sc_as_array_column(sgRNA_df, dataset, k)

    # 3. 加载 bed 文件特征
    bed_files = {
        'ATAC': f'bedfeatures/{dataset}/{k}/targets_with_ATAC.bed',
        'CPG': f'bedfeatures/{dataset}/{k}/targets_with_CPG.bed',
        'CTCF': f'bedfeatures/{dataset}/{k}/targets_with_CTCF.bed',
        'DNA_methylation': f'bedfeatures/{dataset}/{k}/targets_with_DNA_methylation.bed',
        'DNase-seq': f'bedfeatures/{dataset}/{k}/targets_with_DNase-seq.bed',
        'H3K24me3': f'bedfeatures/{dataset}/{k}/targets_with_H3K4me3.bed',  # actually is H3K24me3, but there was something wrong during the operation of data collection
        'H3K27ac': f'bedfeatures/{dataset}/{k}/targets_with_H3K27ac.bed',
        'LoopAnchor1': f'bedfeatures/{dataset}/{k}/targets_with_LoopAnchor1.bed',
        'LoopAnchor2': f'bedfeatures/{dataset}/{k}/targets_with_LoopAnchor2.bed',
        'POLR2A': f'bedfeatures/{dataset}/{k}/targets_with_POLR2A.bed',
        'TAD': f'bedfeatures/{dataset}/{k}/targets_with_TAD.bed'
        # 添加其他bed文件
    }
    sgRNA_df = load_binding_free_energy_inorder(sgRNA_df, dataset, k)
    
    for prefix, bed_file in bed_files.items():
        sgRNA_df = load_bed_features(sgRNA_df, bed_file, prefix)

    # 4. DNA shape: EP,HelT,MGW,ProT,Roll => 可能 HelT,Roll 只有22 => 在load时前面补0 => 23
    shape_features = ['EP','HelT','MGW','ProT','Roll']
    for feat in shape_features:
        sgRNA_df = load_shape_features_inorder(sgRNA_df, dataset, k, feat, lines_per_sgRNA=1)
    
    sgRNA_df.fillna(0, inplace=True)
    # 4.1. 将 {feat}_feat1..23 => stack => (23,5) => df['DNA_shape']
    sgRNA_df = unify_DNA_shape_columns(sgRNA_df, shape_features)

    # 4.2. 其他除('sgRNA','Sc','DNA_shape')之外 => misc
    sgRNA_df = unify_misc_features(sgRNA_df, ignore_cols=('sgRNA','Sc','DNA_shape'))
    
    print(sgRNA_df.head())
    print(f"Sc列类型: {sgRNA_df['Sc'].dtype}")
    print(f"DNA_shape列类型: {sgRNA_df['DNA_shape'].dtype}")
    print(f"misc列类型: {sgRNA_df['misc'].dtype}")

    print("特征整合完成。可以进行train_test_split等操作并在TF里使用。")
    
    return sgRNA_df
    # df.to_pickle(f'combined_{dataset}_{k}.pkl')
if __name__ == "__main__":
    dataset = 'eSp'
    k = 'test'

    sgRNA_df = load_data_func_final(dataset, k)
