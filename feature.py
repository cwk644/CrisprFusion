
import concurrent.futures
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import RNA
import math

from scipy import stats
from random import randint
import sys

def dot_bracket_to_adjacency_matrix(dot_bracket):
    # 获取序列的长度
    n = len(dot_bracket)
    
    # 创建一个 n x n 的邻接矩阵，初始化为 0
    adj_matrix = np.zeros((n, n), dtype=int)
    
    # 使用栈来追踪配对的括号位置
    stack = []
    
    # 遍历 Dot-Bracket 表示法，记录配对的括号
    for i, char in enumerate(dot_bracket):
        if char == '(':  # 左括号，入栈
            stack.append(i)
        elif char == ')':  # 右括号，配对
            # 弹出栈顶元素，得到配对的起始位置
            j = stack.pop()
            # 在邻接矩阵中标记这对碱基配对
            adj_matrix[i][j] = 1
            adj_matrix[j][i] = 1  # 因为 RNA 配对是双向的
    
    return adj_matrix

def get_pair_probability_matrix(sequence):
    # 使用 pf_fold 计算配对概率
    _, _ = RNA.pf_fold(sequence)
    length = len(sequence)
    # 提取配对概率矩阵
    bppm = RNA.export_bppm(sequence)

    # 将 bppm 转换为 numpy 矩阵
    pair_prob_matrix = np.zeros((length, length))
    for i in range(length):
        for j in range(i + 1, length):
            pair_prob_matrix[i][j] = bppm[i * length + j]
    
    return pair_prob_matrix


#Santa Lucia
def calculate_tm(sequence, salt_concentration=0.05):
    # Define the enthalpy (ΔH) and entropy (ΔS) for each base pair
    pair_thermo = {
        ('A', 'T'): (-7.6, -21.0),  # ΔH and ΔS for A-T
        ('T', 'A'): (-7.6, -21.0),  # ΔH and ΔS for T-A
        ('G', 'C'): (-8.0, -22.4),  # ΔH and ΔS for G-C
        ('C', 'G'): (-8.0, -22.4),  # ΔH and ΔS for C-G
        ('A', 'G'): (-6.1, -18.0),  # ΔH and ΔS for A-G
        ('G', 'A'): (-6.1, -18.0),  # ΔH and ΔS for G-A
    }

    # Calculate the total ΔH and ΔS for the RNA sequence
    delta_H = 0.0
    delta_S = 0.0
    sequence = sequence.upper()  # Ensure sequence is uppercase

    # Loop through the sequence and sum the ΔH and ΔS values
    for i in range(len(sequence) - 1):
        base1, base2 = sequence[i], sequence[i+1]
        if (base1, base2) in pair_thermo:
            dH, dS = pair_thermo[(base1, base2)]
            delta_H += dH
            delta_S += dS

    # Use the formula for melting temperature (Tm)
    # Tm = ΔH / (ΔS + R * ln(C/4)) - 273.15
    # R is 1.987 cal/(mol·K), and C is the salt concentration in mol/L
    R = 1.987  # cal/(mol·K)
    Tm = (delta_H) / (delta_S + R * math.log(salt_concentration / 4)) - 273.15
    
    return Tm

def load_feature(dataset,k):
    l = []
    tmp = np.load(f"./features/{dataset}_Sc_{k}.npy")
    l.append(tmp)
    tmp = np.load(f"./features/{dataset}_MFE_{k}.npy")
    l.append(tmp)
    tmp = np.load(f"./features/{dataset}_Ensemble_free_energy_{k}.npy")
    l.append(tmp)
    return l
            #MFE.append(c[1])
if __name__ == "__main__":
    from ParamsDetail2 import ParamsDetail

    np.random.seed(1337)
    # model = transformer_ont_biofeat(params)

    #  print("Loading weights for the models")
    #  model.load_weights("models/BestModel_WT_withbio.h5")

    ModelParam=['ModelParams_WT','ModelParams_ESP','ModelParams_HF','ModelParams_xCas',
                 'ModelParams_SniperCas','ModelParams_SpCas9','ModelParams_HypaCas9']
    
    #use one autoencoder-decoder for all datasets
    #train_decoder(decoderparams)
    datasets=['eSp','HF1','xCas','HypaCas9','SpCas9',"CRISPRON","HT_Cas9"]
    #datasets=['eSp']
    #datasets=['WT']
    
    #datasets=['chari2015Train293T','doench2016_hg19','doench2016plx_hg19','hart2016-Hct1162lib1Avg','hart2016-HelaLib1Avg','hart2016-HelaLib2Avg','hart2016-Rpe1Avg','xu2015TrainHl60']
    #datasets=['WT']
    datasets=['eSp']
    for dataset in datasets:
        res=[222222]
        needtrain=True
        #train_x,train_y,test_x,test_y= load_data(dataset)
        train_x,test_x,train_y,test_y= load_data(dataset)
        k = "test"
        if (k=="train"):
            p = train_x
        else:
            p = test_x
        
        #MFE和RNA二级结构
        
        '''
        Second_Structure = []
        MFE = []
        with open('output_sgrnas.fasta', 'w') as fasta_file:
            for i, sgrna in enumerate(p):
                # 创建Fasta格式的条目，假设每个sgRNA是一个字符串
                fasta_file.write(f">sgRNA_{i+1}\n{sgrna}\n")
                c = RNA.fold(sgrna)
                Second_Structure.append(dot_bracket_to_adjacency_matrix(c[0]))
                MFE.append(c[1])
        Second_Structure = np.array(Second_Structure)
        MFE = np.array(MFE,dtype='float64')
        np.save(f"./features/{dataset}_Sc_{k}.npy",Second_Structure)
        np.save(f"./features/{dataset}_MFE_{k}.npy",MFE)
        '''
        
        
        '''
        Ensemble_free_energy = []
        with open('output_sgrnas.fasta', 'w') as fasta_file:
            for i, sgrna in enumerate(p):
                # 创建Fasta格式的条目，假设每个sgRNA是一个字符串
                fasta_file.write(f">sgRNA_{i+1}\n{sgrna}\n")
                c = RNA.fold_compound(sgrna)
                c2= c.pf()
                Ensemble_free_energy.append(c2[1])

        Ensemble_free_energy = np.array(Ensemble_free_energy)
        np.save(f"./features/{dataset}_Ensemble_free_energy_{k}.npy",Ensemble_free_energy)
                #MFE.append(c[1])
        '''
        
        '''
        Ensemble_free_energy = []
        with open('output_sgrnas.fasta', 'w') as fasta_file:
            for i, sgrna in enumerate(p):
                # 创建Fasta格式的条目，假设每个sgRNA是一个字符串
                fasta_file.write(f">sgRNA_{i+1}\n{sgrna}\n")
                c = calculate_tm(sgrna)
                #c2= c.pf()
                Ensemble_free_energy.append(c)
        
        #Ensemble_free_energy = np.array(Ensemble_free_energy)
        #np.save(f"./features/{dataset}_Ensemble_free_energy_{k}.npy",Ensemble_free_energy)
        '''
        c = load_feature(dataset,"train")