from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np

class Nuc_NtsTokenizer(Tokenizer):

    def __init__(self):
        Tokenizer.__init__(self)
        self.dic = ['SOS']
        self.dic += [a for a in 'ATCG']
        self.fit_on_texts(self.dic)

class Dimer_NtsTokenizer(Tokenizer):

    def __init__(self):
        Tokenizer.__init__(self)
        self.dic = ['SOS']
        self.dic += ['SEP1']

        self.dic += [a for a in 'ATCG']
        self.dic += [a + b for a in 'ATCG' for b in 'ATCG']
        self.dic += [a + '0' for a in 'ATCG']

        self.fit_on_texts(self.dic)

def split_seqs(seq):
    t = Nuc_NtsTokenizer()

    result = 'SOS'
    lens = len(seq)

    for i in range(lens):
        result += ' ' + seq[i].upper()

    seq_result = t.texts_to_sequences([result])

    nuc_seq = pd.Series(seq_result[0]) - 1
    pos_seq = pd.Series(i for i in range(lens + 1))
    seq = pd.concat([nuc_seq, pos_seq], axis=0, ignore_index=True)

    return seq

def Dimer_split_seqs(seq):
    t = Dimer_NtsTokenizer()

    result = 'SOS'
    lens = len(seq)

    for i in range(lens):
        result += ' ' + seq[i].upper()

    # dimer_encode
    result += ' '
    result += 'SEP1'

    seq += '0'
    wt = 2
    for i in range(lens):
        result += ' ' + seq[i:i + wt].upper()

    seq_result = t.texts_to_sequences([result])

    nuc_seq = pd.Series(seq_result[0]) - 1
    pos_seq = pd.Series(i for i in range(lens + 1))

    seq = pd.concat([nuc_seq, pos_seq], axis=0, ignore_index=True)

    return seq


class Result(object):
    Best = -1

from sklearn.metrics import mean_squared_error, r2_score
import scipy as sp

def get_score_at_test(model,input,label,result,save_path):
    y_test=label
    y_test_pred = model.predict(input)
    mse = mean_squared_error(y_test, y_test_pred)
    spearmanr = sp.stats.spearmanr(y_test, y_test_pred)[0]
    r2 = r2_score(y_test, y_test_pred)
    y_test1 = y_test.reshape(-1,)
    y_test_pre1 = y_test_pred.reshape(-1,)
    pearson = sp.stats.pearsonr(y_test1, y_test_pre1)[0]
    if result.Best<spearmanr:
        result.Best = spearmanr
        model.save(save_path)
        print('best')
    return 'MSE:' + str(mse),'Spearman:' + str(spearmanr) ,'Pearson:' + str(pearson) , 'r2:' + str(r2),'Best'+str(result.Best)

def get_score_at_test_weight(model,input,label,result,save_path):
    y_test=label
    y_test_pred = model.predict(input)
    mse = mean_squared_error(y_test, y_test_pred)
    spearmanr = sp.stats.spearmanr(y_test, y_test_pred)[0]
    r2 = r2_score(y_test, y_test_pred)
    y_test1 = y_test.reshape(-1,)
    y_test_pre1 = y_test_pred.reshape(-1,)
    pearson = sp.stats.pearsonr(y_test1, y_test_pre1)[0]
    if result.Best<spearmanr:
        result.Best = spearmanr
        model.save_weights(save_path)
        print('best')
    return 'MSE:' + str(mse),'Spearman:' + str(spearmanr) ,'Pearson:' + str(pearson) , 'r2:' + str(r2),'Best'+str(result.Best)

def get_score_at_testmse(model,input,label,result,save_path):
    y_test=label
    y_test_pred = model.predict(input)
    mse = mean_squared_error(y_test, y_test_pred)
    if mse<result.Best:
        result.Best = mse
        model.save_weights(save_path)
        print('best')
    return 'MSE:' + str(mse),'Best'+str(result.Best)

def get_score_at_testwt(model,input,label):
    y_test=label
    y_test_pred = model.predict(input)
    mse = mean_squared_error(y_test, y_test_pred)
    spearmanr = sp.stats.spearmanr(y_test, y_test_pred)[0]
    r2 = r2_score(y_test, y_test_pred)
    y_test1 = y_test.reshape(-1,)
    y_test_pre1 = y_test_pred.reshape(-1,)
    pearson = sp.stats.pearsonr(y_test1, y_test_pre1)[0]
    return 'MSE:' + str(mse),'Spearman:' + str(spearmanr) ,'Pearson:' + str(pearson) , 'r2:' + str(r2)

    return 'MSE:' + str(mse),'Spearman:' + str(spearmanr) ,'Pearson:' + str(pearson) , 'r2:' + str(r2)

def get_spearman(label1,label2):
    y_test=label2
    y_test_pred = label1
    mse = mean_squared_error(y_test, y_test_pred)
    spearmanr = sp.stats.spearmanr(y_test, y_test_pred)[0]
    r2 = r2_score(y_test, y_test_pred)
    y_test1 = y_test.reshape(-1,)
    y_test_pre1 = y_test_pred.reshape(-1,)
    pearson = sp.stats.pearsonr(y_test1, y_test_pre1)[0]

    return spearmanr

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, BatchNormalization


def mlp(inputs, output_layer_activation, output_dim, output_use_bias,
        hidden_layer_num, hidden_layer_units_num, hidden_layer_activation, dropout,
        name=None, output_regularizer=None):
    if output_layer_activation == 'sigmoid' or output_layer_activation == 'tanh':
        hidden_layer_num -= 1
    x = inputs
    for l in range(hidden_layer_num):
        x = Dense(hidden_layer_units_num, activation=hidden_layer_activation)(inputs)
        x = Dropout(dropout)(x)
    if output_layer_activation == 'sigmoid' or output_layer_activation == 'tanh':
        x = Dense(hidden_layer_units_num)(x)

        x = tf.keras.layers.concatenate([x, inputs])
        x = Activation(hidden_layer_activation)(x)
        x = Dense(output_dim, use_bias=output_use_bias,
                  kernel_regularizer='l2', activity_regularizer=output_regularizer)(x)
        x = Activation(output_layer_activation, name=name)(x)
        return x
    x = Dense(output_dim, activation=output_layer_activation,
              kernel_regularizer='l2', activity_regularizer=output_regularizer,
              use_bias=output_use_bias, name=name)(x)
    return x

def load_data(dataset):
    train_x=np.load(f'./data/{dataset}_train_X.npy')
    train_y=np.load(f'./data/{dataset}_train_y.npy')
    test_x=np.load(f'./data/{dataset}_test_X.npy')
    test_y=np.load(f'./data/{dataset}_test_y.npy')
    return train_x,test_x,train_y,test_y

'''
def convert_one_hot(x):
    lens=x.shape[0]
    res=np.zeros(shape=(lens,23,4))
    for i in range(0,lens):
        for j in range(1,24):
            res[i][j-1][int(x[i][j])-2]=1
    return res
'''
def ATCG(x):
    lens=x.shape[0]
    res=np.zeros(shape=(lens,23,4))
    bio={'A':0,'C':1,'G':2,'T':3}
    for i in range(0,lens):
        tmp=str(x[i])
        for j in range(len(tmp)):
            res[i][j][bio[tmp[j]]]=1
    return res

def convert_one_hot(train_x,test_x,train_y,test_y):
    train_x=ATCG(train_x)
    test_x=ATCG(test_x)
    train_y=np.array(train_y,dtype='float64')
    test_y=np.array(test_y,dtype='float64')
    return train_x,train_y,test_x,test_y

def load_data_final(dataset):
    a,b,c,d=load_data(dataset)
    a,b,c,d=convert_one_hot(a,b,c,d)
    return a,b,c,d

def revise(x):
    lens=x.shape[0]
    x=np.reshape(x,newshape=(-1,23,4))
    k=np.argmax(x,axis=-1)
    bio={0:'A',1:'T',2:'C',3:'G'}
    m=[]
    for i in k:
        tmp=""
        for j in i:
            tmp=tmp+bio[j]
        tmp2=Dimer_split_seqs(tmp)
        m.append(tmp2)
    m=np.array(m)
    return m

def acc_rate(x,m):
    acc=0
    len=x.shape[0]
    
    label=m.predict(x)
    label=revise(label)
    x=np.reshape(x,newshape=(len,92))
    x=revise(x)
    #return label
    #f=x-label
    #return f
    for i in range(0,len):
        c=1
        for j in range(72):
            if (x[i][j]!=label[i][j]):
                c=0
                break
        acc=acc+c
    return (acc*1.0)/(len*1.0)


def acc_rate_2(x,m1,m2):
    acc=0
    len=x.shape[0]
    
    label1=m1.predict(x)
    label=m2.predict(label1)
    label=revise(label)
    x=np.reshape(x,newshape=(len,92))
    x=revise(x)
    #return label
    #f=x-label
    #return f
    for i in range(0,len):
        c=1
        for j in range(72):
            if (x[i][j]!=label[i][j]):
                c=0
                break
        acc=acc+c
    return (acc*1.0)/(len*1.0)


'''
def test(params, train_data, train_label, test_data, test_label,dataset):
    m = transformer_ont(params)
    
    path1="model/"+dataset+".h5"
    path2="model/"+dataset+"after.h5"
    path3="model/"+"decoder.h5"
    path3_5="model/"+"decoderpart2.h5"
    path4="model/"+dataset+"final.h5"
    path5="model/"+dataset+"finetune.h5"
    #np.random.seed(1337)
    
    
    c=transformer_decoder(params)
    c.load_weights(path3)
    spec_train=convert_one_hot(train_data)
    spec_test=convert_one_hot(test_data)
    before=Model(inputs=c.input,outputs=c.get_layer("middle").output)
    

    after=Decoder(params)
    after.load_weights(path3_5)

    #after=Model(inputs=c.get_layer("middle").output,outputs=c.output)
    
    #return revise(after.predict(before.predict(spec_train)))
    print("train_data's acc is:" ,acc_rate_2(spec_train,before,after))
    print("test_data's acc is:" ,acc_rate_2(spec_test,before,after))

    
    m.load_weights(path1)
    
    
    print(get_score_at_testwt(m, test_data, test_label))
    
    m.load_weights(path4)
    
    print(get_score_at_testwt(m, test_data, test_label))
    return 
'''