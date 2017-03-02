# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:23:11 2017

@author: xuedy
"""

from functools import reduce
import math
import numpy as np
from operator import add
from sklearn.cross_validation import StratifiedShuffleSplit

def elystp(x):
	y = int(round(7.2179*math.log(x) - 18.21))
	return y
def hps_epo(x):
    y = int(round(-10.36*math.log(200)+64.459))
    return y
	
def trans(x):
	z = ()
	for i in x:
		z = z + (int(i),)
	return z

#转化数据格式为keras可用格式
g_count = lambda g: reduce(add, (1 for i in g))
f_common = lambda x: x

ntMap = {'A': (1, 0, 0, 0),
         'C': (0, 1, 0, 0),
         'G': (0, 0, 1, 0),
         'T': (0, 0, 0, 1)
         }

def seqCode(seq, ntMap=ntMap):
    return np.array(reduce(add, map(lambda x: ntMap[x], seq.upper()))).reshape((1, len(seq) * len(ntMap['A'])))

def channel_trans(x, cha):
    return np.array(x).reshape((1, -1, cha)).transpose((2, 0, 1))

def f_sgt(x):
    seq, score = x.strip().split('\t')
    return (seqCode(seq), np.array([float(score)]).reshape((1, 1)))

def f_sgt_cnn(x):
    seq, score = x.strip().split('\t')
    xb = channel_trans(seqCode(seq), len(ntMap['A']))
    xb = xb.reshape(1, *xb.shape)
    return (xb, np.array([float(score)]).reshape((1, 1)))

def f_sgt_rnn(x):
    seq, score = x.strip().split('\t')
    
    xb = seqCode(seq)
    xb = xb.reshape(1, -1, len(ntMap['A']))
    return(xb, np.array([float(score)]).reshape((1, 1)))

gD = {'MLP':f_sgt,'CNN':f_sgt_cnn,'RNN':f_sgt_rnn}

def normlize(xb, dt, eps=1e-5):
    dt.setdefault('mode', 'norm')
    if dt['mode'] == 'norm':
        return (xb - dt['p_mean']) / np.sqrt(dt['p_var'] + eps)
    elif dt['mode'] == 'center':
        return xb - dt['p_mean']


    
def c_label(y,value = 0):
    d = {}
    for i in y:
        if i[0] in d:
            d[i[0]] += 1
        else:
            d[i[0]] = 1
    if value == 0:
        return len(d)
    else:
        return list(d.keys())
	
def cls(n):
	if n == 2:
		return('sigmoid')
	else:
		return('softmax')

def l(n):
	if n == 2:
		return('mse')
	else:
		return('categorical_crossentropy')
		
def label_t(y):
    l = np.shape(y)[0]
    label = c_label(y,value = 1)
    n = len(label)
    y_t = np.empty((l,n),dtype='float')
    mx = np.empty((n,n),dtype='float')
    d = {}
    for i in range(n):
        for j in range(n):
            if j == i:
                mx[i][j] = 1
            else:
                mx[i][j] = 0
    for i in range(n):
        d[label[i]] = mx[i]
    for i in range(l):
        y_t[i] = d[y[i][0]]
    return y_t
    
def gnfile(fname, func=f_sgt):
    f = open(fname,'r')
    x_lst = []
    y_lst = []
    for l in f:
        x, y = func(l)
        x_lst.append(x)
        y_lst.append(y)
    x_arr = np.vstack(x_lst)
    y_arr = np.vstack(y_lst)
    f.close()
    skf = StratifiedShuffleSplit(y_arr,n_iter=1,test_size=0.2,random_state=0)
    idx_tr, idx_te = next((tr, te) for (tr, te) in skf)
    x_tr = x_arr[idx_tr]
    x_te = x_arr[idx_te]
    y_tr = y_arr[idx_tr]
    y_te = y_arr[idx_te]
    y_tr = label_t(y_tr)
    y_te = label_t(y_te)
    return (x_tr,x_te,y_tr,y_te)
	
def wr(n):
	f = open('model_name_save.txt','w')
	f.write(n.rstrip('.h5'))
	f.close()
