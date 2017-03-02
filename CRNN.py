# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 15:12:03 2017

@author: xuedy
"""

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, LSTM
from keras.layers.core import Permute, Reshape
from keras.callbacks import EarlyStopping
from keras.regularizers import WeightRegularizer
from sklearn.metrics import roc_auc_score
from datetime import datetime
from data_preprocessing import gnfile, c_label, f_sgt_cnn, elystp, hps_epo, cls, l, wr
import numpy as np
import os


def CRNN_simple(name,niter,max_epo):
    
	 # define the container for hyper-parameter search
    hyper_parameter = []

    # data preprocessing
    x_tr,x_te,y_tr,y_te = gnfile(name,func = f_sgt_cnn)
    n = c_label(y_tr)
    p,x,y,z = np.shape(x_tr)
    shape_x = int(x)
    shape_y = int(y)
    shape_z = int(z)
    
    for nx in range(niter):
        print('round {0}'.format(nx))
        
        # generate hyper-parameters randomly
        mult = []
        dpw = np.random.uniform(low = 0.3, high = 0.55)
        dpu = np.random.uniform(low = 0.3, high = 0.55)
        ini = np.random.choice([4,5,6])
        lay = np.random.choice([2,3],p=[0.7,0.3])
        nf = 32
        dp1 = np.random.uniform(low = 0.2, high=0.5)
        dp2 = np.random.uniform(low = 0.2, high=0.5)
        dp3 = np.random.uniform(low = 0.2, high=0.5)
        out_put_pre = np.random.choice([1,2])
        
        # train model with current hyper-parameters
        
        epo = hps_epo(niter)
        early_stop = EarlyStopping(monitor='val_loss', patience=elystp(epo), verbose=1)
        
        model = Sequential()
        model.add(ZeroPadding2D((0, 2), dim_ordering='th',input_shape=(shape_x, shape_y, shape_z)))
        model.add(ZeroPadding2D((0, 2), dim_ordering='th'))
        model.add(Convolution2D(nf,1,3, activation='relu',name = 'Conv_1'))
        model.add(ZeroPadding2D((0, 2), dim_ordering='th'))
        mult.append(np.random.choice([1,2],p=[0.3,0.7]))
        model.add(Convolution2D(nf*mult[-1],1,3, activation='relu',name = 'Conv_2'))
        if lay == 3:
            model.add(ZeroPadding2D((0, 2), dim_ordering='th'))
            mult.append(np.random.choice([1,2]))
            model.add(Convolution2D(nf*mult[-1],1,3, activation='relu',name = 'Conv_3'))
        model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
        model.add(Dropout(dp1))
        model.add(Flatten())
        model.add(Permute((0, 3, 2, 1)))
        model.add(Reshape(1,shape_z,shape_x))
        model.add(LSTM(int(shape_x*ini),input_shape=(shape_z,shape_x),consume_less='mem',dropout_W=dpw,dropout_U=dpu,name='Lr_1'))
        model.add(Dense(nf))
        model.add(Dropout(dp2))
        model.add(Dense(shape_x*out_put_pre))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dp3))
        
        # output layer
        model.add(Dense(n,name = 'Output'))
        model.add(Activation(cls(n)))
    
        model.compile(optimizer = 'adadelta' , loss = l(n))
    
        model.fit(x_tr,y_tr,batch_size=32, nb_epoch=epo, verbose=2, validation_data=(x_te, y_te),callbacks = [early_stop])
        auc_tr = roc_auc_score(y_tr, model.predict(x_tr))
        auc_te = roc_auc_score(y_te, model.predict(x_te))
        print('Training AUC: {0}'.format(auc_tr))
        print('Testing AUC: {0}'.format(auc_te))
        
        # store hyper-parameters
        hyper_parameter.append([[ini,dpw,dpu,lay,mult,dp1,dp2,dp3,out_put_pre],[auc_te,auc_tr]])
    
    # get the best result of hyper-parameter search
    early_stop = EarlyStopping(monitor='val_loss', patience=elystp(max_epo), verbose=1)
    hyper_parameter_s = {}
    for i in hyper_parameter:
        hyper_parameter_s[i[1][0]] = i[0]
    hp = hyper_parameter_s[sorted(hyper_parameter_s)[-1]]
    [ini,dpw,dpu,lay,mult,dp1,dp2,dp3,out_put_pre] = hp
    
    # train model again with the outcome of hyper-parameter search
    model = Sequential()
    model.add(ZeroPadding2D((0, 2), dim_ordering='th',input_shape=(shape_x, shape_y, shape_z)))
    model.add(ZeroPadding2D((0, 2), dim_ordering='th'))
    model.add(Convolution2D(nf,1,3, activation='relu',name = 'Conv_1'))
    model.add(ZeroPadding2D((0, 2), dim_ordering='th'))
    mult.append(np.random.choice([1,2],p=[0.3,0.7]))
    model.add(Convolution2D(nf*mult[-1],1,3, activation='relu',name = 'Conv_2'))
    if lay == 3:
        model.add(ZeroPadding2D((0, 2), dim_ordering='th'))
        mult.append(np.random.choice([1,2]))
        model.add(Convolution2D(nf*mult[-1],1,3, activation='relu',name = 'Conv_3'))
    model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
    model.add(Dropout(dp1))
    model.add(Flatten())
    model.add(Permute((0, 3, 2, 1)))
    model.add(Reshape(p,shape_z,shape_x))
    model.add(LSTM(int(shape_x*ini),input_shape=(shape_z,shape_x),consume_less='mem',dropout_W=dpw,dropout_U=dpu,name='Lr_1'))
    model.add(Dense(nf))
    model.add(Dropout(dp2))
    model.add(Dense(shape_x*out_put_pre))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dp3))
    
    # output layer
    model.add(Dense(n,name = 'Output'))
    model.add(Activation(cls(n)))

    model.compile(optimizer = 'adadelta' , loss = l(n))
    
    model.fit(x_tr,y_tr,batch_size=32, nb_epoch=epo, verbose=2, validation_data=(x_te, y_te),callbacks = [early_stop])
    auc_tr = roc_auc_score(y_tr, model.predict(x_tr))
    auc_te = roc_auc_score(y_te, model.predict(x_te))
    print('Training AUC: {0}'.format(auc_tr))
    print('Testing AUC: {0}'.format(auc_te))
    timestamp = datetime.now().strftime('%y_%m_%dT%H_%M_%S')
    s = name.split('.')[0]
    if os.path.exists('project/'+s):
        model.save('project/{0}/{0}_RNN_complex_{2}_{1}.h5'.format(s, timestamp, shape_z))
    else:
        os.makedirs('project/'+s)
        model.save('project/{0}/{0}_RNN_complex_{2}_{1}.h5'.format(s, timestamp, shape_z))
    wr('{0}_RCNN_simple_{2}_{1}.h5'.format(s, timestamp,shape_z))
    
    
       
def CRNN_medium(name,niter,max_epo):
    return name
"""    
	 #定义超参存储
    hyper_parameter = []

    #数据准备
    x_tr,x_te,y_tr,y_te = gnfile(name,func = f_sgt_rnn)
    n = c_label(y_tr)
    x,y,z = np.shape(x_tr)
    x = int(x)
    seq_dim = int(y)
    seq_channel = int(z)
    
    for nx in range(niter):
        layer = np.random.choice([3,4],p=[0.6,0.4])
        model = Sequential()

        model.add(LSTM(int(seq_dim*ini),input_shape=(seq_dim,seq_channel),consume_less='mem',dropout_W=dpw,dropout_U=dpu,name='Lr_1'))
"""        
        
def CRNN_complex(name,niter,max_epo):
    return name
    
    """    
	 #定义超参存储
    hyper_parameter = []

    #数据准备
    x_tr,x_te,y_tr,y_te = gnfile(name,func = f_sgt_rnn)
    n = c_label(y_tr)
    x,y,z = np.shape(x_tr)
    x = int(x)
    seq_dim = int(y)
    seq_channel = int(z)
    
    for nx in range(niter):
        model = Sequential()

        model.add(LSTM(int(seq_dim*ini),input_shape=(seq_dim,seq_channel),consume_less='mem',dropout_W=dpw,dropout_U=dpu,name='Lr_1'))
        
"""        