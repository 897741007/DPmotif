# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 15:06:04 2017

@author: xuedy
"""

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation, LSTM
from keras.callbacks import EarlyStopping
from keras.regularizers import WeightRegularizer
from sklearn.metrics import roc_auc_score
from datetime import datetime
from data_preprocessing import gnfile, c_label, f_sgt_rnn, elystp, hps_epo, cls, l, wr
import numpy as np
import os

def RNN_simple(name,niter,max_epo):
    
    # define the container for hyper-parameter search
    hyper_parameter = []

    # data preprocessing
    x_tr,x_te,y_tr,y_te = gnfile(name,func = f_sgt_rnn)
    n = c_label(y_tr)
    x,y,z = np.shape(x_tr)
    seq_dim = int(y)
    seq_channel = int(z)
    
    for nx in range(niter):
        print('round {0}'.format(nx))
        
        # generate hyper-parameters randomly
        ini = np.random.choice([4,5,6])
        dpw = np.random.uniform(low = 0.3, high = 0.55)
        dpu = np.random.uniform(low = 0.3, high = 0.55)
        wr = np.random.uniform(low = -1, high = 0)
        dn = np.random.choice([3,4])
        dp = np.random.uniform(low = 0.15, high = 0.4)
        
        # train model with current hyper-parameters
        
        epo = hps_epo(niter)
        early_stop = EarlyStopping(monitor='val_loss', patience=elystp(epo), verbose=1)

        model = Sequential()

        model.add(LSTM(int(seq_dim*ini),input_shape=(seq_dim,seq_channel),consume_less='mem',dropout_W=dpw,dropout_U=dpu,name='Lr_1'))
        model.add(Dense(seq_dim*dn,W_regularizer=WeightRegularizer(l2=10 ** wr)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dp))
        
        # output layer
        model.add(Dense(n))
        model.add(Activation(cls(n)))

        model.compile(loss= l(n) , optimizer='adam')
        model.fit(x_tr,y_tr,batch_size=32, nb_epoch=epo, verbose=2, validation_data=(x_te, y_te),callbacks = [early_stop])
        auc_tr = roc_auc_score(y_tr, model.predict(x_tr))
        auc_te = roc_auc_score(y_te, model.predict(x_te))
        print('Training AUC: {0}'.format(auc_tr))
        print('Testing AUC: {0}'.format(auc_te))
        
        # store hyper-parameters
        hyper_parameter.append([[ini,dpw,dpu,wr,dn,dp],[auc_te,auc_tr]])
    
    # get the best result of hyper-parameter search
    early_stop = EarlyStopping(monitor='val_loss', patience=elystp(max_epo), verbose=1)
    hyper_parameter_s = {}
    for i in hyper_parameter:
        hyper_parameter_s[i[1][0]] = i[0]
    hp = hyper_parameter_s[sorted(hyper_parameter_s)[-1]]
    [ini,dpw,dpu,wr,dn,dp] = hp
    
    # train model again with the outcome of hyper-parameter search
    model = Sequential()
    model.add(LSTM(int(seq_dim*ini),input_shape=(seq_dim,seq_channel),consume_less='mem',dropout_W=dpw,dropout_U=dpu,name='Lr_1'))
    model.add(Dense(seq_dim*dn,W_regularizer=WeightRegularizer(l2=10 ** wr)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dp))
    
    # output layer
    model.add(Dense(n))
    model.add(Activation(cls(n)))

    model.compile(loss= l(n) , optimizer='adam')
    model.fit(x_tr,y_tr,batch_size=32, nb_epoch=max_epo, verbose=2, validation_data=(x_te, y_te),callbacks = [early_stop])
    auc_tr = roc_auc_score(y_tr, model.predict(x_tr))
    auc_te = roc_auc_score(y_te, model.predict(x_te))
    print('Training AUC: {0}'.format(auc_tr))
    print('Testing AUC: {0}'.format(auc_te))
    timestamp = datetime.now().strftime('%y_%m_%dT%H_%M_%S')
    s = name.split('.')[0]
    if os.path.exists('project/'+s):
        model.save('project/{0}/{0}_RNN_simple_{2}_{1}.h5'.format(s, timestamp, seq_dim))
    else:
        os.makedirs('project/'+s)
        model.save('project/{0}/{0}_RNN_simple_{2}_{1}.h5'.format(s, timestamp, seq_dim))
    wr('{0}_RNN_simple_{2}_{1}.h5'.format(s, timestamp,seq_dim))

	
def RNN_medium(name,niter,max_epo):

	 # define the container for hyper-parameter search
    hyper_parameter = []

    # data preprocessing
    x_tr,x_te,y_tr,y_te = gnfile(name,func = f_sgt_rnn)
    n = c_label(y_tr)
    x,y,z = np.shape(x_tr)
    x = int(x)
    seq_dim = int(y)
    seq_channel = int(z)
    
    for nx in range(niter):
        print('round {0}'.format(nx))
        
        # generate hyper-parameters randomly
        ini1 = np.random.choice([3,4,5],p = [0.4,0.45,0.15])
        ini2 = np.random.choice([ini1*2-1,ini1*2,ini1*2+1],p = [0.3,0.4,0.3])
        dpw1 = np.random.uniform(low = 0.5, high = 0.7)
        dpw2 = np.random.uniform(low = 0.5, high = 0.7)
        dpw3 = np.random.uniform(low = 0.5, high = 0.7)
        dpu1 = np.random.uniform(low = 0.5, high = 0.7)
        dpu2 = np.random.uniform(low = 0.5, high = 0.7)
        dpu3 = np.random.uniform(low = 0.5, high = 0.7)
        dn = np.random.choice([3,4])
        wr = np.random.uniform(low = -1, high = 0)
        dp = np.random.uniform(low = 0.25, high = 0.55)
        
        # train model with current hyper-parameters
        epo = hps_epo(niter)
        early_stop = EarlyStopping(monitor='val_loss', patience=elystp(epo), verbose=1)
	
        model = Sequential()
        model.add(LSTM(int(seq_channel*ini1),input_shape=(seq_dim, seq_channel),consume_less='mem',dropout_W=dpw1,dropout_U=dpu1,return_sequences=True,name='Lr_1'))
        model.add(LSTM(int(seq_channel*ini2),dropout_W=dpw2,dropout_U=dpu2,return_sequences=True,name='Lr_2'))
        model.add(LSTM(int(seq_channel*ini1),dropout_W=dpw3,dropout_U=dpu3,name='Lr_3'))
        model.add(Dense(seq_channel*dn,W_regularizer=WeightRegularizer(l2=10 ** wr)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dp))
        
        # output layer
        model.add(Dense(n))
        model.add(Activation(cls(n)))
	
        model.compile(loss= l(n) , optimizer='adam')
        model.fit(x_tr,y_tr,batch_size=32, nb_epoch=epo, verbose=2, validation_data=(x_te, y_te),callbacks = [early_stop])
        auc_tr = roc_auc_score(y_tr, model.predict(x_tr))
        auc_te = roc_auc_score(y_te, model.predict(x_te))
        print('Training AUC: {0}'.format(auc_tr))
        print('Testing AUC: {0}'.format(auc_te))

        # store hyper-parameters
        hyper_parameter.append([[ini1,ini2,dpw1,dpw2,dpw3,dpu1,dpu2,dpu3,dn,wr,dp],[auc_te,auc_tr]])
    
    # get the best result of hyper-parameter search
    early_stop = EarlyStopping(monitor='val_loss', patience=elystp(max_epo), verbose=1)
    hyper_parameter_s = {}
    for i in hyper_parameter:
        hyper_parameter_s[i[1][0]] = i[0]
    hp = hyper_parameter_s[sorted(hyper_parameter_s)[-1]]
    [ini1,ini2,dpw1,dpw2,dpw3,dpu1,dpu2,dpu3,dn,wr,dp] = hp
    
    # train model again with the outcome of hyper-parameter search
    model = Sequential()
    model.add(LSTM(int(seq_channel*ini1),input_shape=(seq_dim, seq_channel),consume_less='mem',dropout_W=dpw1,dropout_U=dpu1,return_sequences=True,name='Lr_1'))
    model.add(LSTM(int(seq_channel*ini2),dropout_W=dpw2,dropout_U=dpu2,return_sequences=True,name='Lr_2'))
    model.add(LSTM(int(seq_channel*ini1),dropout_W=dpw3,dropout_U=dpu3,name='Lr_3'))
    model.add(Dense(seq_channel*dn,W_regularizer=WeightRegularizer(l2=10 ** wr)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dp))
    
    # output layer
    model.add(Dense(n))
    model.add(Activation(cls(n)))
    
    model.compile(loss= l(n) , optimizer='adam')
    model.fit(x_tr,y_tr,batch_size=32, nb_epoch=max_epo, verbose=2, validation_data=(x_te, y_te),callbacks = [early_stop])
    auc_tr = roc_auc_score(y_tr, model.predict(x_tr))
    auc_te = roc_auc_score(y_te, model.predict(x_te))
    print('Training AUC: {0}'.format(auc_tr))
    print('Testing AUC: {0}'.format(auc_te))
    timestamp = datetime.now().strftime('%y_%m_%dT%H_%M_%S')
    s = name.split('.')[0]
    if os.path.exists('project/'+s):
        model.save('project/{0}/{0}_RNN_medium_{2}_{1}.h5'.format(s, timestamp, seq_dim))
    else:
        os.makedirs('project/'+s)
        model.save('project/{0}/{0}_RNN_medium_{2}_{1}.h5'.format(s, timestamp, seq_dim))
    wr('{0}_RNN_medium_{2}_{1}.h5'.format(s, timestamp,seq_dim))
	
	
def RNN_complex(name,niter,max_epo):

	 # define the container for hyper-parameter search
    hyper_parameter = []

    # data preprocessing
    x_tr,x_te,y_tr,y_te = gnfile(name,func = f_sgt_rnn)
    n = c_label(y_tr)
    x,y,z = np.shape(x_tr)
    x = int(x)
    seq_dim = int(y)
    seq_channel = int(z)
    
    for nx in range(niter):
        print('round {0}'.format(nx))
        
        # generate hyper-parameters randomly
        ini1 = np.random.choice([2,3,4],p = [0.4,0.45,0.15])
        ini2 = np.random.choice([5,6,7],p = [0.4,0.45,0.15])
        ini3 = np.random.choice([8,9])
        dpn = []
        for i in range(10):
            dpn.append(np.random.uniform(low = 0.5, high = 0.6))
        dp1 = np.random.uniform(low = 0.3, high = 0.6)
        dp2 = np.random.uniform(low = 0.25, high = 0.5)
        dp3 = np.random.uniform(low = 0.2, high = 0.5)
        dn1 = np.random.choice([3,4])
        dn2 = np.random.choice([1,2])
        wr1 = np.random.uniform(low = -1, high = 0)
        wr2 = np.random.uniform(low = -1, high = 0)
        
        # train model with current hyper-parameters
        
        epo = hps_epo(niter)
        early_stop = EarlyStopping(monitor='val_loss', patience=elystp(epo), verbose=1)
        
        model = Sequential()
        model.add(LSTM(int(seq_channel*ini1),input_shape=(seq_dim, seq_channel),consume_less='mem',dropout_W=dpn[0],dropout_U=dpn[1],return_sequences=True,name='Lr_1'))
        model.add(LSTM(int(seq_channel*ini2),dropout_W=dpn[2],dropout_U=dpn[3],return_sequences=True,name='Lr_2'))
        model.add(LSTM(int(seq_channel*ini3),dropout_W=dpn[4],dropout_U=dpn[5],return_sequences=True,name='Lr_3'))
        model.add(LSTM(int(seq_channel*ini2),dropout_W=dpn[6],dropout_U=dpn[7],return_sequences=True,name='Lr_4'))
        model.add(LSTM(int(seq_channel*ini1),dropout_W=dpn[8],dropout_U=dpn[9],name='Lr_5'))
        model.add(Dropout(dp1))
        model.add(Dense(seq_channel*dn1,W_regularizer=WeightRegularizer(l2=10 ** wr1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dp2))
        model.add(Dense(seq_channel*dn2,W_regularizer=WeightRegularizer(l2=10 ** wr2)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dp3))
        
        # output layer
        model.add(Dense(n))
        model.add(Activation(cls(n)))

        model.compile(loss= l(n) , optimizer='adam')
        model.fit(x_tr,y_tr,batch_size=32, nb_epoch=max_epo, verbose=2, validation_data=(x_te, y_te),callbacks = [early_stop])
        auc_tr = roc_auc_score(y_tr, model.predict(x_tr))
        auc_te = roc_auc_score(y_te, model.predict(x_te))
        print('Training AUC: {0}'.format(auc_tr))
        print('Testing AUC: {0}'.format(auc_te))

        # store hyper-parameters
        hyper_parameter.append([[ini1,ini2,ini3,dpn,dp1,dp2,dp3,dn1,dn2,wr1,wr2],[auc_te,auc_tr]])
        
    ## get the best result of hyper-parameter search
    early_stop = EarlyStopping(monitor='val_loss', patience=elystp(max_epo), verbose=1)
    hyper_parameter_s = {}
    for i in hyper_parameter:
        hyper_parameter_s[i[1][0]] = i[0]
    hp = hyper_parameter_s[sorted(hyper_parameter_s)[-1]]
    [ini1,ini2,ini3,dpn,dp1,dp2,dp3,dn1,dn2,wr1,wr2] = hp
    
    # train model again with the outcome of hyper-parameter search
    model = Sequential()
    model.add(LSTM(int(seq_channel*ini1),input_shape=(seq_dim,seq_channel),consume_less='mem',dropout_W=dpn[0],dropout_U=dpn[1],return_sequences=True,name='Lr_1'))
    model.add(LSTM(int(seq_channel*ini2),dropout_W=dpn[2],dropout_U=dpn[3],return_sequences=True,name='Lr_2'))
    model.add(LSTM(int(seq_channel*ini3),dropout_W=dpn[4],dropout_U=dpn[5],return_sequences=True,name='Lr_3'))
    model.add(LSTM(int(seq_channel*ini2),dropout_W=dpn[6],dropout_U=dpn[7],return_sequences=True,name='Lr_4'))
    model.add(LSTM(int(seq_channel*ini1),dropout_W=dpn[8],dropout_U=dpn[9],name='Lr_5'))
    model.add(Dropout(dp1))
    model.add(Dense(seq_channel*dn1,W_regularizer=WeightRegularizer(l2=10 ** wr1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dp2))
    model.add(Dense(seq_channel*dn2,W_regularizer=WeightRegularizer(l2=10 ** wr2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dp3))
    
    # output layer
    model.add(Dense(n))
    model.add(Activation(cls(n)))

    model.compile(loss= l(n) , optimizer='adam')
    model.fit(x_tr,y_tr,batch_size=32, nb_epoch=max_epo, verbose=2, validation_data=(x_te, y_te),callbacks = [early_stop])
    auc_tr = roc_auc_score(y_tr, model.predict(x_tr))
    auc_te = roc_auc_score(y_te, model.predict(x_te))
    print('Training AUC: {0}'.format(auc_tr))
    print('Testing AUC: {0}'.format(auc_te))
    timestamp = datetime.now().strftime('%y_%m_%dT%H_%M_%S')
    s = name.split('.')[0]
    if os.path.exists('project/'+s):
        model.save('project/{0}/{0}_RNN_complex_{2}_{1}.h5'.format(s, timestamp, seq_dim))
    else:
        os.makedirs('project/'+s)
        model.save('project/{0}/{0}_RNN_complex_{2}_{1}.h5'.format(s, timestamp, seq_dim))
    wr('{0}_RNN_complex_{2}_{1}.h5'.format(s, timestamp,seq_dim))

