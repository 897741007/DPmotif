# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:45:42 2017

@author: xuedy
"""

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score
from datetime import datetime
from data_preprocessing import gnfile, c_label, f_sgt, elystp, hps_epo, cls, l, wr
import numpy as np
import os


def MLP_simple(name,niter,max_epo):

	 # define the container for hyper-parameter search
    hyper_parameter = []

    # data preprocessing
    x_tr,x_te,y_tr,y_te = gnfile(name,func = f_sgt)
    n = c_label(y_tr)
    y,z = np.shape(x_tr)
    shape_z = int(z)
    for nx in range(niter):
        print('round {0}'.format(nx))
        
        # generate hyper-parameters randomly
        lay = np.random.choice([2,3,4])
        ini = np.random.choice([shape_z*1,shape_z*2,shape_z*3])
        dp = []
        ndss = []
        nds = []
        dpf = np.random.uniform(low = 0.15, high = 0.45)
                
	     # train model with current hyper-parameters
        epo = hps_epo(niter)
        early_stop = EarlyStopping(monitor='val_loss', patience=elystp(epo), verbose=1)
        nd = ini
        model = Sequential()
        nds.append(nd)
        model.add(Dense(nd, input_dim=shape_z, name = 'Den_1'))
        model.add(Activation('relu'))
        for i in range(lay-1):
            if i < lay/2+1:
                nd = nd+shape_z*np.random.choice([1,2])
                nds.append(nd)
            else:
                nd = nds[-1]
                nds = nds[:-1]
                ndss.append(nd)
            model.add(Dense(nd, name = 'Den_'+str(i+2)))
            ndss.append(nd)
            model.add(Activation('relu'))
            dps = np.random.uniform(low = 0.15, high = 0.45)
            model.add(Dropout(dps))
            dp.append(dps)
        model.add(Dense(ini,name = 'out_put_pre'))
        model.add(Activation('relu'))
        model.add(Dropout(dpf))

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
        hyper_parameter.append([[lay,ini,dp,ndss,dpf],[auc_te,auc_tr]])

    # get the best result of hyper-parameter search
    early_stop = EarlyStopping(monitor='val_loss', patience=elystp(max_epo), verbose=1)
    hyper_parameter_s = {}
    for i in hyper_parameter:
        hyper_parameter_s[i[1][0]] = i[0]
    hp = hyper_parameter_s[sorted(hyper_parameter_s)[-1]]
    [lay,ini,dp,ndss,dpf] = hp
    
    # train model again with the outcome of hyper-parameter search
    model = Sequential()
    nds = []
    nds.append(nd)
    nd = ini
    model.add(Dense(nd, input_dim=shape_z, name = 'Den_1'))
    model.add(Activation('relu'))
    for i in range(lay-1):
        model.add(Dense(ndss[0], name = 'Den_'+str(i+2)))
        ndss = ndss[1:]
        model.add(Activation('relu'))
        model.add(Dropout(dp[0]))
        dp = dp[1:]
    model.add(Dense(ini,name = 'out_put_pre'))
    model.add(Activation('relu'))
    model.add(Dropout(dpf))

    # output layer
    model.add(Dense(n,name = 'Output'))
    model.add(Activation(cls(n)))
    model.compile(optimizer = 'adadelta' , loss = l(n))

    model.fit(x_tr,y_tr,batch_size=32, nb_epoch=max_epo, verbose=2, validation_data=(x_te, y_te),callbacks = [early_stop])
    auc_tr = roc_auc_score(y_tr, model.predict(x_tr))
    auc_te = roc_auc_score(y_te, model.predict(x_te))
    print('Training AUC: {0}'.format(auc_tr))
    print('Testing AUC: {0}'.format(auc_te))
    timestamp = datetime.now().strftime('%y_%m_%dT%H_%M_%S')
    s = name.split('.')[0]
    if os.path.exists('project/'+s):
        model.save('project/{0}/{0}_MLP_simple_{2}_{1}.h5'.format(s, timestamp, int(shape_z/4)))
    else:
        os.makedirs('project/'+s)
        model.save('project/{0}/{0}_MLP_simple_{2}_{1}.h5'.format(s, timestamp, int(shape_z/4)))
    wr('{0}_MLP_simple_{2}_{1}.h5'.format(s, timestamp,int(shape_z/4)))
	

def MLP_medium(name,niter,max_epo):

    # define the container for hyper-parameter search
    hyper_parameter = []

    # data preprocessing
    x_tr,x_te,y_tr,y_te = gnfile(name,func = f_sgt)
    n = c_label(y_tr)
    y,z = np.shape(x_tr)
    shape_z = int(z)
    for nx in range(niter):
        print('round {0}'.format(nx))
        
        # generate hyper-parameters randomly
        lay = np.random.choice([5,6,7,8])
        ini = np.random.choice([shape_z*1,shape_z*2,shape_z*3])
        dpf = np.random.uniform(low = 0.15, high = 0.45)
        dp = []
        ndss = []
        nds = []

        # train model with current hyper-parameters
        epo = hps_epo(niter)
        early_stop = EarlyStopping(monitor='val_loss', patience=elystp(epo), verbose=1)
        nd = ini
        model = Sequential()
        model.add(Dense(nd, input_dim=shape_z, name = 'Den_1'))
        model.add(Activation('relu'))
        for i in range(lay-1):
            if i < lay/2+1:
                nd = nd+np.random.choice([shape_z*1,shape_z*2])
                nds.append(nd)
            else:
                nd = nds[-1]
                nds = nds[:-1]
            model.add(Dense(nd,name = 'Den_'+str(i+2)))
            ndss.append(nd)
            if i>3:
                model.add(BatchNormalization())
            model.add(Activation('relu'))
            if i>4:
                dps = np.random.uniform(low = 0.15, high = 0.45)
                model.add(Dropout(dps))
                dp.append(dps)
        model.add(Dense(ini,name = 'out_put_pre'))
        model.add(Activation('relu'))
        model.add(Dropout(dpf))
        
        # output layer
        model.add(Dense(n,name = 'Output'))
        model.add(Activation(cls(n)))

        model.compile(optimizer = 'adadelta' , loss = l(n) )
        model.fit(x_tr,y_tr,batch_size=32, nb_epoch=epo, verbose=2, validation_data=(x_te, y_te),callbacks = [early_stop])
        auc_tr = roc_auc_score(y_tr, model.predict(x_tr))
        auc_te = roc_auc_score(y_te, model.predict(x_te))
        print('Training AUC: {0}'.format(auc_tr))
        print('Testing AUC: {0}'.format(auc_te))

        # store hyper-parameters
        hyper_parameter.append([[lay,ini,dpf,dp,ndss],[auc_te,auc_tr]])
    
    # get the best result of hyper-parameter search
    early_stop = EarlyStopping(monitor='val_loss', patience=elystp(max_epo), verbose=1)
    hyper_parameter_s = {}
    for i in hyper_parameter:
        hyper_parameter_s[i[1][0]] = i[0]
    hp = hyper_parameter_s[sorted(hyper_parameter_s)[-1]]
    [lay,ini,dpf,dp,ndss] = hp
    
    # train model again with the outcome of hyper-parameter search
    model = Sequential()
    model.add(Dense(nd, input_dim=shape_z, name = 'Den_1'))
    model.add(Activation('relu'))
    for i in range(lay-1):
        model.add(Dense(ndss[0],name = 'Den_'+str(i+2)))
        ndss = ndss[1:]
        if i>3:
            model.add(BatchNormalization())
        model.add(Activation('relu'))
        if i>4:
            dps = np.random.uniform(low = 0.15, high = 0.45)
            model.add(Dropout(dp[0]))
            dp = dp[1:]
    model.add(Dense(ini,name = 'out_put_pre'))
    model.add(Activation('relu'))
    model.add(Dropout(dpf))
    
    # output layer
    model.add(Dense(n,name = 'Output'))
    model.add(Activation(cls(n)))
    model.compile(optimizer = 'adadelta' , loss = l(n))
    
    model.fit(x_tr,y_tr,batch_size=32, nb_epoch=max_epo, verbose=2, validation_data=(x_te, y_te),callbacks = [early_stop])
    auc_tr = roc_auc_score(y_tr, model.predict(x_tr))
    auc_te = roc_auc_score(y_te, model.predict(x_te))
    print('Training AUC: {0}'.format(auc_tr))
    print('Testing AUC: {0}'.format(auc_te))
    timestamp = datetime.now().strftime('%y_%m_%dT%H_%M_%S')
    s = name.split('.')[0]
    if os.path.exists('project/'+s):
        model.save('project/{0}/{0}_MLP_medium_{2}_{1}.h5'.format(s, timestamp, int(shape_z/4)))
    else:
        os.makedirs('project/'+s)
        model.save('project/{0}/{0}_MLP_medium_{2}_{1}.h5'.format(s, timestamp, int(shape_z/4)))
    wr('{0}_MLP_medium_{2}_{1}.h5'.format(s, timestamp,int(shape_z/4)))


def MLP_complex(name,niter,max_epo):
	

    # define the container for hyper-parameter search
    hyper_parameter = []

    # data preprocessing
    x_tr,x_te,y_tr,y_te = gnfile(name,func = f_sgt)
    n = c_label(y_tr)
    y,z = np.shape(x_tr)
    shape_z = int(z)
    for nx in range(niter):
        print('round {0}'.format(nx))
        
        # generate hyper-parameters randomly
        lay = np.random.choice([9,10,11,12])
        ini = np.random.choice([shape_z*1,shape_z*2,shape_z*3])
        dpf = np.random.uniform(low = 0.15, high = 0.45)
        dp = []
        ndss = []
        nds = []


        # train model with current hyper-parameters
        epo = hps_epo(niter)
        early_stop = EarlyStopping(monitor='val_loss', patience=elystp(epo), verbose=1)
        nd = ini
        model = Sequential()
        model.add(Dense(nd, input_dim=shape_z,name = 'Den_1'))
        model.add(Activation('relu'))
        lw = 0.15
        hg = 0.4
        for i in range(lay-1):
            if i < lay/2+1:
                pd = 0.05*i+0.1
                pds = (1-pd)/2
                nd = nd+np.random.choice([shape_z*1,shape_z*2,shape_z*3],p = [pds,pds,pd])
                nds.append(nd)
            else:
                nd = nds[-1]
                nds = nds[:-1]
            model.add(Dense(nd,name = 'Den_'+str(i+2)))
            ndss.append(nd)
            if i>lay-4:
                model.add(BatchNormalization())
            model.add(Activation('relu'))
            if i>lay-3:
                dps = np.random.uniform(low = lw, high = hg)
                model.add(Dropout(dps))
                dp.append(dps)
                lw += 0.03
                hg += 0.025
        model.add(Dense(ini,name = 'out_put_pre'))
        model.add(Activation('relu'))
        model.add(Dropout(dpf))
        
        # output layer
        model.add(Dense(n,name = 'Output'))
        model.add(Activation(cls(n)))

        model.compile(optimizer = 'adadelta' , loss = l(n) )
        model.fit(x_tr,y_tr,batch_size=32, nb_epoch=epo, verbose=2, validation_data=(x_te, y_te),callbacks = [early_stop])
        auc_tr = roc_auc_score(y_tr, model.predict(x_tr))
        auc_te = roc_auc_score(y_te, model.predict(x_te))
        print('Training AUC: {0}'.format(auc_tr))
        print('Testing AUC: {0}'.format(auc_te))

        # store hyper-parameters
        hyper_parameter.append([[lay,ini,dpf,dp,ndss],[auc_te,auc_tr]])
    
    # get the best result of hyper-parameter search
    early_stop = EarlyStopping(monitor='val_loss', patience=elystp(max_epo), verbose=1)
    hyper_parameter_s = {}
    for i in hyper_parameter:
        hyper_parameter_s[i[1][0]] = i[0]
    hp = hyper_parameter_s[sorted(hyper_parameter_s)[-1]]
    [lay,ini,dpf,dp,ndss] = hp
    
    # train model again with the outcome of hyper-parameter search
    model = Sequential()
    model.add(Dense(nd, input_dim=shape_z,name = 'Den_1'))
    model.add(Activation('relu'))
    for i in range(lay-1):
        model.add(Dense(ndss[0],name = 'Den_'+str(i+2)))
        ndss = ndss[1:]
        if i>lay-4:
            model.add(BatchNormalization())
        model.add(Activation('relu'))
        if i>lay-3:
            dps = np.random.uniform(low = 0.15, high = 0.45)
            model.add(Dropout(dp[0]))
            dp = dp[1:]
    model.add(Dense(ini,name = 'out_put_pre'))
    model.add(Activation('relu'))
    model.add(Dropout(dpf))
    
    # output layer
    model.add(Dense(n,name = 'Output'))
    model.add(Activation(cls(n)))
    model.compile(optimizer = 'adadelta' , loss = l(n))
    
    model.fit(x_tr,y_tr,batch_size=32, nb_epoch=max_epo, verbose=2, validation_data=(x_te, y_te),callbacks = [early_stop])
    auc_tr = roc_auc_score(y_tr, model.predict(x_tr))
    auc_te = roc_auc_score(y_te, model.predict(x_te))
    print('Training AUC: {0}'.format(auc_tr))
    print('Testing AUC: {0}'.format(auc_te))
    timestamp = datetime.now().strftime('%y_%m_%dT%H_%M_%S')
    s = name.split('.')[0]
    if os.path.exists('project/'+s):
        model.save('project/{0}/{0}_MLP_complex_{2}_{1}.h5'.format(s, timestamp, int(shape_z/4)))
    else:
        os.makedirs('project/'+s)
        model.save('project/{0}/{0}_MLP_complex_{2}_{1}.h5'.format(s, timestamp, int(shape_z/4)))
    
    wr('{0}_MLP_complex_{2}_{1}.h5'.format(s, timestamp,int(shape_z/4)))
	