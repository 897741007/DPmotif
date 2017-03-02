# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 13:58:09 2017

@author: xuedy
"""

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping
from keras.regularizers import WeightRegularizer
from sklearn.metrics import roc_auc_score
from datetime import datetime
from data_preprocessing import gnfile, c_label, f_sgt_cnn, elystp, hps_epo, cls, l, wr
import numpy as np
import os

def CNN_simple(name, niter, max_epo):

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
        lay = np.random.choice([2,3,4])
        mult = np.random.choice([1,2],p=[0.3,0.7])
        dp1 = np.random.uniform(low = 0.2, high=0.5)
        dp2 = np.random.uniform(low = 0.2, high=0.5)
        dp3 = np.random.uniform(low = 0.2, high=0.5)
        out_put_pre = np.random.choice([1,2])

        # train model with current hyper-parameters
        epo = hps_epo(niter)
        early_stop = EarlyStopping(monitor='val_loss', patience=elystp(epo), verbose=1)

        model = Sequential()
        model.add(ZeroPadding2D((0, 2), dim_ordering='th',input_shape=(shape_x, shape_y, shape_z)))
        nf = 32
        model.add(Convolution2D(nf,1,3, activation='relu',name = 'Conv_1'))
        for i in range(lay-1):
            model.add(ZeroPadding2D((0, 2 ),dim_ordering='th'))
            nf = nf*mult
            model.add(Convolution2D(nf,1,3, activation='relu',name = 'Conv_'+str(i+2)))
        model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
        model.add(Dropout(dp1))
        model.add(Flatten())
        model.add(Dense(nf))
        model.add(Dropout(dp2))
        model.add(Dense(shape_z*out_put_pre))
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
        hyper_parameter.append([[lay,mult,dp1,dp2,dp3,out_put_pre],[auc_te,auc_tr]])

    # get the best result of hyper-parameter search
    early_stop = EarlyStopping(monitor='val_loss', patience=elystp(max_epo), verbose=1)
    hyper_parameter_s = {}
    for i in hyper_parameter:
        hyper_parameter_s[i[1][0]] = i[0]
    hp = hyper_parameter_s[sorted(hyper_parameter_s)[-1]]
    [lay,mult,dp1,dp2,dp3,out_put_pre] = hp
    
    # train model again with the outcome of hyper-parameter search
    model = Sequential()
    model.add(ZeroPadding2D((0, 2), dim_ordering='th',input_shape=(shape_x, shape_y, shape_z)))
    nf = 32
    model.add(Convolution2D(nf,1,3, activation='relu',name = 'Conv_1'))
    for i in range(lay-1):
        model.add(ZeroPadding2D((0, 2 ),dim_ordering='th'))
        nf = nf*(mult)
        model.add(Convolution2D(nf,1,3, activation='relu',name = 'Conv_'+str(i+2)))
    model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
    model.add(Dropout(dp1))
    model.add(Flatten())
    model.add(Dense(nf))
    model.add(Dropout(dp2))
    model.add(Dense(shape_z*out_put_pre,))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dp3))
    
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
        model.save('project/{0}/{0}_CNN_simple_{2}_{1}.h5'.format(s, timestamp,shape_z))
    else:
        os.makedirs('project/'+s)
        model.save('project/{0}/{0}_CNN_simple_{2}_{1}.h5'.format(s, timestamp,shape_z))
    wr('{0}_CNN_simple_{2}_{1}.h5'.format(s, timestamp,shape_z))
    
def CNN_medium(name, niter, max_epo):
	
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
        lay = np.random.choice([5,6,7,8])
        itz = np.random.choice([32,64])
        mult = np.random.choice([1,2],p=[0.3,0.7])
        dp1 = np.random.uniform(low = 0.2, high=0.5)
        dp2 = np.random.uniform(low = 0.2, high=0.5)
        dp3 = np.random.uniform(low = 0.2, high=0.5)
        out_put_pre1 = np.random.choice([2,3])
        out_put_pre2 = np.random.choice([1,2])
        wr = np.random.uniform(low = -1, high = 0)
		
        # train model with current hyper-parameters
        epo = hps_epo(niter)
        early_stop = EarlyStopping(monitor='val_loss', patience=elystp(epo), verbose=1)
	
        model = Sequential()
        model.add(ZeroPadding2D((0, 2), dim_ordering='th', batch_input_shape=(None, shape_x, shape_y, shape_z)))
        nf = itz
        model.add(Convolution2D(nf,1,3, activation='relu',name = 'Conv_1'))
        z = 0
        for i in range(lay-1):
            model.add(ZeroPadding2D((0, 2), dim_ordering='th'))
            nf = nf*mult
            model.add(Convolution2D(nf,1,3, activation='relu', name = 'Conv_'+str(i+2)))
            if (i+2)%3 == 0:
                model.add(MaxPooling2D((1, 2), strides=(1, 2)))
                z += 1
        if z < 2:
            model.add(MaxPooling2D((1, 2), strides=(1, 2)))
        model.add(Dropout(dp1))
        model.add(Flatten())
        model.add(Dense(shape_z*out_put_pre1, W_regularizer = WeightRegularizer(l2=10 ** wr)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dp2))
        model.add(Dense(shape_z*out_put_pre2))
        model.add(Activation('relu'))
        model.add(Dropout(dp3))
        
        # output layer
        model.add(Dense(n,name = 'Output'))
        model.add(Activation(cls(n)))
	
        model.compile(optimizer = 'adadelta' , loss = l(n)  )
        model.fit(x_tr,y_tr,batch_size=32, nb_epoch=epo, verbose=2, validation_data=(x_te, y_te),callbacks = [early_stop])
        auc_tr = roc_auc_score(y_tr, model.predict(x_tr))
        auc_te = roc_auc_score(y_te, model.predict(x_te))
        print('Training AUC: {0}'.format(auc_tr))
        print('Testing AUC: {0}'.format(auc_te))
		
		  # store hyper-parameters
        hyper_parameter.append([[lay,mult,dp1,dp2,dp3,out_put_pre1,out_put_pre2],[auc_te,auc_tr]])
	
	 # get the best result of hyper-parameter search
    early_stop = EarlyStopping(monitor='val_loss', patience=elystp(max_epo), verbose=1)
    hyper_parameter_s = {}
    for i in hyper_parameter:
        hyper_parameter_s[i[1][0]] = i[0]
    hp = hyper_parameter_s[sorted(hyper_parameter_s)[-1]]
    [lay,mult,dp1,dp2,dp3,out_put_pre1,out_put_pre2] = hp
	 
    # train model again with the outcome of hyper-parameter search
    model = Sequential()
    model.add(ZeroPadding2D((0, 2), dim_ordering='th', batch_input_shape=(None, shape_x, shape_y, shape_z)))
    nf = itz
    model.add(Convolution2D(nf,1,3, activation='relu',name = 'Conv_1'))
    z = 0
    for i in range(lay-1):
        model.add(ZeroPadding2D((0, 2),dim_ordering='th'))
        nf = nf*mult
        model.add(Convolution2D(nf,1,3, activation='relu', name = 'Conv_'+str(i+2)))
        if (i+2)%3 == 0:
            model.add(MaxPooling2D((1, 2), strides=(1, 2)))
            z += 1
    if z < 2:
        model.add(MaxPooling2D((1, 2), strides=(1, 2)))
    model.add(Dropout(dp1))
    model.add(Flatten())
    model.add(Dense(nf*out_put_pre1,W_regularizer=WeightRegularizer(l2=10 ** wr)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dp2))
    model.add(Dense(shape_z*out_put_pre2))
    model.add(Activation('relu'))
    model.add(Dropout(dp3))
    
    # output layer
    model.add(Dense(n,name = 'Output'))
    model.add(Activation(cls(n)))

    model.compile(optimizer = 'adadelta' , loss = l(n)  )
    model.fit(x_tr,y_tr,batch_size=32, nb_epoch=max_epo, verbose=2, validation_data=(x_te, y_te),callbacks = [early_stop])
    auc_tr = roc_auc_score(y_tr, model.predict(x_tr))
    auc_te = roc_auc_score(y_te, model.predict(x_te))
    print('Training AUC: {0}'.format(auc_tr))
    print('Testing AUC: {0}'.format(auc_te))
    timestamp = datetime.now().strftime('%y_%m_%dT%H_%M_%S')
    s = name.split('.')[0]
    if os.path.exists('project/'+s):
        model.save('project/{0}/{0}_CNN_medium_{2}_{1}.h5'.format(s, timestamp,shape_z))
    else:
        os.makedirs('project/'+s)
        model.save('project/{0}/{0}_CNN_medium_{2}_{1}.h5'.format(s, timestamp,shape_z))
    wr('{0}_CNN_medium_{2}_{1}.h5'.format(s, timestamp,shape_z))

def CNN_complex(name, niter, max_epo):
    
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
        lay = np.random.choice([9,10,11,12])
        ini = np.random.choice([32,64],p = [0.3,0.7])
        dp = []
        for dpx in range(lay):
            dp.append(np.random.uniform(low = 0.2, high=0.5))
        dps = []
        dpn = 0
        den_1 = np.random.choice([shape_z*5,shape_z*6,shape_z*7])
        wr = np.random.uniform(low = -1, high = 0)
        den_2 = np.random.choice([shape_z*4,shape_z*3])
        den_3 = np.random.choice([shape_z,shape_z*2])

        # train model with current hyper-parameters
        epo = hps_epo(niter)
        early_stop = EarlyStopping(monitor='val_loss', patience=elystp(epo), verbose=1)

        model = Sequential()
        model.add(ZeroPadding2D((0, 2), dim_ordering='th', batch_input_shape=(None, shape_x, shape_y, shape_z)))
        nf = ini
        model.add(Convolution2D(nf,1,3, activation='relu',name = 'Conv_1'))
        zs = 0
        for i in range(lay-1):
            model.add(ZeroPadding2D((0, 2)))
            model.add(Convolution2D(nf,1,3, activation='relu', name = 'Conv_'+str(i+2)))
            if (i+2)%3 == 0:
                model.add(MaxPooling2D((1, 2), strides=(1, 2)))
                zs += 1
                if zs>1:
                    model.add(Dropout(dp[dpn]))
                    dpn = dpn + 1
        if zs < 3:
            model.add(MaxPooling2D((1, 2), strides=(1, 2)))
        model.add(Dropout(dp[dpn]))
        dpn += 1
        model.add(Flatten())
        model.add(Dense(den_1,W_regularizer=WeightRegularizer(l2=10 ** wr)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dp[dpn]))
        dpn += 1
        model.add(Dense(den_2))
        model.add(Activation('relu'))
        model.add(Dropout(dp[dpn]))
        dpn += 1
        model.add(Dense(den_3))
        model.add(Activation('relu'))
        model.add(Dropout(dp[dpn]))
        dpn += 1
        
        # output layer
        model.add(Dense(n,name = 'Output'))
        model.add(Activation(cls(n)))
	
        model.compile(optimizer = 'adadelta' , loss = l(n))
    
        model.fit(x_tr,y_tr,batch_size=32, nb_epoch=epo, verbose=2, validation_data=(x_te, y_te),callbacks = [early_stop])
        auc_tr = roc_auc_score(y_tr, model.predict(x_tr))
        auc_te = roc_auc_score(y_te, model.predict(x_te))
        print('Training AUC: {0}'.format(auc_tr))
        print('Testing AUC: {0}'.format(auc_te))

        dps = dp[:dpn]

        # store hyper-parameters
        hyper_parameter.append([[lay,ini,dps,den_1,den_2,den_3,wr],[auc_te,auc_tr]])

    # get the best result of hyper-parameter search
    early_stop = EarlyStopping(monitor='val_loss', patience=elystp(max_epo), verbose=1)
    hyper_parameter_s = {}
    for i in hyper_parameter:
        hyper_parameter_s[i[1][0]] = i[0]
    hp = hyper_parameter_s[sorted(hyper_parameter_s)[-1]]
    [lay,ini,dps,den_1,den_2,den_3,wr] = hp
    dpn = 0
    
    # train model again with the outcome of hyper-parameter search
    model = Sequential()
    model.add(ZeroPadding2D((0, 2), dim_ordering='th', batch_input_shape=(None, shape_x, shape_y, shape_z)))
    nf = ini
    model.add(Convolution2D(nf,1,3, activation='relu',name = 'Conv_1'))
    zs = 0
    for i in range(lay-1):
        model.add(ZeroPadding2D((0, 2)))
        model.add(Convolution2D(nf,1,3, activation='relu', name = 'Conv_'+str(i+2)))
        if (i+2)%3 == 0:
            model.add(MaxPooling2D((1, 2), strides=(1, 2)))
            zs += 1
            if zs>1:
                model.add(Dropout(dps[dpn]))
                dpn = dpn + 1
    if zs < 3:
        model.add(MaxPooling2D((1, 2), strides=(1, 2)))
    model.add(Dropout(dps[dpn]))
    dpn += 1
    model.add(Flatten())
    model.add(Dense(den_1,W_regularizer=WeightRegularizer(l2=10 ** wr)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dps[dpn]))
    dpn += 1
    model.add(Dense(den_2))
    model.add(Activation('relu'))
    model.add(Dropout(dps[dpn]))
    dpn += 1
    model.add(Dense(den_3))
    model.add(Activation('relu'))
    model.add(Dropout(dps[dpn]))
    dpn += 1
    
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
        model.save('project/{0}/{0}_CNN_complex_{2}_{1}.h5'.format(s, timestamp, shape_z))
    else:
        os.makedirs('project/'+s)
        model.save('project/{0}/{0}_CNN_complex_{2}_{1}.h5'.format(s, timestamp, shape_z))
    wr('{0}_CNN_complex_{2}_{1}.h5'.format(s, timestamp,shape_z))