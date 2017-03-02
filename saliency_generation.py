from keras import backend as K
import numpy as np

def gninputdata(name):
    name = name.split('_')
    mdl = name[-8]
    length = int(name[-6]) 
    if mdl == 'CNN':
        input_data = np.array([0.25]*length*4).reshape(1,4,1,length)
    elif mdl == 'MLP':
        input_data = np.array([0.25]*length*4).reshape(1,4*length)
    elif mdl == 'RNN':
        input_data = np.array([0.25]*length*4).reshape(1,length,4)
    return input_data

def generate_saliency_map(model, class_num, input_img_data, niter, step):
    first_layer = model.layers[0]
    dense_layer = model.layers[-2]
    out_layer = dense_layer

    input_img = first_layer.input
    output_index = class_num
    loss = out_layer.output[0, output_index]
    grads = K.gradients(loss, input_img)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([input_img, K.learning_phase()], [loss, grads])
    s_map = np.copy(input_img_data)
    for i in range(niter):
        loss_value, grads_value = iterate([s_map, 0])
        # print(loss_value)
        s_map += grads_value * step
    return s_map

def transmtx(mtx):
    sp = np.shape(mtx)
    z = []
    if len(sp) == 2:
        z = mtx.reshape(int(sp[-1]/4),4)
    elif len(sp) == 4:
        for i in range(sp[-1]):
            z.append([])
            for j in range(4):
                z[-1].append(mtx[0][j][0][i])
        z = np.array(z,dtype = 'float32')
    elif len(sp) == 3:
        for i in range(sp[-2]):
            z.append([])
            for j in range(4):
                z[-1].append(mtx[0][i][j])
        z = np.array(z,dtype = 'float32')
    return z