# -*- coding: utf-8 -*-
#! /usr/bin/env python
"""
Created on Thu Mar  2 15:24:14 2017

@author: xuedy
"""

import subprocess
import argparse

parser = argparse.ArgumentParser(description='train deep learning model to see and understand motif')
parser.add_argument('-d', metavar='data', type=str, nargs='?',
                    dest='data',
                    help='training data')
parser.add_argument('-c', metavar='complexity', type=str, nargs='?',choices=['simple','medium','complex'],
                    dest='complexity',
                    help='the complexity of model:simple\tmedium\tcomplex')
parser.add_argument('-e', metavar='max_epoch', type=int, nargs='?',
                    dest='max_epoch',
                    help='the number of times for model training')
parser.add_argument('-m', metavar='model type', type=str, nargs='?',choices=['CNN','MLP','RNN','CRNN'],
                    dest='model_type',
                    help='which type of deeplearning model will be selected')
parser.add_argument('-n', metavar='niter', type=int, nargs='?',
                    dest='niter',
                    help='the number of the times for hyperparameter searching')
parser.add_argument('-l', metavar='log', type=bool, nargs='?',choices=[True,False],
                    dest='log',
                    help='if log is needed')
parser.add_argument('-f', metavar='frame', type=int, nargs='?',choices=[1,2,3],
                    dest='frame',
                    help='''the model of picture print:
                        1
                        2
                        3''')

args = parser.parse_args()
print('Initializing...')
print('Entering Parameter...')
print('Generating Optimized Model...') 

if args.log:
    cmd = 'python model_generation.py {0} {1} {2} {3} {4} | tee mnist_hs.log'.format(args.complexity,args.data,args.max_epoch,args.model_type,args.niter)
else:
    cmd = 'python model_generation.py {0} {1} {2} {3} {4}'.format(args.complexity,args.data,args.max_epoch,args.model_type,args.niter)
subprocess.call(cmd, executable='/bin/bash', shell=True)

from keras.models import load_model
from saliency_generation import generate_saliency_map,transmtx,gninputdata
from weblogo_generation import gntf,gnwl

f = open('model_name_save.txt','r')
model_name = f.readline()
f.close()
model = load_model('project/{0}/'.format(args.data.rstrip('.txt'))+model_name+'.h5')

print('Generating Saliency Map...')

f = open(args.data,'r')
c_n = []
for l in f:
    label = l.rstrip('\n').split('\t')[-1]
    if (label in c_n) == False:
        c_n.append(label)
f.close()
niter = {'simple':100,'medium':200,'complex':400}
for i in range(len(c_n)):
    print('class_{0}'.format(c_n[i]))
    
    mtx = generate_saliency_map(model,i,gninputdata(model_name),niter[args.complexity],1)
    mtxs = transmtx(mtx)

    if args.frame == 1:
        from project_re_weblogo import model1
        mtxs = model1(mtxs)
    elif args.frame == 2:
        from project_re_weblogo import model2
        mtxs = model2(mtxs)
    elif args.frame == 3:
        from project_re_weblogo import model3
        mtxs = model3(model_name,mtxs,c_n[i])

    gntf(mtxs,model_name+'@'+c_n[i])
    gnwl(model_name+'@'+c_n[i])

print('Done')
