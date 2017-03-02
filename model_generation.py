# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 15:20:54 2017

@author: xuedy
"""
import sys
from MLP import *
from CNN import *
from RNN import *
from CRNN import *

def DM(mdl,cplx,name,niter,max_epo):
    models = {'CNN':{'simple':CNN_simple,'medium':CNN_medium,'complex':CNN_complex},
              'MLP':{'simple':MLP_simple,'medium':MLP_medium,'complex':MLP_complex},
              'RNN':{'simple':RNN_simple,'medium':RNN_medium,'complex':RNN_complex},
              'CRNN':{'simple':CRNN_simple,'medium':CRNN_medium,'complex':CRNN_complex}}
    models[mdl][cplx](name,niter,max_epo)
    
import sys
parameter = sys.argv[1:]
DM(parameter[3],parameter[0],parameter[1],int(parameter[4]),int(parameter[2]))