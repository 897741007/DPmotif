# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 19:29:22 2017

@author: xuedy
"""

import subprocess
import numpy as np

from project_re_heatmap import draw_heatmap


def gntf(z,name):
    f = open('project/'+name+'.transfac','w')
    f.write('ID Matrix\nBF\n')
    f.write('{0}\n'.format('\t'.join(['PO', 'A', 'C', 'G', 'T'])))
    for i in range(len(z)):
        f.write(str(i+1)+'\t'+'\t'.join(z[i])+'\n')
    f.close()
    
def gnwl(name):
    cmd = '''weblogo -f {0}.transfac -D transfac -F pdf -o {1}.pdf \
    -c classic \
    --errorbars NO \
    -X NO -Y NO \
    -s large \
    -P '' \
    --logo-font Helvetica \
    --scale-width NO \
    --reverse-stacks YES'''.format('project/'+name,'project/'+name+'_logo')
    status = subprocess.call(cmd, executable='/bin/bash', shell=True)
    return status
    
def model1(mtx):
    z = []
    for i in range(len(mtx)):
        z.append([])
        for j in range(len(mtx[i])):
            z[-1].append(str((mtx[i][j]-mtx.min())/(mtx.max()-mtx.min())*100))
    return z

def model2(mtx):
    z = []
    for i in range(len(mtx)):
        z.append([])
        for j in range(len(mtx[i])):
            z[-1].append(str((mtx[i][j]-mtx[i].min())/(mtx[i].max()-mtx[i].min())*100))
    return z

def model3(name,mtx,p):
    z = []
    length = int(name.split('_')[-6])
    x_l = list(range(length+1))[1:]
    for i in range(length):
        x_l[i] = str(x_l[i])
    mtxs = np.abs(mtx)
    draw_heatmap(p,name,np.transpose(mtx),x_l,['A','C','G','T'],length)
    for i in range(len(mtxs)):
        z.append([])
        for j in range(len(mtxs[i])):
            z[-1].append(str((mtxs[i][j]-mtxs[i].min())/(mtxs[i].max()-mtxs[i].min())*100))
    return z