# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 14:03:28 2017

@author: xuedy
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm 
import numpy as np
import seaborn as sns

def set_font(x):
    if x<10:
        return 18
    else:
        return round(280/(2*x-9))

fontdicts = {'weight':'light','size':'xx-large','family' : 'monospace','color':'dimgrey'}
def draw_heatmap(p,name,data,xlabels,ylabels,s):
    cmap = cm.bwr
    figure=plt.figure(facecolor='w')
    ax=figure.add_subplot(2,1,1,position=[0.1,0.15,0.8,0.8])
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels,fontdicts)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, {'weight':'light','size':set_font(s),'family' : 'monospace','color':'dimgrey'})
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    vmax = data.max()
    vmin = abs(data.min())
    v = max(vmax,vmin)
    map=ax.imshow(data,interpolation='nearest',cmap=cmap,aspect='auto',vmin= -1*v,vmax=v)
    cb=plt.colorbar(mappable=map,cax=None,ax=None,shrink=1)
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                labelbottom="on", left="off", right="off", labelleft="on")
    cb.set_ticks(np.linspace(-v,v,3))
    cb.set_ticklabels((-1.0,0,1.0),{'color':'dimgrey','size':'large','weight':'light'})
    cb.outline.set_visible(False)
    cb.outline.remove()
    sns.set_style('darkgrid')
    #sns.plt.show()
    sns.plt.savefig('project/{0}_heatmap{1}.pdf'.format(name,'@'+p),dpi = 400, bbox_inches='tight',bottom = 'off',top = 'off',left='off',right='off',labelbottom='on',labelleft='on')