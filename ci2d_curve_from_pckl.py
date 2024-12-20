# importing modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lmfit as lm
import mathematical_functions as fn
import other_functions as oth
import os
import shutil
from datetime import datetime
from multiprocessing import Pool
##############################################
#Calculate confidence interval
##############################################

file= os.getcwd()
file = file.replace('\\','/')
file= file.split('/')[-1]

ci2d_result=oth.load_object(f'{file}_ci2d.pckl')

minimizer_result=oth.load_object(f'{file}_minimized.pckl')

chisqr=minimizer_result.chisqr

num=len(ci2d_result)
y=int(num**0.5)
x=int(num/y)
while num % x !=0:
    y=y-1
    x=int(num/y)

#levels=[1.0, 1.1, 1.2, 1.3]
levels=None


fig,ax=plt.subplots(x,y)
for idx,result in enumerate(ci2d_result):
    i,j=idx//len(ax[0]),idx % len(ax[0])
    pair,x,y,grid=result[0],result[1],result[2],result[3]
    print(grid)
    grid = grid/chisqr
    # Plot chi-sqr
    cnt=ax[i,j].contour(x,y,grid,levels=levels)
    ax[i,j].set_xlabel(pair[0],horizontalalignment='left')
    #ax[i,j].annotate(f'χsq:{chisqr:.3e}',(0.8,0.8),xycoords='axes fraction')
    ax[i,j].xaxis.set_label_coords(0.1,0.1)
    ax[i,j].set_ylabel(pair[1],verticalalignment='bottom')
    ax[i,j].yaxis.set_label_coords(0.1,0.35)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.82, 0.15, 0.015, 0.7])
cbar_ax.annotate(f'χsq:{chisqr:.3e}',(0,-0.05),xycoords='axes fraction')
fig.colorbar(cnt, cax=cbar_ax)
fig.set_size_inches(3*(len(ax)+1),4*len(ax[0]))
plt.savefig(f'{file}_ci2d.pdf', format="pdf", bbox_inches="tight")
plt.savefig(f'{file}_ci2d.png', format="png", bbox_inches="tight")
plt.show()
