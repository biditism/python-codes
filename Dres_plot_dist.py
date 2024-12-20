# importing modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lmfit as lm
import mathematical_functions as fn
import other_functions as oth
import os
import shutil
from scipy.stats import lognorm

file= os.getcwd()
file = file.replace('\\','/')
file= file.split('/')[-1]



minimized=oth.load_object(f'{file}_minimized.pckl')

result=minimized.params.valuesdict()

connectivity=['first','second']

#Plot of the Dres distribution
prob = {}
Dres = {}
lower,upper=(0.0001,0.3)
for x in connectivity:
    Dres[x] =np.linspace(lower,upper,1000)
    prob[x]=lognorm.pdf(Dres[x], result[f'{x}_sigma'],scale=result[f'{x}_Dres'])
    top=max(prob[x])
    prob[x] =prob[x]/top
    a=result[f'{x}_A']
    plt.text(result[f'{x}_Dres'],1,f'a={a:.2g}')
    plt.plot(Dres[x], prob[x],label=x)
plt.show()

