# importing modules

import os
import sys
import csv# importing modules

import os
import sys
import nmrglue as ng
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lmfit as lm
import mathematical_functions as fn
import other_functions as oth
from multiprocessing import Pool
# from random import uniform

# =============================================================================
# Definition of constants
# =============================================================================

k = 1.38e-23 #Boltzmann's constatnt Joule per Kelvin
T = 30+273.15 #Experimental temperature Kelvin
eta = 1e-3 #Viscosity Pascal.second
Gamma = 267.522e6 #Magnetogyric ratio of proton in radian per second per Tesla



# =============================================================================
# Import and preprocess the data
# =============================================================================

#Get the current folder and processed data

exp_info = oth.read_exp_parameters()

if exp_info['EXP_TYPE'] not in ['diffSte','diffSe']:
    
    print(os.path.basename(os.getcwd()),'Experiment not marked as diffusion')
    
    sys.exit()


BIGDELTA = exp_info['BIGDELTA']/1000 #Diffusion time in second
smalldelta = exp_info['smalldelta']/1000 #Effective gradient duration in second

dic,spectra = ng.bruker.read_pdata('./pdata/1/',acqus_files=["acqus", "acqu2s"])

test=False
if exp_info['Repetition']:
    spectra_all=dict()
    
    for x in range(8):
        run= int(exp_info['EXP']) + x
        dic1,spectra_all[x] = ng.bruker.read_pdata(f'../{run}/pdata/1/',acqus_files=["acqus", "acqu2s"])
    spectra = sum(spectra_all.values())
    
    if test:
        spectra_check=[spectra/a for a in spectra_all.values()]
        for a in spectra_check: plt.plot(np.transpose(a))
        plt.show()

gradlist=np.squeeze(pd.read_csv('./gradlist',header=None)) / 100 #Gradient strength in Tesla per meter


Blist=np.squeeze(pd.read_csv('./difflist',header=None))

Bcalc= Gamma**2 * gradlist**2 * smalldelta**2 * (BIGDELTA - smalldelta / 3)

spectra=np.delete(spectra, range(len(gradlist),len(spectra)),0)
# # =============================================================================
# # Create ppm axis
# # =============================================================================

ppm=oth.make_ppm_axis1d(dic)


# plt.plot(ppm,np.transpose(spectra))
# plt.show()
 # # ==========================================================================
 # # Write data to a dataframe
 # # ==========================================================================


df= pd.DataFrame(spectra.real,columns=ppm)
df['gradient']= round(gradlist,4)
df=pd.melt(df,id_vars='gradient',var_name='ppm',value_name='Intensity')
df['Sample'] = exp_info['Sample']
df['BIGDELTA'] = exp_info['BIGDELTA']
df['smalldelta']=exp_info['smalldelta']

if exp_info['Repetition']:
    df.to_csv('diffusion_combined.csv',index=False)
else:
    df.to_csv('diffusion.csv',index=False)

import nmrglue as ng
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lmfit as lm
import mathematical_functions as fn
import other_functions as oth
# from random import uniform

# =============================================================================
# Definition of constants
# =============================================================================

k = 1.38e-23 #Boltzmann's constatnt Joule per Kelvin
T = 32+273.15 #Experimental temperature Kelvin
eta = 1e-3 #Viscosity Pascal.second
Gamma = 267.522e6 #Magnetogyric ratio of proton in radian per second per Tesla



# =============================================================================
# Import and preprocess the data
# =============================================================================

#Get the current folder and processed data

exp_info = oth.read_exp_parameters()

if exp_info['EXP_TYPE'] not in ['diffSte','diffSe']:
    
    print(os.path.basename(os.getcwd()),'Experiment not marked as diffusion')
    
    sys.exit()


BIGDELTA = exp_info['BIGDELTA']/1000 #Diffusion time in second
smalldelta = exp_info['smalldelta']/1000 #Effective gradient duration in second

dic,spectra = ng.bruker.read_pdata('./pdata/1/',acqus_files=["acqus", "acqu2s"])


gradlist=np.squeeze(pd.read_csv('./gradlist',header=None)) / 100 #Gradient strength in Tesla per meter


Blist=np.squeeze(pd.read_csv('./difflist',header=None))

Bcalc= Gamma**2 * gradlist**2 * smalldelta**2 * (BIGDELTA - smalldelta / 3)



spectra=np.delete(spectra, range(len(gradlist),len(spectra)),0)
# # =============================================================================
# # Experimental parameters definition
# # =============================================================================

ppm=oth.make_ppm_axis1d(dic)


# plt.plot(ppm,np.transpose(spectra))
# plt.show()

# p90= dic['acqus']['P'][1]
# p180= 2*p90


# if abs(cpmg_period-exp_info['mix_time'])>0.0001:
#     raise ValueError('CPMG time from data and supplied value do not match.')
    


df= pd.DataFrame(spectra.real,columns=ppm)
df['gradient']= round(gradlist,4)
df=pd.melt(df,id_vars='gradient',var_name='ppm',value_name='Intensity')
df['Sample'] = exp_info['Sample']
df['BIGDELTA'] = exp_info['BIGDELTA']
df['smalldelta']=exp_info['smalldelta']
df.to_csv('diffusion.csv',index=False)

