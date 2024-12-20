# importing modules

import os
import sys
import csv
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

if exp_info['EXP_TYPE'] != 'CPMG':
    
    print(os.path.basename(os.getcwd()),'Experiment not marked as CPMG')
    
    sys.exit()


dic,spectra = ng.bruker.read_pdata('./pdata/1/',acqus_files=["acqus", "acqu2s"])

vclist=dic['acqus']['VCLIST']
n_loops=np.squeeze(pd.read_csv(f'./lists/vc/{vclist}',header=None))

len(n_loops)
spectra=np.delete(spectra, range(len(n_loops),len(spectra)),0)
# # =============================================================================
# # Experimental parameters definition
# # =============================================================================

ppm_range= dic['procs']['SW_p']/dic['procs']['SF']
ppm_ref = dic['procs']['OFFSET']
ppm=np.array([-1*(i+(1/2))/dic['procs']['SI']*ppm_range for i in range(dic['procs']['SI'])])+ppm_ref

p180= dic['acqus']['P'][2]
d20=dic['acqus']['D'][20]

cpmg_period= p180 /1000 + 2* d20 * 1000

echo_delay=cpmg_period*n_loops

if abs(cpmg_period-exp_info['mix_time'])>0.0001:
    raise ValueError('CPMG time from data and supplied value do not match.')
    


df= pd.DataFrame(spectra.real,columns=ppm)
df['tau']= round(echo_delay,1)
df=pd.melt(df,id_vars='tau',var_name='ppm',value_name='Intensity')
df['Sample'] = exp_info['Sample']
df['CPMG_period'] = round(cpmg_period,3)
df.to_csv('CPMG.csv',index=False)

