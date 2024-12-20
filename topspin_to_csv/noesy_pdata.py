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

if exp_info['EXP_TYPE'] != 'NOESY':
    
    print(os.path.basename(os.getcwd()),'Experiment not marked as NOESY')
    
    sys.exit()

dic,spectra = ng.bruker.read_pdata('./pdata/1/')

# # =============================================================================
# # Experimental parameters definition
# # =============================================================================
ppm_axis=dict()
for key,value  in dic.items():
    ppm_range= value['SW_p']/value['SF']
    ppm_ref = value['OFFSET']
    ppm=np.array([-1*(i+(1/2))/value['SI']*ppm_range for i in range(value['SI'])])+ppm_ref
    ppm_axis[key]=ppm

df= pd.DataFrame(spectra,columns=ppm_axis['procs'])
df['F1']=ppm_axis['proc2s']
df=pd.melt(df,id_vars='F1',var_name='F2',value_name='Intensity')
df['Sample'] = exp_info['Sample']
df['Mixing_time'] = exp_info['mix_time']
df.to_csv('NOESY.csv',index=False)

