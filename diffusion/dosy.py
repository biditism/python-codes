# importing modules

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



remove_first_point=True

if remove_first_point:
    gradlist=gradlist[1:]
    Bcalc=Bcalc[1:]
    spectra=np.delete(spectra, 0,0)
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

analyse= True

if analyse == False:
    sys.exit()
# =============================================================================
# Select the interested region
# =============================================================================   
spectra_full = spectra
ppm_full=ppm
ppm_min,ppm_max=-2,12
start=np.argwhere(ppm_full>ppm_max)[-1][0]
stop= np.argwhere(ppm_full<ppm_min)[0][0]
ppm=ppm_full[start:stop]
spectra=spectra_full[:,start:stop]

if test:
    plt.plot(ppm,np.transpose(spectra))
    plt.show()

# =============================================================================
# Integrate the whole region
# =============================================================================

Area= np.trapz(spectra)
Area=Area/Area[0]


# =============================================================================
# Integrate the curve slice
# =============================================================================

#Define slicing and area variable
No_of_slices=10

Area_slice=np.zeros((len(gradlist),No_of_slices))
ppm_slice=np.array_split(ppm, No_of_slices)


norm=np.max(spectra)

for a,b in enumerate(spectra):
          
    #Take the slices of the data within the required limits
    slices=np.array_split(b, No_of_slices)
    #freq_slices=np.array_split(freq, No_of_slices)
    
    
    #Integrate each slice of the associated data
    for x,y in enumerate(slices):
        Area_slice[a,x] =  np.trapz(y)

    #     plt.plot(y)
    # plt.show()

    # Some plots for testing purpose
    if test: plt.plot(ppm+a,np.abs(b)+norm*a/len(gradlist)/2)
if test: plt.show()    

Area_slice= Area_slice/Area_slice[0]

if test:
    plt.semilogy(Bcalc, Area_slice)
    plt.show()
# =============================================================================
# Fitting of the integrated curves to find the diffusion coeffecient
# =============================================================================


# Define models, fitting parameters and other required variables


diff= getattr(oth,exp_info['diff_model'])
    
fixed_comp ='const' in exp_info['diff_model']

diff_model,diff_params,components = diff()

fitter = {
    'model': diff_model,
    'params' : diff_params,
    'B' : Bcalc,
    'method':'basinhopping'    
    }

# Fit the whole curve

full_region= (99,Area,fitter,ppm)

fitted_full= oth.diff_fit_single(full_region)
fitted_full_parameters=fitted_full[2].params.valuesdict()

# Write the results to a dataframe
fitted_full_df=pd.DataFrame(fitted_full_parameters,index=[0])
fitted_full_df['BIGDELTA'] = exp_info['BIGDELTA']
fitted_full_df['diff_model']=exp_info['diff_model']
fitted_full_df['Sample'] = exp_info['Sample']
fitted_full_df.to_csv('DOSY_full.csv',index=False)


# Plot the full region fitting
# fit=fitted_full[2]    
# z=fit.eval_components()
# fit.plot(title=f'Diffusion in slice {fitted_full[0]}')
# # for keys,k in z.items():
# #     plt.plot(Bcalc,k,label=keys)
# # plt.yscale('log')
# plt.show()

# Create a list of fitting splace for all the slices

space = [(n,a,fitter,ppm_slice[n]) for n,a in enumerate(Area_slice.T)]

# Prepare the result dictionary
fitted_result=dict()
for x in components:
    fitted_result[f'{x}_diff'] =[]
    fitted_result[f'{x}_frac'] =[]

if fixed_comp:
    fitted_result['fixed_frac'] =[]   
fitted_result['slice_number'] =[]
fitted_result['ppm_mean'] =[]
diffusion_fitted=[]


# Fit and extract the results
for a in space:
    result=oth.diff_fit_single(a)
    diffusion_fitted.append(result)
    fitted_result['slice_number'].append(result[0])
    fitted_result['ppm_mean'].append(np.mean(result[3]))
    
    parameters= result[2].params.valuesdict()
    for x in components:
        fitted_result[f'{x}_diff'].append(parameters[f'{x}_D'])
        fitted_result[f'{x}_frac'].append(parameters[f'{x}_A'])
    if fixed_comp:
        fitted_result['fixed_frac'].append(parameters['fixed_A'])
    
    # result[2].plot(title=f'Diffusion in slice {result[0]}')
  

# Plot the results
fig,ax1 = plt.subplots()

for x in components:
    ax1.scatter(fitted_result['ppm_mean'],fitted_result[f'{x}_diff'])
    ax1.axhline(y=fitted_full_parameters[f'{x}_D'])
plt.yscale('log')
ax2 = ax1.twinx()
ax2.plot(ppm,np.transpose(spectra)/np.max(spectra))
plt.savefig(f"../{exp_info['EXP']}.png", format="png", bbox_inches="tight",dpi=300)
plt.close()

# Write the results to a dataframe
fitted_df=pd.DataFrame(fitted_result)
fitted_df['BIGDELTA'] = exp_info['BIGDELTA']
fitted_df['diff_model']=exp_info['diff_model']
fitted_df['Sample'] = exp_info['Sample']
fitted_df.to_csv('DOSY.csv',index=False)
