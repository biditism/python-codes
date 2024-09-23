# importing modules

import os
# import sys
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

exp=os.getcwd().split('\\')[-1]

dic, spectra = ng.bruker.read_pdata("./pdata/1")


# =============================================================================
# Experimental parameters definition
# =============================================================================


#Experimental parameters
#Read parameters from a CSV file

# Open the CSV file in read mode
with open('../exp_info.csv', 'r') as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file)
    
    # Read the header line
    headers = next(csv_reader)
    
    # Specify the EXPNO value you're looking for
    target_EXPNO = exp
    
    # Iterate through each line in the CSV file
    for line in csv_reader:
        # Find the index of the column named "EXPNO"
        EXPNO_index = headers.index("EXPNO")
        
        # Check if the current line's EXPNO matches the target EXPNO
        if line[EXPNO_index] == target_EXPNO:
            # Create variables dynamically using header names
            variables = {}
            for header, value in zip(headers, line):
                try:
                    # Try converting the value to float
                    value = float(value)
                except ValueError:
                    # If conversion to float fails, keep the value as string
                    pass
                variables[header] = value
                
            # Now you can access each value using its corresponding header as the variable name
            for header, value in variables.items():
                exec(f"{header} = {value!r}")
                print(f"{header}: {value}")
###############################################################################

#Convert delays to second
delta=delta/1000
DELTA=DELTA/1000


# plt.plot(spectra[0]/max(spectra[0]),label=sample)
# plt.legend(loc='upper right')
# plt.close()



# Prefactor constant of -g^2 D in the intensity vs g squared curve
prefactor=  Gamma**2  * delta**2 * (DELTA - delta/3) 


#Load the list of field gradients (Gauss per cm)
gradient=np.loadtxt('./difflist')

#Check if the gradient values are bigger than 100 to determine the unit of gradient
if gradient[-1]>100:
    gradient = gradient /100 #Conversion to Tesla per meter

g2=gradient**2

#Get the spectral width and other related processing parameters
# sw2=dic['acqus']['SW_h']
# dw2=1/sw2

# SF= dic['acqus']['SFO1']


#Define the time axis
#p90=7.4 # pi/2 pulse length in microsecond
#freq=np.fft.fftfreq(len(spectra[0]),dw2)




# =============================================================================
# Integrate the curve slice
# =============================================================================

#Define slicing and area variable
No_of_slices=10

Area=np.zeros((len(gradient),No_of_slices))
start=2900
stop=3800


for a,b in enumerate(spectra):
   
    plt.plot(b[start:stop])
    
    
   
    #Take the slices of the data within the required limits
    slices=np.array_split(b[start:stop], No_of_slices)
    #freq_slices=np.array_split(freq, No_of_slices)
    
    
    #Integrate each slice of the associated data
    for x,y in enumerate(slices):
        Area[a,x] =  np.trapz(y)

    #     plt.plot(y)
    # plt.show()

    # Some plots for testing purpose
    # plt.plot(ppm+a,np.abs(spectra)+5e6*a)
plt.show()    

Area= Area/Area[0]

# plt.semilogy(g2, Area)
# plt.show()

# =============================================================================
# Fitting of the integrated curves to find the diffusion coeffecient
# =============================================================================

# =============================================================================
# Model and parameters for the first component
# =============================================================================
fit_result=[]
Solvent=[]
Probe=[]

# g2=g2[:14]
for n,slice in enumerate(Area.T):
    
    print('Diffusion in slice '+str(n))
    
    #Remove constant background from the data
    y=slice[-4:]
    gel=np.mean(y)
    
    
    # plt.plot(g2,0*g2+gel)
    # plt.scatter(g2,slice)
    # plt.show()
    slice=slice-gel
    
    # slice=slice[:14]
    print(gel)

    diffusion_model = lm.Model(fn.diffusion_decay,prefix='first_',
                               param_names=('D','A'),
                               k=prefactor)
    
    diffusion_params = lm.Parameters()
    
    
    diffusion_params.add('first_D',1e-9,vary=True,min=1e-10,max=1e-8)
    diffusion_params.add('first_A',0.5,vary=True,min=0.0,max=1)  
    
    
    # =============================================================================
    # Model and parameters for the second component
    # =============================================================================
    
    diffusion_model = diffusion_model + lm.Model(fn.diffusion_decay,
                                                 prefix='second_',
                                                 param_names=('D','A'),
                                                 k=prefactor)
       
        
    
    diffusion_params.add('second_D',1e-11,vary=True,min=1e-13,max=1e-10)
    diffusion_params.add('second_A',0.5,vary=True,min=0.0,max=gel)
    
    
    diffusion_fitted= diffusion_model.fit(slice,diffusion_params,
                                          gsquared=g2,
                                          method='basinhopping')
    
    
    fit_result.append(diffusion_fitted)
    print(diffusion_fitted.params.pretty_print())
    diffusion_fitted.plot(title='Diffusion in slice '+str(n))
    
    z=diffusion_fitted.eval_components()
    
    water_subtracted=slice-z['first_']
    
    for keys,k in z.items():
        plt.plot(g2,k,label=keys)
    
    plt.show()

    S=diffusion_fitted.params.valuesdict()['first_D']
    P=diffusion_fitted.params.valuesdict()['second_D']
    Solvent.append(S)
    Probe.append(P)
    
    
    # plt.scatter(g2,Area.T[i],marker='1',label='data')
    # plt.scatter(g2,water_subtracted,marker='2',label='data-water')
    # plt.scatter(g2,water_zero_bkg_subtracted,marker=i,label=i)
    # plt.legend(loc='upper right')
    # plt.yscale('log')
    # plt.ylim(2e-1, 1)
    
    
# plt.show()

comp=['first_','second_','third_']


plt.plot(Solvent,'o')
plt.plot(Probe,'v')
plt.yscale('log')
plt.savefig("../"+exp+".png", format="png", bbox_inches="tight",dpi=300)
plt.show()

diffusion=pd.DataFrame({"Solvent":Solvent,"Probe":Probe})
diffusion.to_csv('../'+exp+'.csv',index=True)
# for idx,a in enumerate(fit_result):
#     result=a.params.valuesdict()
    
    
#     i=0
#     for x in ['first_','second_']:
#         plt.scatter(idx,result[x+'D'],marker=i) 
    
#         i=i+1
    
# plt.yscale('log') 
# plt.show()