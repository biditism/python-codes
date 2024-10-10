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
from scipy.stats import lognorm
from multiprocessing import Pool

#############################################
#Definitions of constants and functions

def DQ_intensity(tau,parameters):  
        
    first_comp= fn.abragam_dist(tau, Dres_med=parameters['first_Dres'], Dres_sigma=parameters['first_sigma'], A=parameters['first_A'])   
    
    intensity= first_comp

    return intensity
    

#############################################

a=datetime.now()
print(a)
##############################################
#Variable Declarations
##############################################

exp_info = oth.read_exp_parameters()

data_cutoff=exp_info['data_cutoff'] #Cutoff for excess data points
tail_cutoff=exp_info['tail_cutoff'] #Start point for tail fitting
DQ_cutoff=tail_cutoff #Final point for DQ curve fitting



connectivity=['first'] #Number of components


##############################################
#Cleaning of Data
##############################################

#Import the data
df,file = oth.rawdata()


df_main = df.copy() #Backup of original table

#Cleaning of data

df = oth.clean(df,data_cutoff)


##############################################
#Tail substraction
##############################################

tail_fitted = oth.tail(df,tail_cutoff,vary_beta=False)

tail_result=tail_fitted.best_values


#Write fit report and subtracted data to file
df.to_csv(file+'_no_tail.csv',index=False)


file1 = open(file+"_tail_report.txt", "w")
print(lm.fit_report(tail_fitted),file=file1)
file1.close()

###############################################
#Define ranges for parameters of elastic components
###############################################

ranges = {
    'first_Dres': (0.0001, 0.5), 'first_sigma': (0.001, 5), 'first_A': (0.01, 0.99)
}


###############################################
#Initialize parameters for elastic components
###############################################


DQ_params = lm.Parameters()


# Add parameters to DQ_params
for param, (min_val, max_val) in ranges.items():
    initial_value = (min_val + max_val) / 2  # Default initial value is the midpoint of the range
    DQ_params.add(param, value=initial_value, vary=True, min=min_val, max=max_val)

# Randomize the parameters and get the updated parameters
DQ_params = oth.randomize_parameters(DQ_params, ranges)


##############################################
#Define DQ and MQ models and other variables
##############################################

tau=df['Time']
DQ=df['I_DQ']
nDQ=df['I_nDQ']
MQ=df['I_MQ']

IDQ_model=DQ_intensity
 
DQ_lim= len(df[df['Time']<=DQ_cutoff].index)


# #=============================================================================
# #Define fittng function
# #=============================================================================

fit=lm.Minimizer(oth.fit_single,DQ_params,fcn_args=(IDQ_model,tau[:DQ_lim],nDQ[:DQ_lim]))
  
fitted= fit.minimize(method='basinhopping',params=DQ_params)


print(fitted.params.pretty_print())


# =============================================================================
# Determine individual fits
# =============================================================================

result=fitted.params.valuesdict()

components_nDQ = {}
for x in connectivity:
    components_nDQ[x] = fn.abragam_dist(tau, result[f'{x}_Dres'], result[f'{x}_sigma'], 
                                         result[f'{x}_A'])

# Full DQ Fit
fitted_points_nDQ = {
    **components_nDQ,
    'Full_Fit_': IDQ_model(tau,result)
}



##############################################
#Plot and save graphs
##############################################


# Plot DQ data with linear-log scale
plt.xlim(0, DQ_cutoff * 1.5)
oth.plotmq(tau, DQ, MQ, nDQ, y_axis='log', save=file+'linlog', **fitted_points_nDQ)

# Plot DQ data with linear-linear scale
plt.xlim(0, DQ_cutoff * 1.5)
oth.plotmq(tau, DQ, MQ, nDQ, y_axis='log', save=file+'linlog', **fitted_points_nDQ)


#Plot of the Dres distribution
prob = {}
Dres = {}
for x in connectivity:
    lower,upper=lognorm.interval(0.99, result[f'{x}_sigma'],scale=result[f'{x}_Dres'])
    Dres[x] =np.linspace(lower,upper,1000)
    prob[x] =lognorm.pdf(Dres[x], result[f'{x}_sigma'],scale=result[f'{x}_Dres'])
    plt.plot(Dres[x], prob[x],label=x)    
plt.savefig(file+'_Dres.pdf', format="pdf", bbox_inches="tight")
plt.savefig(file+'_Dres.png', format="png", bbox_inches="tight")
plt.close()

##############################################
#Create a result dataframe and write outputs to file
##############################################

df_result= oth.files_report_InDQ(df,file,fitted_points_nDQ,fitted)

##############################################
#Calculate confidence interval
##############################################

calculate_ci=False

if calculate_ci is True:
    for p in fitted.params:
        if fitted.params[p].stderr is None:
            fitted.params[p].stderr = abs(fitted.params[p].value * 0.1)

    ci,trace=lm.conf_interval(fit, fitted,trace=True)

    lm.printfuncs.report_ci(ci)

    #Pickel the confidence interval object
    oth.write_object(ci,file+'_ci.pckl')
    oth.write_object(trace,file+'_ci-trace.pckl')

b=datetime.now()
print(b)

print(f'Execution time is {b-a}')
