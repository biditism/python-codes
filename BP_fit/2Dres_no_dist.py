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

#############################################
#Definitions of constants and functions


def DQ_intensity(tau, tail_A,
                 first_Dres,first_T2,first_beta,first_A,
                 second_Dres,second_T2,second_beta,second_A):  
    
    first_comp= fn.DQ_1_Dres_1_T2(tau, first_Dres, first_T2, first_A, first_beta)   
    
    second_comp= fn.DQ_1_Dres_1_T2(tau, second_Dres, second_T2, second_A, second_beta)
    
    intensity= first_comp + second_comp

    return intensity
    
def MQ_intensity_with_tail(tau, tail_T2, tail_beta, tail_A,
                           first_T2,first_beta,first_A,
                           second_T2,second_beta,second_A):
    
    first_comp = fn.T2_decay(tau, first_T2,first_A,first_beta)
    
    second_comp = fn.T2_decay(tau, second_T2,second_A,second_beta)
    
    tail_comp = fn.T2_decay(tau, tail_T2,tail_A,tail_beta)
    
    intensity = first_comp + second_comp + tail_comp
    return intensity

def MQ_intensity_without_tail(tau, tail_A,
                           first_T2,first_beta,first_A,
                           second_T2,second_beta,second_A):
    
    first_comp = fn.T2_decay(tau, first_T2,first_A,first_beta)
    
    second_comp = fn.T2_decay(tau, second_T2,second_A,second_beta)
    
    intensity = first_comp + second_comp
    return intensity
#############################################


##############################################
#Variable Declarations
##############################################

data_cutoff=None #Cutoff for excess data points
tail_cutoff=None #Start point for tail fitting
DQ_cutoff=40 #Final point for DQ curve fitting

tail_freedom=0.1 #Percentage freedeom for tail to vary later


connectivity=['first','second'] #Number of components

#Define overlap penalty
T2_penalty=dict(amount=1, width=1)
Dres_penalty=dict(amount=1, width=0.005)


tail_subtraction=False

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
    'first_Dres': (0.0001, 0.5), 'first_T2': (0.5, tail_result['T2']), 'first_A': (0.01, 0.99), 'first_beta': (0.5, 2),
    'second_Dres': (0.0001, 0.5), 'second_T2': (0.5, tail_result['T2']), 'second_A': (0.01, 0.99), 'second_beta': (0.5, 2),
    'tail_T2': (tail_result['T2'] * (1 - tail_freedom), tail_result['T2'] * (1 + tail_freedom)),
    'tail_A': (tail_result['A'] * (1 - tail_freedom), tail_result['A'] * (1 + tail_freedom)),
    'tail_beta': (0.5, 2)
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

###############################################
#Make tail parameters limited
###############################################

DQ_params['tail_beta'].set(value=1,vary=False)

if tail_subtraction:
    DQ_params['tail_T2'].set(value=tail_result['T2'],vary=False)
    DQ_params['tail_A'].set(value=tail_result['A'],vary=False)
    


# Define the difference of T2 and Dres values as parameters
DQ_params.add('diff_Dres_1', expr='first_Dres-second_Dres')
DQ_params.add('diff_T2_1', expr='second_T2-first_T2')

##############################################
#Define DQ and MQ models and other variables
##############################################

tau=df['Time']
DQ=df['I_DQ']

IDQ_model=lm.Model(DQ_intensity)
 


DQ_lim= len(df[df['Time']<=DQ_cutoff].index)
 
if tail_subtraction:
    IMQ_model=lm.Model(MQ_intensity_without_tail)
    MQ=df['I_MQ_no_tail']
    fitMQ=MQ[:DQ_lim]
    fittau=tau[:DQ_lim]
    fitDQ=DQ[:DQ_lim]
else:
    IMQ_model=lm.Model(MQ_intensity_with_tail)
    MQ=df['I_MQ']
    fitMQ=MQ
    fittau=tau
    fitDQ=DQ




# #=============================================================================
# #Single fit for parameter initialization
# #=============================================================================


#Simultaneous fitting of DQ and MQ curve
sim_fit=lm.Minimizer(
    oth.fit_simultaneous,DQ_params,
    fcn_args=(fittau,fitDQ,fitMQ,IDQ_model,IMQ_model,T2_penalty,Dres_penalty,DQ_lim),
    max_nfev=10000000)

search=False
repeat=5

if search is True:
    A=np.arange(0.1, 0.9,0.01)
    parameter= 'first_A'
    search_result=[]

    for x in A:
        DQ_params = oth.randomize_parameters(DQ_params, ranges)
        DQ_params[parameter].set(value=x,vary=False)
        lowest=sim_fit.minimize(method='leastsq',params=DQ_params)
        for x in range(repeat-1):
            temp =sim_fit.minimize(method='leastsq',params=DQ_params)
            if temp.chisqr < lowest.chisqr:
                lowest=temp
        search_result.append(lowest)
    
    #Write Chi-square vs a to the file
    chi_sqr =[a.chisqr for a in search_result]
    
    np.savetxt(file+"_chisqr.txt", list(zip(A,chi_sqr)),delimiter=',',comments='',header='A,chi_sqr')
      
    plt.plot(A, chi_sqr)
    plt.savefig(file+'_chisqr.pdf', format="pdf", bbox_inches="tight")
    plt.savefig(file+'_chisqr.png', format="png", bbox_inches="tight")
    plt.show()

    #Pickel the minimizer result object
    oth.write_object(search_result,file+'_search.pckl')
    

    sim_fitted= search_result[chi_sqr.index(min(chi_sqr))]

else:

    sim_fitted= sim_fit.minimize(method='leastsq',params=DQ_params)


print(sim_fitted.params.pretty_print())



# =============================================================================
# Determine individual fits
# =============================================================================

result=sim_fitted.params.valuesdict()

components_DQ = {}
for x in connectivity:
    components_DQ[x] = fn.DQ_1_Dres_1_T2(fittau, result[f'{x}_Dres'], result[f'{x}_T2'], 
                                         result[f'{x}_A'], result[f'{x}_beta'])

# Full DQ Fit
fitted_points_DQ = {
    **components_DQ,
    'Full_Fit_': IDQ_model.eval(params=sim_fitted.params, tau=fittau)
}

# Calculate MQ components
components_MQ = {}
for x in connectivity:
    components_MQ[x] = fn.T2_decay(fittau, result[f'{x}_T2'], result[f'{x}_A'], result[f'{x}_beta'])

# Include tail component if not tail subtraction
if not tail_subtraction:
    components_MQ['tail_'] = fn.T2_decay(fittau, result['tail_T2'], result['tail_A'], result['tail_beta'])

# Full MQ Fit
fitted_points_MQ = {
    **components_MQ,
    'Full_Fit_': IMQ_model.eval(params=sim_fitted.params, tau=fittau)
}



##############################################
#Plot and save graphs
##############################################

oth.plot_results(tau, DQ, MQ, DQ_cutoff, fitted_points_DQ, fitted_points_MQ, file)


##############################################
#Calculate confidence interval
##############################################

calculate_ci=False

if calculate_ci is True:
    for p in sim_fitted.params:
        if sim_fitted.params[p].stderr is None:
            sim_fitted.params[p].stderr = abs(sim_fitted.params[p].value * 0.1)

    ci,trace=lm.conf_interval(sim_fit, sim_fitted,trace=True)

    lm.printfuncs.report_ci(ci)

    #Pickel the confidence interval object
    oth.write_object(ci,file+'_ci.pckl')
    oth.write_object(trace,file+'_ci-trace.pckl')

##############################################
#Create a result dataframe and write outputs to file
##############################################

df_result= oth.files_report(df,file,fitted_points_DQ,fitted_points_MQ,sim_fitted)