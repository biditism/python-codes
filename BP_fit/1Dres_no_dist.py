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

#############################################
#Definitions of constants and functions


def DQ_intensity(tau,parameters):  
        
    first_comp= fn.DQ_1_Dres_1_T2(tau, Dres=parameters['first_Dres'], T2=parameters['first_T2'], A=parameters['first_A'], beta=parameters['first_beta'])   
    
    intensity= first_comp

    return intensity
    
def MQ_intensity_with_tail(tau,parameters):
    
    elastic_comp=MQ_intensity_without_tail(tau,parameters)
    
    tail_comp = fn.T2_decay(tau, T2=parameters['tail_T2'], A=parameters['tail_A'], beta=parameters['tail_beta'])
    
    intensity = elastic_comp + tail_comp
    return intensity

def MQ_intensity_without_tail(tau,parameters):
    
    first_comp = fn.T2_decay(tau, T2=parameters['first_T2'], A=parameters['first_A'], beta=parameters['first_beta'])
    
    intensity = first_comp 
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
DQ_cutoff=exp_info['DQ_cutoff'] #Final point for DQ curve fitting

tail_freedom=0.1 #Percentage freedeom for tail to vary later


connectivity=['first'] #Number of components

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

omit_points=None

df = oth.clean(df,data_cutoff,DQ_cutoff,omit=omit_points)


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
    'first_Dres': (0.8, 0.0001), 'first_T2': (0, tail_result['T2']), 'first_A': (0.01, 0.99), 'first_beta': (0.8, 2),
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
    
DQ_params.add('diff_T2_1',value=0.1,min=0,max=1)
DQ_params['first_T2'].set(expr='tail_T2 * diff_T2_1')

##############################################
#Define DQ and MQ models and other variables
##############################################

tau=df['Time']
DQ=df['I_DQ']

IDQ_model=DQ_intensity
 


DQ_lim= len(df[df['Time']<=DQ_cutoff].index)
 
if tail_subtraction:
    IMQ_model=MQ_intensity_without_tail
    MQ=df['I_MQ_no_tail']
    fitMQ=MQ[:DQ_lim]
    fittau=tau[:DQ_lim]
    fitDQ=DQ[:DQ_lim]
else:
    IMQ_model=MQ_intensity_with_tail
    MQ=df['I_MQ']
    fitMQ=MQ
    fittau=tau
    fitDQ=DQ




#Simultaneous fitting of DQ and MQ curve
sim_fit=lm.Minimizer(
    oth.fit_simultaneous,DQ_params,
    fcn_args=(fittau,fittau[:DQ_lim],fitDQ[:DQ_lim],fitMQ,IDQ_model,IMQ_model,T2_penalty,Dres_penalty))

# #=============================================================================
# #Initial fit for parameter initialization
# #=============================================================================
read= False

if read is True:
    DQ_params= oth.load_object('last_fit_result.pckl')
    sim_fitted= sim_fit.minimize(method='leastsq',params=DQ_params)
else:
    sim_fitted= sim_fit.minimize(method='basinhopping',params=DQ_params)
    oth.write_object(sim_fitted.params,'last_fit_result.pckl')

  
temp=sim_fitted.params.valuesdict()

change=[]
for x in connectivity:
    for y in ["_Dres","_T2"]:
        change.append(f'{x}{y}')

update_list={key: temp[key] for key in change}
   
DQ_params=oth.update_bounds(DQ_params, update_list)

  
search=False
repeat=5

if search is True:
    A=np.arange(0.1, 0.9,0.01)
    parameter= 'first_A'
    fitter={
        'params':DQ_params,
        'fixed':parameter,
        'range':ranges,
        'fit':sim_fit,
        'repeat':repeat
        }
    space= [(fitter,k) for k in A]
    pool= Pool()
    search_result=pool.map(oth.single_point,space)

#Write Chi-square vs a to the file
    chi_sqr =[(x,y.chisqr) for (x,y) in search_result]
    
    np.savetxt(file+"_chisqr.txt", chi_sqr,delimiter=',',comments='',header='A,chi_sqr')
    
    chi_min=min([y for (x,y) in chi_sqr])
    plt.plot(*zip(*chi_sqr))
    plt.ylim(chi_min*0.99, chi_min*1.3)
    plt.savefig(file+'_chisqr.pdf', format="pdf", bbox_inches="tight")
    plt.savefig(file+'_chisqr.png', format="png", bbox_inches="tight")
    plt.close()

    #Pickel the minimizer result object
    oth.write_object(search_result,file+'_search.pckl')
    np.argmin([y for (x,y) in chi_sqr])

    sim_fitted= search_result[np.argmin([y for (x,y) in chi_sqr])][1]


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
    'Full_Fit_': IDQ_model(fittau,result)
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
    'Full_Fit_': IMQ_model(fittau,result)
}



##############################################
#Plot and save graphs
##############################################

oth.plot_results(tau, DQ, MQ, DQ_cutoff, fitted_points_DQ, fitted_points_MQ, file)


##############################################
#Create a result dataframe and write outputs to file
##############################################

df_result= oth.files_report(df,file,fitted_points_DQ,fitted_points_MQ,sim_fitted)

##############################################
#Calculate confidence interval
##############################################

calculate_ci=False

ci_params=['first_A','first_Dres','first_T2']
ci2d_pairs=[]
for i in range(len(ci_params)):
    for j in range(i+1,len(ci_params)):
        ci2d_pairs.append((ci_params[i],ci_params[j]))


if calculate_ci is True:
    for p in sim_fitted.params:
        sim_fitted.params[p].stderr = abs(sim_fitted.params[p].value * 0.2)


    try:
        ci,trace=lm.conf_interval(sim_fit, sim_fitted,trace=True,p_names=ci_params)

    except Exception as error:
        print("CI not calculated",error)
    else:
        lm.ci_report(ci)

        #Write fit report to the file
        file1 = open(file+"_conf_interval.txt", "w")
        print(lm.ci_report(ci),file=file1)
        file1.close()

        #Pickel the confidence interval object
        oth.write_object(ci,file+'_ci.pckl')
        oth.write_object(trace,file+'_ci-trace.pckl')

    ci_dict={
        'minimizer': sim_fit,
        'result': sim_fitted
    }
    ci_space= [(ci_dict,k) for k in ci2d_pairs]
    pool= Pool()
    ci2d_result=pool.map(oth.single_ci2d,ci_space)

    #Pickel the confidence interval 2d object
    oth.write_object(ci2d_result,file+'_ci2d.pckl')

    fig,ax=plt.subplots(len(ci2d_result)//3,3)

    for idx,result in enumerate(ci2d_result):
        i,j=idx//3,idx % 3
        pair,x,y,grid=result[0],result[1],result[2],result[3]
        # Plot chi-sqr
        ax[i,j].contour(x,y,grid)
        ax[i,j].set_xlabel(pair[0],horizontalalignment='left')
        ax[i,j].xaxis.set_label_coords(0.1,0.1)
        ax[i,j].set_ylabel(pair[1],verticalalignment='bottom')
        ax[i,j].yaxis.set_label_coords(0.1,0.35)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.15, 0.015, 0.7])
    fig.colorbar(plt.cm.ScalarMappable(), cax=cbar_ax)
    fig.set_size_inches(16,12)
    plt.savefig(f'{file}_ci2d.pdf', format="pdf", bbox_inches="tight")
    plt.savefig(f'{file}_ci2d.png', format="png", bbox_inches="tight")
    plt.close()


b=datetime.now()
print(b)

print(f'Execution time is {b-a}')