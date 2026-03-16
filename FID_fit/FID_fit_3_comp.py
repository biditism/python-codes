#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# Definitions of constants and functions

# In[2]:


def FID_intensity(tau,parameters):

    first_comp = fn.sinc(parameters['first_b']*tau)*fn.T2_decay(tau, T2=parameters['first_T2'], A=parameters['first_A'], beta=parameters['first_beta'])

    second_comp = fn.T2_decay(tau, T2=parameters['second_T2'], A=parameters['second_A'], beta=parameters['second_beta'])

    third_comp = fn.T2_decay(tau, T2=parameters['third_T2'], A=parameters['third_A'], beta=parameters['third_beta'])

    intensity = first_comp + second_comp + third_comp
    return intensity

def derived_parameters(FID_params):
        FID_params.add('total_A', expr='first_A + second_A + third_A',min=0.01)
        FID_params.add('first_frac', expr='first_A/total_A')
        FID_params.add('second_frac', expr='second_A /total_A')
        FID_params.add('third_frac', expr='third_A /total_A')
        FID_params.add('first_a', expr='(2**0.5)/first_T2')
        FID_params.add('first_M2', expr='(first_a**2)+(first_b**2)/3')
        
        return FID_params

if __name__ == '__main__':

    # In[3]:


    a=datetime.now()
    print(a)


    # In[4]:


    ##############################################
    #Variable Declarations
    ##############################################



    exp_info = oth.read_exp_parameters()

    data_cutoff=exp_info['data_cutoff'] #Cutoff for fitting limit


    connectivity=['first','second','third'] #Number of components


    # In[5]:


    #Import the data
    df,file = oth.high_field_FID_rawdata()
    df['Sample']=file


    # In[7]:


    df_main = df.copy() #Backup of original table

    #Cleaning of data

    omit_points=None
    fluctuations=False
    norm_factor=None

    if fluctuations==True:
        norm_factor=np.mean(df['I_real'][0:3])

    df = oth.high_field_FID_clean(df,data_cutoff,omit=omit_points,norm_factor=norm_factor)


    # In[8]:


    ###############################################
    #Define ranges for parameters of elastic components
    ###############################################

    ranges = {
        'first_T2': (0, 0.030), 'first_A': (0.01, 10), 'first_beta': (1, 2), 'first_b': (0, 1000),
        'second_T2': (0.02, 0.100), 'second_A': (0.0001, 10), 'second_beta': (0.8, 2),
        'third_T2': (0.1, 0.500), 'third_A': (0.0001, 10), 'third_beta': (0.8, 2),
    }


    ###############################################
    #Initialize parameters for elastic components
    ###############################################


    FID_params = lm.Parameters()


    # Add parameters to FID_params
    for param, (min_val, max_val) in ranges.items():
        initial_value = (min_val + max_val) / 2  # Default initial value is the midpoint of the range
        FID_params.add(param, value=initial_value, vary=True, min=min_val, max=max_val)

    # Randomize the parameters and get the updated parameters
    FID_params = oth.randomize_parameters(FID_params, ranges)


    # In[9]:
    # Add derived parameters
    FID_params =derived_parameters(FID_params)

    ###############################################
    #Make tail parameters limited
    ###############################################


    # In[10]:


    
    FID_params['first_b'].set(value=200,vary=True)
    FID_params['first_beta'].set(value=2,vary=False)

    FID_params['second_A'].set(value=0,vary=False)
    FID_params['second_T2'].set(value=1,vary=False)
    FID_params['second_beta'].set(value=1,vary=False)

    FID_params['third_beta'].set(value=1,vary=False)

    # Define the difference of T2 and Dres values as parameters
    FID_params.add('diff_T2_1',value=0.1,min=0,max=1)
    FID_params['first_T2'].set(expr='second_T2 * diff_T2_1')


    FID_params.add('diff_T2_2',value=0.1,min=0,max=1)
    FID_params['second_T2'].set(expr='third_T2 * diff_T2_2')


    
    ##############################################
    #Define DQ and MQ models and other variables
    ##############################################

    FID_model=FID_intensity

    tau=df['Time']
    FID=df['I_real']


    ################################################
    #Post fit derived parameters
    ###############################################
    
    

    
    # In[11]:


    FID_fit=lm.Minimizer(oth.fit_single,FID_params,fcn_args=(FID_model,tau,FID))

    # In[12]:


    read= True
    
    if read is True:
        FID_params= oth.load_object('last_fit_result.pckl')
        for x in FID_params:
            FID_params[x].set(vary=False)
        FID_params['first_A'].set(vary=True)
        FID_params['third_A'].set(vary=True)
        FID_fitted= FID_fit.minimize(method='leastsq',params=FID_params)
    else:
        FID_fitted= FID_fit.minimize(method='basinhopping',params=FID_params)

    
    
    oth.write_object(FID_fitted.params,'last_fit_result.pckl')

    print(FID_fitted.params.pretty_print())


    # In[13]:

    # #=============================================================================
    # #Draw a chi-sqr map
    # #=============================================================================

    fit_chi_sqr=FID_fitted.chisqr
    chi_sqr_map=False
    map_parameters= ['first_A']

    if chi_sqr_map is True:

        for parameter in map_parameters:
            if FID_fitted.params[parameter].stderr is None:
                FID_fitted.params[parameter].stderr = abs(FID_fitted.params[parameter].value * 0.1)
            lower = max(min(FID_fitted.params[parameter] - 2*FID_fitted.params[parameter].stderr,FID_fitted.params[parameter].value * 0.5),FID_fitted.params[parameter].min)
            higher = min(max(FID_fitted.params[parameter] + 2*FID_fitted.params[parameter].stderr,FID_fitted.params[parameter].value * 1.5),FID_fitted.params[parameter].max)


            A=np.linspace(lower, higher,num=80)
            fitter={
                'params':FID_params,
                'fixed':parameter,
                'fit':FID_fit,
                'method':'leastsq'
                }
            space= [(fitter,k) for k in A]
            pool= Pool()
            search_result=pool.map(oth.single_point_no_randomize,space)

            #Write Chi-square vs a to the file
            chi_sqr =[(x,y.chisqr/fit_chi_sqr) for (x,y) in search_result]

            chi_df=pd.DataFrame(chi_sqr,columns=['Value','chi_sqr'])
            chi_df['Sample']=file
            splitted_parameter=parameter.split('_', 1)
            chi_df['Fraction']=splitted_parameter[0]
            chi_df['Quantity']=splitted_parameter[1]
            chi_df.to_csv(f'{file}_{parameter}_chisqr.txt',index=False,header=True)
            # np.savetxt(f'{file}_{parameter}_chisqr.txt', chi_sqr,delimiter=',',comments=f'{file}',header=f'{parameter},chi_sqr')

            chi_min=min([y for (x,y) in chi_sqr])
            plt.plot(*zip(*chi_sqr))
            plt.ylim(chi_min*0.99, chi_min*3)
            plt.xlabel(f'{parameter}')
            plt.ylabel("Chi-Square")
            plt.savefig(f'{file}_{parameter}_chisqr.pdf', format="pdf", bbox_inches="tight")
            plt.savefig(f'{file}_{parameter}_chisqr.png', format="png", bbox_inches="tight")
            plt.close()

            #Pickel the minimizer result object
            oth.write_object(search_result,f'{file}_{parameter}_search.pckl')




    # =============================================================================
    # Determine individual fits
    # =============================================================================

    result=FID_fitted.params.valuesdict()



    components_FID = {}
    for x in connectivity:
        components_FID[f'{x}_'] = fn.T2_decay(tau, result[f'{x}_T2'],result[f'{x}_A'], result[f'{x}_beta'])

    #Add sinc oscillations for rigid component
    components_FID['first_']=components_FID['first_'] * fn.sinc(tau*result['first_b'])

    # Full FID Fit
    fitted_points_FID = {
        **components_FID,
        'Full_Fit_': FID_model(tau,result)
    }



    # In[15]:


    ##############################################
    #Plot and save graphs
    ##############################################

    oth.plot_results_FID(tau, FID, fitted_points_FID, file)


    # In[16]:


    ##############################################
    #Create a result dataframe and write outputs to file
    ##############################################

    df_result= oth.files_report_FID(df,file,fitted_points_FID,FID_fitted)

    b=datetime.now()
    print(b)
    print(f'Execution time is {b-a}')






