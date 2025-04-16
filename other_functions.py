###################################################
# This is my personal collection of non mathematical functions
###################################################


###################################################
# Importing of modules
###################################################

import numpy as np
import matplotlib.pyplot as plt
import lmfit as lm
import pandas as pd
import os
import pickle
import mathematical_functions as fn
import shutil
from datetime import datetime

###################################################
# Declaration of functions
###################################################

#Overlap penalty for values
def overlap_penalty(diff, amount=0, width=1):
    penalty = 1
    for x in diff:
        penalty = penalty * fn.heaviside(-x, amount, 1/width)#Punish when the difference is negative
    return penalty


#Get the list of all folder in a location which has a file with fixed ending
def file_list(ending,root_folder=None,exclude=['temp']):
    file=[]
    if root_folder is None:
        root_folder=os.getcwd()
    for root,dirs, files in os.walk(root_folder):
        dirs[:] = [d for d in dirs if d not in exclude]
        for each in files:
            if each.endswith(ending):
                file.append(each[:-len(ending)])
    return file


#Get the foldername and the file with Baum-Pines data
def rawdata(filename="BP_303.txt",sample=None):
    if sample == None:
        sample= os.getcwd()
        print(sample)
        sample = sample.replace('\\','/')
        sample= sample.split('/')[-1]
    df = pd.read_csv(filename, sep='\t',header=None,names=['Time','I_ref','I_DQ','Im'])
    
    #delete old files from directory
    for f in os.listdir(os.curdir):
        if f.startswith(sample):
            os.remove(f)
    
    return df,sample

#Normalization to the first point of the data and cutoff extra data
def clean(df,cutoff=None,DQ_cutoff=None,omit=None,k=4,norm_factor=None):
    
    
    #Remove the unnecessary Im axis
    df=df.drop(['Im'],axis=1)
    
    #Remove points with artefacts
    if omit is not None:
        print("No of experimental points:",df.shape[0])
        for x in omit:
            df = df.drop(x-1)
        print("No of experimental points after artefact removal:",df.shape[0])
    
    a=0
    if df['Time'][0] == 0:
        print('Measured zero time Iref=',df['I_ref'][0]) 
        k=k+1
        a=a+1
    y=df['I_ref'][a:k]
    x=df['Time'][a:k]
    m,c=np.polyfit(x, y, 1)
    print('Extrapolated zero time Iref=',c)           
    
    if df['Time'][0] != 0:
        df.loc[-1] = [0,c,0]  # add 0 time point
        df.index = df.index + 1  # shifting index
        df.sort_index(inplace=True)
    
    #test()
    
    if DQ_cutoff is not None:
        df.loc[df.Time > DQ_cutoff, 'I_DQ'] = 0
    
    if norm_factor==None: norm_factor = df['I_DQ'][0] + df['I_ref'][0]
    df['I_DQ'] = df['I_DQ'] / norm_factor
    df['I_ref'] = df['I_ref'] / norm_factor
    df['I_MQ'] = df['I_DQ'] + df['I_ref']
    df['I_diff'] = df['I_ref'] - df['I_DQ']
    if cutoff==None:
        plotmq(df['Time'],df['I_DQ'],df['I_ref'])
    
    if cutoff==None:
        finish=1
        while finish ==1:
            cutoff = float(input('Enter the time cutoff for further calculations'))
            df_check = df[df['Time'] <= cutoff].copy()
            plotmq(df_check['Time'],df_check['I_DQ'],df_check['I_ref'],y_axis='log',show=True)
            finish = int(input('Press 1 if you want another cutoff'))

    df_new = df[df['Time'] <= cutoff].copy()

    if cutoff==None:
        plotmq(df_new['Time'],df_new['I_DQ'],df_new['I_ref'],y_axis='log')
    
    
    
    return df_new

#Scatter plot of multiple files
def plotmq(tau,*args,y_axis='linear',save=None,show=False,**kwargs):
    for I in args:
        plt.scatter(tau,I)
    for key,values in kwargs.items():
        plt.plot(tau,values,label=key)
        plt.legend(loc='upper right')
    plt.yscale(y_axis)
    xmin, xmax, ymin, ymax = plt.axis()
    plt.ylim(bottom=0.001,top=min(ymax,1.1))
    if y_axis=='linear':
        plt.ylim(bottom=0)
    if save is not None:
        plt.savefig(save+".pdf", format="pdf", bbox_inches="tight")
        plt.savefig(save+".png", format="png", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()



#Grab only required parameters from the parameter set
def grab(parameter,string):
    dictionary=parameter.valuesdict()
    new=[value for key,value in dictionary.items() if string in key]
    return np.asarray(new)



##Stop program because it is being tested
def test(status=None):
    if status == None: raise ValueError('Exiting because program is set to test')

##Dump the object to file
def write_object(name, filename):
    f = open(filename, 'wb')
    pickle.dump(name, f)
    f.close()

def load_object(filename):
    f = open(filename, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj

def randomize_parameters(params, ranges):
    """
    Randomizes the starting values of all parameters within their specified ranges.

    :param params: lmfit.Parameters object
    :param ranges: Dictionary containing the range for each parameter
    :return: Updated lmfit.Parameters object with randomized values
    """
    for param in params:
        if param in ranges:
            min_val, max_val = ranges[param]
            rand_val = np.random.uniform(min_val, max_val)
            params[param].set(value=rand_val)
    return params

def plot_results(tau, DQ, MQ, DQ_cutoff, fitted_points_DQ, fitted_points_MQ, file):
    """
    Plot the results of the fitted models.

    :param tau: Array of tau values
    :param DQ: Array of DQ data points
    :param MQ: Array of MQ data points
    :param DQ_cutoff: Cutoff value for DQ
    :param fitted_points_DQ: Dictionary of fitted points for DQ model
    :param fitted_points_MQ: Dictionary of fitted points for MQ model
    :param file: Filename prefix for saving the plots
    """
    # Plot DQ data with log-log scale
    plt.xscale('log')
    plotmq(tau, DQ, y_axis='log', save=file+'_IDQ_loglog', **fitted_points_DQ)
    
    # Plot DQ data with linear-log scale
    plt.xscale('linear')
    plt.xlim(0, DQ_cutoff * 1.5)
    plotmq(tau, DQ, y_axis='log', save=file+'_IDQ_linlog', **fitted_points_DQ)
    
    # Plot MQ data with log-log scale
    plt.xscale('log')
    plotmq(tau, MQ, y_axis='log', save=file+'_IMQ_loglog', **fitted_points_MQ)
    
    # Plot MQ data with linear-log scale
    plt.xscale('linear')
    plotmq(tau, MQ, y_axis='log', save=file+'_IMQ_linlog', **fitted_points_MQ)
    
    # Plot residuals
    plt.plot(tau, MQ - fitted_points_MQ['Full_Fit_'], label='MQ Residual')
    plt.plot(tau, DQ - fitted_points_DQ['Full_Fit_'], label='DQ Residual')
    
    plt.legend(loc='upper right')
    
    # Save residuals plots
    plt.savefig(file+'_residuals.pdf', format="pdf", bbox_inches="tight")
    plt.savefig(file+'_residuals.png', format="png", bbox_inches="tight")
    plt.close()


def minimizer_result_to_dataframe(result, file):
    """
    Convert lmfit.MinimizerResult parameters to a pandas DataFrame.

    :result: lmfit.MinimizerResult object containing the fit results
    :file: String to be used as the sample name for all rows
    :return: pandas DataFrame with parameter details and a sample column
    """
    # Extract parameter details
    data = {
        'Sample': [],
        'Fraction': [],
        'Quantity': [],
        'Parameter': [],
        'Value': [],
        'Error': [],
        'Min': [],
        'Max': [],
        'Vary': []
    }

    for param_name, param in result.params.items():
        if 'diff' not in param_name:
            fraction, quantity = param_name.split('_', 1)
            data['Sample'].append(file)
            data['Fraction'].append(fraction)
            data['Quantity'].append(quantity)
            data['Parameter'].append(param_name)
            data['Value'].append(param.value)
            data['Error'].append(param.stderr)
            data['Min'].append(param.min)
            data['Max'].append(param.max)
            data['Vary'].append(param.vary)

    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df

#Write results to file and make a bakup for simultaneous fit
def files_report(df,file,fitted_points_DQ,fitted_points_MQ,sim_fitted):
    #dump final parameters to a file
    f = open(file+"_fit_parameters.json", "w")
    sim_fitted.params.dump(f)
    f.close()  
    
    
    #Write the fitted data points to file
    df_DQ = pd.DataFrame(fitted_points_DQ).add_suffix('DQ')
    df_MQ = pd.DataFrame(fitted_points_MQ).add_suffix('MQ')
    df_fit = df_DQ.copy().assign(**df_MQ)
    df_result = df.copy().assign(**df_fit)
    df_result = df_result.assign(Sample=file)
    df_result.to_csv(file+'_fit_value.csv',index=False)
    
    print(lm.fit_report(sim_fitted))
    
    #Write fit report to the file
    file1 = open(file+"_fit_report.txt", "w")
    print(lm.fit_report(sim_fitted),file=file1)
    file1.close()
    
    #Write fit report in dataframe fromat
    df_params = minimizer_result_to_dataframe(sim_fitted, file)
    df_params.to_csv(file+'_fit_report.csv',index=False)
    
    
    #Pickel the minimizer result object
    write_object(sim_fitted,file+'_minimized.pckl')
    
    now=datetime.now()
    path='./temp/'+ now.strftime('%Y%m%d%H%M%S')+'/'
    os.makedirs(path,exist_ok=True)
    
    for f in os.listdir(os.curdir):
        if f.startswith(file):
            shutil.copy2(f, path)
    
    return df_result

#Write results to file and make a bakup for InDQ fit
def files_report_InDQ(df,file,fitted_points_nDQ,fitted):
    #dump final parameters to a file
    f = open(file+"_fit_parameters.json", "w")
    fitted.params.dump(f)
    f.close()  
    
    
    #Write the fitted data points to file
    df_nDQ = pd.DataFrame(fitted_points_nDQ).add_suffix('nDQ')
    df_result = df.copy().assign(**df_nDQ)
    df_result = df_result.assign(Sample=file)
    df_result.to_csv(file+'_fit_value.csv',index=False)
    
    print(lm.fit_report(fitted))
    
    #Write fit report to the file
    file1 = open(file+"_fit_report.txt", "w")
    print(lm.fit_report(fitted),file=file1)
    file1.close()
    
    #Write fit report in dataframe fromat
    df_params = minimizer_result_to_dataframe(fitted, file)
    df_params.to_csv(file+'_fit_report.csv',index=False)
    
    
    #Pickel the minimizer result object
    write_object(fitted,file+'_minimized.pckl')
    
    now=datetime.now()
    path='./temp/'+ now.strftime('%Y%m%d%H%M%S')+'/'
    os.makedirs(path,exist_ok=True)
    
    for f in os.listdir(os.curdir):
        if f.startswith(file):
            shutil.copy2(f, path)
    
    return df_result


#Function for slicing a spectra
def spectra_slice(spectra, no_of_slices=1, start=None, stop=None, slice_points=None):
    if slice_points is None:
        slices=np.array_split(spectra[start:stop], no_of_slices)
    return slices

#Function for determining the area of a set of spectra slices
def slice_area(spectra_set, no_of_slices=1, start=None, stop=None, slice_points=None):
    Area =np.zeros((len(spectra_set),no_of_slices))    
    for idx,spectra in enumerate(spectra_set):
        slices=spectra_slice(spectra, no_of_slices, start, stop, slice_points)
        for x,y in enumerate(slices):
            Area[idx,x] =  np.trapz(y)
    return Area

def read_exp_parameters(filename='../exp_info.csv', index_col='EXP'):
    # Get the current folder name
    current_folder = os.path.basename(os.getcwd())
    # Read the CSV file, using the specified index column
    df = pd.read_csv(filename, index_col=index_col)
    df.index = df.index.map(str)
    # Check if the current folder matches any values in the index (EXPNO)
    if current_folder in df.index.astype(str):  # Convert index to string for comparison
        # Filter the DataFrame to return only the row(s) that match the current folder name
        
        matching_row = df.loc[current_folder]
        
        # Dictionary to store the column data as variables
        variables = {}
        
        # Loop through the columns and store values in the dictionary
        for column in df.columns:
            variables[column] = matching_row[column] if isinstance(matching_row, pd.Series) else matching_row[column].values[0]
        
        variables[index_col] = current_folder
        
        return variables
    else:
        return f"No matching EXPNO found for folder: {current_folder}"

def single_ci2d(i): #i is a tuple in the form (dictionary,tuple)
    fitter,params=i[0],i[1]
    x,y,grid = lm.conf_interval2d(fitter['minimizer'], fitter['result'], params[0], params[1],prob_func=chi)    

    return (params,x,y,grid)

def single_ci(i): #i is a tuple in the form (dictionary,parameter)
    fitter,params=i[0],i[1]
    print(params)
    ci,trace = lm.conf_interval(fitter['minimizer'], fitter['result'], trace=True,p_names=params,prob_func=chi,sigmas=fitter['sigmas'])    

    return (params[0],ci,trace)

##Update intial values and bounds to new parameter values
def update_bounds(parameter,value_dict,change=None,width=0.4):
    for key,values in value_dict.items():
        parameter[key].set(value=values,min=values*(1-width),max=values*(1+width))
    return parameter

#Calculate the ratio of chi-sqr of the new fit and the best fit
def chi_ratio(best_fit,new_fit):
    ratio=new_fit.chisqr/best_fit.chisqr
    return ratio

#Calculate the chi-sqr of the new fit (required for conf_interval2d to work)
def chi(best_fit,new_fit):
    return new_fit.chisqr

#Prepare ppm axis from 2d experiment
def make_ppm_axis2d(dic):
    ppm_axis= dict()
    for key,value  in dic.items():
        ppm_range= value['SW_p']/value['SF']
        ppm_ref = value['OFFSET']
        ppm=np.array([-1*(i+(1/2))/value['SI']*ppm_range for i in range(value['SI'])])+ppm_ref
        ppm_axis[key]=ppm
    return ppm_axis

#Prepare ppm axis from 1d experiment
def make_ppm_axis1d(dic):
    ppm_range= dic['procs']['SW_p']/dic['procs']['SF']
    ppm_ref = dic['procs']['OFFSET']
    ppm=np.array([-1*(i+(1/2))/dic['procs']['SI']*ppm_range for i in range(dic['procs']['SI'])])+ppm_ref
    return ppm
###################################################
# Creation of fitting models
###################################################

#Tail sustraction
def tail(df,tau_start=None,tail_model=None,params=None,vary_beta=False,space='diff'):
    #Ready the model   
    if tail_model==None: tail_model=lm.Model(fn.T2_decay)

    #Ready the parameters
    if params==None:
        params = lm.Parameters() 
        params.add('A', value=0.5 ,min=0.01 , max=0.9) #Fraction of tail
        params.add('T2', value=450, min=0) #T2 of tail
        params.add('beta', value=1,vary=vary_beta) #Stretching exponent

    a = tau_start
    
    finish = 1
    while finish == 1:
    #Ready the data 
        if a==None: tau_start = float(input('Enter the starting time for tail fitting'))
        df_tail = df[df['Time'] >= tau_start ].copy()
        tau = df_tail['Time']

    #Determine which data to fit in the tail
        if space=='ref':
                I = df_tail['I_ref']
        elif space== 'diff':
                I = df_tail['I_diff']
        elif space== 'sum':
                I = df_tail['I_MQ']
        
    ##Fit the data to the model
        tail_fit=tail_model.fit(I,params,tau=tau,method='leastsq')
        fitted= tail_fit.eval(tail_fit.params,tau=df['Time'])
        

    ##Print the fit report
        print(tail_fit.fit_report())

        df['I_MQ_no_tail'] = df['I_MQ']  - fitted
        df['I_nDQ'] = df['I_DQ'] / df['I_MQ_no_tail']
        df['Tail']= fitted

        if a!=None: return tail_fit
        
        plt.ylim(0.001, 1)    
        plotmq(df['Time'],df[f'I_{space}'],df['I_DQ'],df['I_nDQ'],y_axis='log', show=True,
               tail=df['Tail'],subtracted=df['I_MQ_no_tail'])
        
        finish = int(input('Press 1 if you want another cutoff'))    
        
   
    

    return tail_fit

def single_point(i): #i is a tuple in the form (dictionary,fixed value)
    x,y=i[0],i[1]    
    params = randomize_parameters(x['params'], x['range'])
    params[x['fixed']].set(value=y,vary=False)
    params['tail_beta'].set(value=1,vary=False)
    lowest=x['fit'].minimize(method='leastsq',params=params)
    for j in range(x['repeat']-1):
        params = randomize_parameters(x['params'], x['range'])
        params[x['fixed']].set(value=y,vary=False)
        params['tail_beta'].set(value=1,vary=False)
        temp =x['fit'].minimize(method='leastsq',params=params)
        if temp.chisqr < lowest.chisqr:
            lowest=temp
    print(y,datetime.now(),lowest.chisqr)
    return (y,lowest)

def single_point_no_randomize(i): #i is a tuple in the form (dictionary,fixed value)
    #Define some defaults
    x,y=i[0],i[1] 
    method= x.get('method','leastsq')
    params = x['params']
    params[x['fixed']].set(value=y,vary=False)
    lowest=x['fit'].minimize(method=method,params=params)
    return (y,lowest)

#Simultaneous fit
def fit_simultaneous(parameter,tau,tau_truncated,DQ,MQ,model_DQ,model_MQ,T2_penalty=None,Dres_penalty=None, scale=fn.nothing):
    params=parameter.valuesdict()
    
    residual_DQ = (scale(DQ) - scale(model_DQ(tau_truncated,params)))/scale(max(DQ))
    
    residual_MQ = (scale(MQ) - scale(model_MQ(tau,params)))/scale(max(MQ))
    
    residual = np.append(residual_DQ, residual_MQ)    

    return residual #Minimization parameter

#Simultaneous fit
def fit_single(parameter,model,x,y):
    
    params=parameter.valuesdict()
    
    residual = (y - model(x,params))  
    
    return residual #Minimization parameter

def diff_fit_single(i):
    n,area,fitter=i[0],i[1],i[2]
    result = fitter['model'].fit(area,fitter['params'],B=fitter['B'],method=fitter['method'])
    return (n,area,result,i[3])


#Fitting of diffusion curve with two moving and one fixed components
def diff_2_comp_1_const():
    
    diff_components=['first','second']
    
    first = lm.Model(fn.diffusion_decay,independent_vars=['B'],
                     prefix='first_',param_names=('D','A'))
    
    second= lm.Model(fn.diffusion_decay,independent_vars=['B'],
                     prefix='second_',param_names=('D','A'))
    
    fixed = lm.Model(fn.diffusion_decay,independent_vars=['B'],
                     prefix='fixed_',param_names=('D','A'))
    
    model= first+second+fixed
    
    params = lm.Parameters()
    
    
    params.add('first_D',1e-9,vary=True,min=1e-10,max=4e-9)
    params.add('first_A',0.3,vary=True,min=0.0)
    
    params.add('diff_D',0.1,vary=True,min=0,max=1)
    
    params.add('second_D',1e-11,vary=True,min=1e-13,max=4e-10,expr='first_D*diff_D')
    params.add('second_A',0.3,vary=True,min=0.0)
    
    params.add('fixed_D',0.0,vary=False,min=0,max=1e-10)
    params.add('fixed_A',0.3,vary=True,min=0.0)
    
    return model,params,diff_components

#Fitting of diffusion curve with one moving and one fixed components
def diff_1_comp_1_const():
    
    diff_components=['first']
    
    first = lm.Model(fn.diffusion_decay,independent_vars=['B'],
                     prefix='first_',param_names=('D','A'))
    
    fixed = lm.Model(fn.diffusion_decay,independent_vars=['B'],
                     prefix='fixed_',param_names=('D','A'))
    
    model= first+fixed
    
    params = lm.Parameters()
    
    
    params.add('first_D',1e-9,vary=True,min=1e-10,max=4e-9)
    params.add('first_A',0.3,vary=True,min=0.0)
    
    params.add('fixed_D',0.0,vary=False,min=0,max=1e-10)
    params.add('fixed_A',0.3,vary=True,min=0.0)
    
    return model,params,diff_components


#Fitting of diffusion curve with two moving components
def diff_2_comp():
    
    diff_components=['first','second']
    
    first = lm.Model(fn.diffusion_decay,independent_vars=['B'],
                     prefix='first_',param_names=('D','A'))
    
    second= lm.Model(fn.diffusion_decay,independent_vars=['B'],
                     prefix='second_',param_names=('D','A'))
    
    model= first+second
    
    params = lm.Parameters()
    
    
    params.add('first_D',1e-9,vary=True,min=1e-10,max=4e-9)
    params.add('first_A',0.3,vary=True,min=0.0)
    
    params.add('diff_D',0.1,vary=True,min=0,max=1)
    
    params.add('second_D',1e-11,vary=True,min=1e-13,max=4e-10,expr='first_D*diff_D')
    params.add('second_A',0.3,vary=True,min=0.0)
    
    return model,params,diff_components

#Fitting of diffusion curve with one moving component
def diff_1_comp():
    
    diff_components=['first']
    
    first = lm.Model(fn.diffusion_decay,independent_vars=['B'],
                     prefix='first_',param_names=('D','A'))

    
    model= first
    
    params = lm.Parameters()
    
    
    params.add('first_D',1e-9,vary=True,min=1e-10,max=4e-9)
    params.add('first_A',0.3,vary=True,min=0.0)
    
    
    return model,params,diff_components



#This section contains non standard codes
###################################################

#Extract T2 and Dres difference
def extract_diff(parameter,connectivity=None):
    diff_T2 = grab(parameter,"diff_T2")

    diff_Dres = grab(parameter,"diff_Dres")    
    return diff_T2,diff_Dres

#Extract and penalize T2 and Dres difference
def extract_and_penalize(parameter,T2_penalty,Dres_penalty):
    
    diff_T2, diff_Dres = extract_diff(parameter)
    a = 1 * overlap_penalty(diff_T2,**T2_penalty) * overlap_penalty(diff_Dres,**Dres_penalty)
    return a
