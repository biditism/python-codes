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

#Get the foldername and the file with Baum-Pines data
def high_field_FID_kinetics_rawdata(filename="clean_fid",sample=None,fidlist=[0],cutoff=200,omit=None):
    if sample == None:
        sample= os.getcwd()
        print(sample)
        sample = sample.replace('\\','/')
        sample= sample.split('/')[-1]
    df_dict={}

    for a in fidlist:
        df= pd.read_csv(f'{filename}_{a}.txt', sep=',',header=None,names=['Time','I_real','I_imag','Magnitude','Time_point'])
        
        if omit is not None:
            #print("No of experimental points:",df.shape[0])
            for x in omit:
                df = df.drop(x-1)
            #print("No of experimental points after artefact removal:",df.shape[0])


        df = df[df['Time'] <= cutoff]


        df_dict[a] = df
    
    return df_dict,sample

def high_field_FID_rawdata(filename="clean_fid.txt",sample=None):
    if sample == None:
        sample= os.getcwd()
        print(sample)
        sample = sample.replace('\\','/')
        sample= sample.split('/')[-1]
    df = pd.read_csv(filename, sep=',',header=None,names=['Time','I_real','I_imag','Magnitude'])
    
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


#Normalization to the first point of the data and cutoff extra data
def high_field_FID_clean(df,cutoff=100,omit=None,norm_factor=None):
    
    #Remove points with artefacts
    if omit is not None:
        print("No of experimental points:",df.shape[0])
        for x in omit:
            df = df.drop(x-1)
        print("No of experimental points after artefact removal:",df.shape[0])
    
    
    if norm_factor==None: norm_factor = max(df['I_real'])

    df['I_real'] = df['I_real'] / norm_factor
    df['I_imag'] = df['I_imag'] / norm_factor
    df['Magnitude'] = df['Magnitude'] / norm_factor
    
    
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

        #Scatter plot of multiple files
def plotFID(tau,*args,y_axis='linear',save=None,show=False,**kwargs):
    for I in args:
        plt.scatter(tau,I,alpha=0.5)
    for key,values in kwargs.items():
        plt.plot(tau,values,label=key)
        plt.legend(loc='upper right')
    plt.yscale(y_axis)
    #xmin, xmax, ymin, ymax = plt.axis()
    #plt.ylim(bottom=0.001,top=min(ymax,1.1))
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


def plot_results_FID(tau, FID, fitted_points_FID, file):
    """
    Plot the results of the fitted models.

    :tau: Array of tau values
    :FID: Array of FID data points
    :fitted_points_FID: Dictionary of fitted points for FID model
    :param file: Filename prefix for saving the plots
    """

    #FID_dict={'FID':FID}
    # Plot FID data with linear-linear scale
    plt.xscale('linear')
    #plotFID(tau, y_axis='linear', save=file+'_FID_fit', **FID_dict, **fitted_points_FID)
    plotFID(tau, FID, y_axis='linear', save=file+'_FID_fit',  **fitted_points_FID)

    # Plot residuals
    plt.plot(tau, FID - fitted_points_FID['Full_Fit_'], label='FID Residual')
    
    plt.legend(loc='upper right')
    
    # Save residuals plots
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


#Write results to file and make a bakup for simultaneous fit
def files_report_FID(df,file,fitted_points_FID,FID_fitted,write=True):
    #dump final parameters to a file
    
    if write:
        f = open(file+"_fit_parameters.json", "w")
        FID_fitted.params.dump(f)
        f.close()  
    
    
    #Write the fitted data points to file
    df_FID = pd.DataFrame(fitted_points_FID).add_suffix('FID')
    df_result = df.copy().assign(**df_FID)
    df_result = df_result.assign(Sample=file)
    
    if write:
        df_result.to_csv(file+'_fit_value.csv',index=False)
    
    #print(lm.fit_report(FID_fitted))
    if write:

        #Write fit report to the file
        file1 = open(file+"_fit_report.txt", "w")
        print(lm.fit_report(FID_fitted),file=file1)
        file1.close()
    
    #Write fit report in dataframe fromat
    df_params = minimizer_result_to_dataframe(FID_fitted, file)
    if write:
        df_params.to_csv(file+'_fit_report.csv',index=False)
    
    
        #Pickel the minimizer result object
        write_object(FID_fitted,file+'_minimized.pckl')
    
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
        params.add('A', value=0.1 ,min=0.01 , max=0.9) #Fraction of tail
        params.add('T2', value=450, min=0) #T2 of tail
        params.add('beta', value=1,vary=vary_beta,min=0.8,max=2) #Stretching exponent

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

def single_point_no_randomize_2fixed(i): #i is a tuple in the form (dictionary,fixed value1,fixed value2)
    #Define some defaults
    x,y,z=i[0],i[1],i[2] 
    method= x.get('method','leastsq')
    params = x['params']
    params[x['fixed']].set(value=y,vary=False)
    params[x['fixed2']].set(value=z,vary=False)
    lowest=x['fit'].minimize(method=method,params=params)
    return (y,z,lowest)

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

# Define the fitting functio for T2 decay based on the number of components
def T2_model(components):
    a=['first','second','third','fourth']
    fractions=a[:components]

    #Make the first T2 component
    model=lm.Model(fn.T2_decay,independent_vars=['tau'],
                     prefix=f'{fractions[0]}_',param_names=('T2','A','beta'))   
    
    # Add each T2 component
    if components>1:
        for a in fractions[1:]:
            model= model +  lm.Model(fn.T2_decay,independent_vars=['tau'],
                     prefix=f'{a}_',param_names=('T2','A','beta'))
   
    params=model.make_params()

    
    #Set the bounds of parameters
    for idx,a in enumerate(fractions):
        params[f'{a}_A'].set(min=0) #make A positive
        params[f'{a}_T2'].set(min=0.0000001) #make T2 positive
        params[f'{a}_beta'].set(value=1,vary=False,min=0.8,max=2) #make a non-exponential T2 decay


    #Define the diff variables:
    for value in range(components):
        if value>0:
            params.add(f'diff_T2_{value}',0.1,vary=True,min=0,max=1)

    # Set each T2 longer than the other
    if components>1:
        params['first_T2'].set(expr='second_T2 * diff_T2_1')
    if components>2:
        params['second_T2'].set(expr='third_T2 * diff_T2_2')
    if components>3:
        params['third_T2'].set(expr='fourth_T2 * diff_T2_3')

    return model,params,fractions

#Fit T2 to the data
def T2_fit_single(i):
    n,area,fitter=i[0],i[1],i[2]
    result = fitter['model'].fit(area,fitter['params'],tau=fitter['tau'],method=fitter['method'])
    return (n,area,result,i[3])


def phasecorr_time_domain(data, notebook=False):
    """

    Changed from nmrglue to show both real and imaginary part.

    Manual Phase correction using matplotlib

    A matplotlib widget is used to manually correct the phase of a Fourier
    transformed dataset. If the dataset has more than 1 dimensions, the first
    trace will be picked up for phase correction.  Clicking the 'Set Phase'
    button will print the current linear phase parameters to the console.
    A ipywidget is provided for use with Jupyter Notebook to avoid changing
    backends. This can be accessed with notebook=True option in this function

    .. note:: Needs matplotlib with an interactive backend.

    Parameters
    ----------
    data : ndarray
        Array of NMR data.
    notebook : Bool
        True for plotting interactively in Jupyter Notebook
        Uses ipywidgets instead of matplotlib widgets

    Returns
    -------
    p0, p1 : float
        Linear phase correction parameters. Zero and first order phase
        corrections in degrees calculated from pc0, pc1 and pivot displayed
        in the interactive window.

    Examples
    --------
    >>> import nmrglue as ng
    >>> p0, p1 = ng.process.proc_autophase.manual_ps(data)
    >>> # do manual phase correction and close window
    >>> phased_data = ng.proc_base.ps(data, p0=p0, p1=p1)

    If you are using a Jupyter Notebook::

        In  [1] ng.process.proc_autophase.manual_ps(data)
        Out [1] # do manual phase correction. p0 and p1 values will be updated
                # continuously as you do so and are printed below the plot
        In  [2] phased_data = ng.proc_base.ps(data, p0=p0, p1=p1)

    """

    if len(data.shape) == 2:
        data = data[0, ...]
    elif len(data.shape) == 3:
        data = data[0, 0, ...]
    elif len(data.shape) == 4:
        data = data[0, 0, 0, ...]

    if notebook:
        from ipywidgets import interact, fixed

        def phasecorr(dataset, phcorr0, phcorr1, pivot):
            fig, ax = plt.subplots(figsize=(10, 7))
            phaseddata = dataset * np.exp(
                1j * (phcorr0 + phcorr1 * (
                    np.arange(-pivot, -pivot+dataset.size)/dataset.size)))

            ax.plot(np.real(phaseddata), lw=1, color='black')
            ax.plot(np.imag(phaseddata), lw=1, color='red')
            ax.set(ylim=(np.min(np.real(data))*2, np.max(np.real(data))*2))
            ax.axvline(pivot, color='r', alpha=0.5)
            plt.show()

            p0 = np.round(
                (phcorr0 - phcorr1 * pivot/dataset.size) * 360 / 2 / np.pi, 3)
            p1 = np.round(phcorr1*360/2/np.pi, 3)

            print('p0 =', p0, 'p1 =', p1)

        interact(
            phasecorr,
            dataset=fixed(data),
            phcorr0=(-np.pi, np.pi, 0.01),
            phcorr1=(-10*np.pi, 10*np.pi, 0.01),
            pivot=(0, data.size, 1))

    else:

        from matplotlib.widgets import Slider, Button

        # --- figure/axes ---
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.35)

        # plot BOTH real and imaginary parts
        (line_re,) = ax.plot(data.real, lw=1, color="black", label="Re")
        (line_im,) = ax.plot(data.imag, lw=1, color="tab:red", label="Im")

        ax.legend(loc="upper right")
        ax.set_xlabel("index")
        ax.set_ylabel("value")
        ax.grid()

        # --- widgets ---
        axcolor = "white"
        axpc0 = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
        axpc1 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
        axpiv = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)
        axpst = plt.axes([0.25, 0.25, 0.15, 0.04], facecolor=axcolor)

        spc0 = Slider(axpc0, "p0", -360, 360, valinit=0)
        spc1 = Slider(axpc1, "p1", -360, 360, valinit=0)
        spiv = Slider(axpiv, "pivot", 0, data.size, valinit=0)
        axps = Button(axpst, "Set Phase", color=axcolor)

        def update(val):
            pc0 = spc0.val * np.pi / 180
            pc1 = spc1.val * np.pi / 180
            pivot = spiv.val

            phase_ramp = pc0 + (pc1 * np.arange(-pivot, -pivot + data.size) / data.size)
            rotated = (data * np.exp(1.0j * phase_ramp)).astype(data.dtype)

            # update BOTH traces
            line_re.set_ydata(rotated.real)
            line_im.set_ydata(rotated.imag)

            fig.canvas.draw_idle()

        def setphase(event):
            p0 = spc0.val - spc1.val * spiv.val / data.size
            p1 = spc1.val
            print(p0, p1)

        spc0.on_changed(update)
        spc1.on_changed(update)
        spiv.on_changed(update)
        axps.on_clicked(setphase)

        plt.show(block=True)

        p0 = spc0.val - spc1.val * spiv.val / data.size
        p1 = spc1.val
        return p0, p1

def semilog_series(length, k):
    series = np.zeros(length)
    for n in range(1, length):
        series[n] = 2 ** ((n-1)  // k)
    return series

def bl_FID_kin_time_axis(d5,TD,l5=1000):
    """
    The time axis for kinetics measurement in high field using the
    pulse program bl_FID_kin
    d5= time between two experiments (including the experiment time)
    l5= double d5 after these number of experiments (integer)
    TD= total number of experiments (integer)
    """
    N=semilog_series(TD,l5)
    delay=d5*N
    time_point=np.cumsum(delay)
    return time_point

def second_to_hrminsec(exp_time):
    exp_time_hr=exp_time // 3600
    s_remaining=(exp_time-exp_time_hr*3600)
    exp_time_min=s_remaining//60
    s_remaining=s_remaining-exp_time_min*60
    #print(f"Total experiment time is {exp_time} second or {exp_time_hr} hour, {exp_time_min} minute and {s_remaining} second ")
    hrminsec=f"{exp_time} second or {exp_time_hr} hour, {exp_time_min} minute and {s_remaining} second "
    return hrminsec

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
