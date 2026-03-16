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


def FID_intensity(tau,parameters):  
        
    rigid_comp= fn.T2_decay(tau, T2=parameters['rigid_T2'],A=parameters['rigid_FIDA'],beta=2) * fn.sinc(parameters['rigid_Mb']*tau)
    
    interface_comp= fn.T2_decay(tau, T2=parameters['interface_T2'],A=parameters['interface_FIDA'],beta=parameters['interface_beta'])
    
    mobile_comp= fn.T2_decay(tau, T2=parameters['mobile_T2'],A=parameters['mobile_FIDA'],beta=parameters['mobile_beta'])
    
    intensity= rigid_comp + interface_comp + mobile_comp

    return intensity
    
def MSE_intensity(tau,parameters):  
        
    rigid_comp= fn.T2_decay(tau, T2=parameters['rigid_T2'],A=parameters['rigid_MSEA'],beta=2) * fn.sinc(parameters['rigid_Mb']*tau)
    
    interface_comp= fn.T2_decay(tau, T2=parameters['interface_T2'],A=parameters['interface_MSEA'],beta=parameters['interface_beta'])
    
    mobile_comp= fn.T2_decay(tau, T2=parameters['mobile_T2'],A=parameters['mobile_MSEA'],beta=parameters['mobile_beta'])
    
    intensity= rigid_comp + interface_comp + mobile_comp

    return intensity


#############################################

a=datetime.now()
print(a)
##############################################
#Variable Declarations
##############################################

#exp_info = oth.read_exp_parameters()



fractions=['rigid','interface','mobile'] #Number of components



##############################################
#Cleaning of Data

####Get the foldername and the time axis for the kinetics data

def timedata_FIDMSE(filename="FIDMSE.txt",sample=None):
    if sample == None:
        sample= os.getcwd()
        print(sample)
        sample = sample.replace('\\','/')
        sample= sample.split('/')[-1]
    
    df = pd.read_csv(f'{filename}_time', sep='\t',header=None,names=['EXP_CNT','Time'])
    
    #delete old files from directory
    for f in os.listdir(os.curdir):
        if f.startswith(sample):
            os.remove(f)
    
    return df,sample

def rawdata_FIDMSE(filename="FIDMSE.txt"):
    
    df = pd.read_csv(filename, sep='\t',header=None,names=['FID_time','I_FID_real','I_FID_imag','MSE_time','I_MSE_real','I_MSE_imag'])
    
    return df


###########################################

#Import the data

filename="kin-323.txt" #Experiment filename
df,file = timedata_FIDMSE(filename) #Get time data from _time file


for idx,a in df['Time'].items():
    if a <0:
        df['Time'][idx]=df['Time'][idx] + 2641338 + 36723 + 338


#Get the data for each time point and the maximum value of the whole FID
raw_dict = {} 
max_intensity =1
for a in df['EXP_CNT']:
    data= rawdata_FIDMSE(f'{filename}_fidmse{a}')
    raw_dict[a] = data
        
    #Find the highest point in the whole FID set
    intensity =  data.drop(['FID_time', 'MSE_time'], axis=1).max().max()
    
    if intensity > max_intensity: max_intensity = intensity


for index,FID in raw_dict.items(): 
    for intensity in list(FID.columns):
        if intensity not in ['FID_time', 'MSE_time']:
            raw_dict[index][intensity]=FID[intensity].div(max_intensity)





###############################################
#Define ranges for parameters of each components
###############################################

ranges = {
    'rigid_T2': (0.001, 0.05), 'rigid_MSEA': (0.001, 1.1), 'rigid_FIDA': (0.001, 1.1),
    'interface_T2': (0.01, 0.1),'interface_MSEA': (0.001,1.1),  'interface_FIDA': (0.001,1.1), 'interface_beta': (0.8, 2),
    'mobile_T2': (0.1, 1),'mobile_MSEA': (0.001,1.1),  'mobile_FIDA': (0.001,1.1), 'mobile_beta': (0.8, 2)
}


###############################################
#Initialize parameters for each components
###############################################

FID_params = lm.Parameters()


# Add parameters to DQ_params
for param, (min_val, max_val) in ranges.items():
    initial_value = (min_val + max_val) / 2  # Default initial value is the midpoint of the range
    FID_params.add(param, value=initial_value, vary=True, min=min_val, max=max_val)

# Randomize the parameters and get the updated parameters
FID_params = oth.randomize_parameters(FID_params, ranges)

FID_params['interface_beta'].set(value=1)
FID_params['mobile_beta'].set(value=1)

###############################################
#Define the range of T2
###############################################

# Define the difference of T2 values as parameters
FID_params.add('diff_T2_1',value=0.01,min=0,max=0.5)
FID_params['rigid_T2'].set(expr='interface_T2 * diff_T2_1')


FID_params.add('diff_T2_2',value=0.01,min=0,max=0.5)
FID_params['interface_T2'].set(expr='mobile_T2 * diff_T2_2')


FID_params.add('diff_A_1',value=0.9,min=0,max=1)
FID_params['rigid_MSEA'].set(expr='rigid_FIDA * diff_A_1')

FID_params.add('diff_A_2',value=0.9,min=0,max=1)
FID_params['interface_MSEA'].set(expr='interface_FIDA * diff_A_2')

FID_params.add('diff_A_3',value=0.9,min=0,max=1)
FID_params['mobile_MSEA'].set(expr='mobile_FIDA * diff_A_3')

###############################################
#Define the moment parameters
###############################################
#FID_params.add('rigid_Ma',min=0,expr='sqrt(2)/rigid_T2')
FID_params.add('rigid_Mb',min=0,value = 0,vary=False)

##############################################
#Define DQ and MQ models and other variables
##############################################

FID_model=FID_intensity

 
MSE_model=MSE_intensity





#Simultaneous fit
def fit_MSEFID_simultaneous(parameter,time_FID,FID,time_MSE,MSE,model_FID,model_MSE):
    params=parameter.valuesdict()
    
    residual_FID = (FID - model_FID(time_FID,params))
    
    residual_MSE = (MSE - model_MSE(time_MSE,params))
    
    residual = np.append(residual_FID, residual_MSE)    

    return residual #Minimization parameter


#Simultaneous fitting of FID and MSE curve
def random_leastsq_FIDMSE(i):
    x,y,df=i[0],i[1],i[2] 
    method= x.get('method','leastsq')
    attempts = x.get('attempts',5)

    params = x['params']


    sim_fit=lm.Minimizer(
    fit_MSEFID_simultaneous,FID_params,
    fcn_args=(df['FID_time'],df['I_FID_real'],df['MSE_time'],df['I_MSE_real'],FID_model,MSE_model))
    
    
    best_result = sim_fit.minimize(method=method,params=params)

    for a in range(attempts):

        params = oth.randomize_parameters(params, x['ranges'])

        trial = sim_fit.minimize(method='leastsq', params=params)
        if trial.chisqr < best_result.chisqr:
            best_result = trial

    print(f"FIT {y} completed.")    
    
    return (y,best_result)

def single_fit_FIDMSE(i): #i is a tuple in the form (dictionary,time index, data)
    #Define some defaults
    x,y,df=i[0],i[1],i[2] 
    method= x.get('method','leastsq')
    params = x['params']
    
    sim_fit=lm.Minimizer(
    fit_MSEFID_simultaneous,FID_params,
    fcn_args=(df['FID_time'],df['I_FID_real'],df['MSE_time'],df['I_MSE_real'],FID_model,MSE_model))
    
    fitted=sim_fit.minimize(method=method,params=params)
    
    print(f"FIT {y} completed.")    
    
    return (y,fitted)
    
# temp_params=FID_params

# for x in ['rigid','interface']:    
#     for y in ['FIDA','MSEA']:
#         temp_params[f'{x}_{y}'].set(value=0,vary=False)
        
# temp_params['mobile_beta'].set(value=1,vary=False)

# fitter={
#     'params':FID_params,
#     'method':'leastsq'
#     }

# i=(fitter,len(raw_dict),raw_dict[len(raw_dict)])



# initial_fit= single_fit_FIDMSE(i)

# FID_params= initial_fit[1].params

# plt.scatter(i[2]['FID_time'],i[2]['I_FID_real'], facecolors='none', edgecolors='b')
# plt.plot(i[2]['FID_time'],FID_intensity(i[2]['FID_time'],FID_params.valuesdict()))

# plt.scatter(i[2]['MSE_time'],i[2]['I_MSE_real'], facecolors='none', edgecolors='r')
# plt.plot(i[2]['MSE_time'],MSE_intensity(i[2]['MSE_time'],FID_params.valuesdict()))

# plt.show()


# FID_params.pretty_print()

# print(lm.fit_report(initial_fit[1]))

# search_result =[]

# for a,data in reversed(raw_dict.items()):
#     fitter={
#         'params':FID_params,
#         'method':'basinhopping'
#         }
    
    
#     i= (fitter,a,data) 
       
    
    
#     fit=single_fit_FIDMSE(i)
#     search_result.append(fit)
    
#     FID_params = fit[1].params
# search_result.reverse()    


fitter={
        'params':FID_params,
        'method':'basinhopping',
        'ranges':ranges
        }
space= [(fitter,a,data) for a,data in raw_dict.items()]
pool= Pool()
search_result=pool.map(single_fit_FIDMSE,space)

#search_result=pool.map(random_leastsq_FIDMSE,space)




oth.write_object(search_result,'last_fit_result.pckl') 

oth.test()
search_result=oth.load_object('last_fit_result.pckl')


result_param = [fit[1].params.valuesdict() for fit in search_result]
result_param_idx=[(fit[0],fit[1].params.valuesdict()) for fit in search_result]

result_df = pd.DataFrame(result_param)

time_list = [df['Time'][a-1] for a,b in result_param_idx]


for a in list(result_df.columns):
    plt.scatter(time_list,result_df[a])
    plt.xlabel("Time")
    plt.ylabel(a)
    plt.xscale('log')
    plt.show()



for (idx,result) in result_param_idx:

    plt.scatter(raw_dict[idx]['FID_time'],raw_dict[idx]['I_FID_real'], facecolors='none', edgecolors='r')
    plt.plot(raw_dict[idx]['FID_time'],FID_intensity(raw_dict[idx]['FID_time'],result))
    
    plt.scatter(raw_dict[idx]['MSE_time'],raw_dict[idx]['I_MSE_real'], facecolors='none', edgecolors='b')
    plt.plot(raw_dict[idx]['MSE_time'],MSE_intensity(raw_dict[idx]['MSE_time'],result))

    plt.show()




# #=============================================================================
# #Initial fit for parameter initialization
# #=============================================================================
read= False

if read is True:
    FID_params= oth.load_object('last_fit_result.pckl')
    sim_fitted= sim_fit.minimize(method='leastsq',params=FID_params)
else:
    sim_fitted= sim_fit.minimize(method='leastsq',params=FID_params)



  
temp=sim_fitted.params.valuesdict()

change=[]
for x in fractions:
    for y in ["_T2"]:
        change.append(f'{x}{y}')

update_list={key: temp[key] for key in change}
   
FID_params=oth.update_bounds(FID_params, update_list)



print(sim_fitted.params.pretty_print())



# #=============================================================================
# #Draw a chi-sqr map
# #=============================================================================

fit_chi_sqr=sim_fitted.chisqr
chi_sqr_map=False
map_parameters= ['rigid_A','interface_A','mobile_A']

if chi_sqr_map is True:
    
    for parameter in map_parameters:
        if sim_fitted.params[parameter].stderr is None:
            sim_fitted.params[parameter].stderr = abs(sim_fitted.params[parameter].value * 0.1)
        lower = max(sim_fitted.params[parameter] - 5*sim_fitted.params[parameter].stderr,sim_fitted.params[parameter].min)
        higher = min(sim_fitted.params[parameter] + 5*sim_fitted.params[parameter].stderr,sim_fitted.params[parameter].max)

        A=np.linspace(lower, higher,num=80)
        fitter={
            'params':FID_params,
            'fixed':parameter,
            'fit':sim_fit,
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




components_FID = {}
for x in fractions:
    components_FID[x] = fn.DQ_1_Dres_1_T2(fittau, result[f'{x}_Dres'], result[f'{x}_T2'], 
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
#Calculate confidence interval
##############################################

calculate_ci=False
calculate_ci2d=False


ci_params=['first_A','second_A','third_A']
ci2d_pairs=[]
for i in range(len(ci_params)):
    for j in range(i+1,len(ci_params)):
        ci2d_pairs.append((ci_params[i],ci_params[j]))


if calculate_ci is True:
    
    for p in sim_fitted.params:
        if sim_fitted.params[p].stderr is None:
            sim_fitted.params[p].stderr = abs(sim_fitted.params[p].value * 0.1)
        if p in ci_params:
            sim_fitted.params[p].min = sim_fitted.params[p].value - 5 * sim_fitted.params[p].stderr
            lower = sim_fitted.params[p].value + 5 * sim_fitted.params[p].stderr
            if lower<=0:
                lower = 0.00000001
            sim_fitted.params[p].max = lower 

    try:
        ci_dict={
        'minimizer': sim_fit,
        'result': sim_fitted,
        'sigmas':[0.5,1,1.5,2,2.5,3]
        }
        ci_space= [(ci_dict,[k]) for k in ci_params]
        pool= Pool()
        ci_result=pool.map(oth.single_ci,ci_space)

    except Exception as error:
        print("One dimensional CI not calculated",error)
    else:
        #Pickel the confidence interval object
        oth.write_object(ci_result,file+'_ci.pckl')

        num=len(ci_result)
        y=int(num**0.5)
        x=int(num/y)
        while num % x !=0:
            y=y-1
            x=int(num/y)

        if y ==1:
            y=int(num**0.5)
            x=y+1
            while x*y<num:
                x = x+1

        fig,ax=plt.subplots(x,y)

        for idx,result in enumerate(ci_result):
            i,j=idx//y,idx % y
            parameter,ci,trace=result[0],result[1],result[2]
            # Plot chi-sqr
            fixed, vary, prob = trace[parameter][parameter], trace[parameter]['second_A'], trace[parameter]['prob']
            prob=prob/sim_fitted.chisqr
            ax[i,j].scatter(fixed,prob)
            ax[i,j].axhline(y=3)
            ax[i,j].set_xlabel(parameter,horizontalalignment='left')
            ax[i,j].xaxis.set_label_coords(0.1,0.1)
            ax[i,j].set_ylabel('rel_chisqr',verticalalignment='bottom')
            ax[i,j].yaxis.set_label_coords(0.1,0.35)
        fig.suptitle(f'Best fit χ-sq:{sim_fitted.chisqr}')
        fig.set_size_inches(3*(x),4*(y))
        plt.savefig(f'{file}_ci.pdf', format="pdf", bbox_inches="tight")
        plt.savefig(f'{file}_ci.png', format="png", bbox_inches="tight")
        plt.close()


if calculate_ci2d is True:

    try:
        ci2d_dict={
            'minimizer': sim_fit,
            'result': sim_fitted
        }
        ci2d_space= [(ci2d_dict,k) for k in ci2d_pairs]
        pool= Pool()
        ci2d_result=pool.map(oth.single_ci2d,ci2d_space)

    except Exception as error:
        print("Two dimensional CI not calculated",error)

    else:
        #Pickel the confidence interval 2d object
        oth.write_object(ci2d_result,file+'_ci2d.pckl')

        num=len(ci2d_result)
        y=int(num**0.5)
        x=int(num/y)
        while num % x !=0:
            y=y-1
            x=int(num/y)

        if y ==1:
            y=int(num**0.5)
            x=y+1
            while (x*y)<num:
                x = x+1

        fig,ax=plt.subplots(x,y)

        for idx,result in enumerate(ci2d_result):
            i,j=idx//y,idx % y
            pair,x_value,y_value,grid=result[0],result[1],result[2],result[3]
            # Plot chi-sqr
            cnt=ax[i,j].contour(x_value,y_value,grid,levels=[1.1, 1.3, 1.5, 1.7, 1.9, 2.1])
            ax[i,j].set_xlabel(pair[0],horizontalalignment='left')
            ax[i,j].xaxis.set_label_coords(0.1,0.1)
            ax[i,j].set_ylabel(pair[1],verticalalignment='bottom')
            ax[i,j].yaxis.set_label_coords(0.1,0.35)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.82, 0.15, 0.015, 0.7])
        cbar_ax.annotate(f'χsq:{sim_fitted.chisqr:.3e}',(0,-0.1),xycoords='axes fraction')
        fig.colorbar(cnt, cax=cbar_ax)
        fig.set_size_inches(3*(x+1),4*(y))
        plt.savefig(f'{file}_ci2d.pdf', format="pdf", bbox_inches="tight")
        plt.savefig(f'{file}_ci2d.png', format="png", bbox_inches="tight")
        plt.close()

##############################################
#Create a result dataframe and write outputs to file
##############################################

df_result= oth.files_report(df,file,fitted_points_DQ,fitted_points_MQ,sim_fitted)

b=datetime.now()
print(b)

print(f'Execution time is {b-a}')


