###################################################
# This is my personal collection of functions
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


###################################################
# Declaration of functions
###################################################


#Smooth Heaviside funtion
def heaviside(x, amplitude=1, steepness=1, base=1, center=0):
    return base + amplitude * (0.5 + 0.5 * np.tanh((x-center) * steepness))


#Overlap penalty for values
def overlap_penalty(diff, amount=0, width=1):
    penalty = 1
    for x in diff:
        penalty = penalty * heaviside(x, amount, 1/width)
    return penalty

#Abragam like function with a single Dres
def abragam(tau,Dres,A=0.5): 
    return A*(1- np.exp(-(0.378*2*np.pi*Dres*tau)**1.5)*np.cos(0.583*2*np.pi*Dres*tau))

#Abragam like function with a Dres distribution
def abragam_dist(tau_input, Dres, prob, A=0.5):
    tau = np.array(tau_input)
    y = (1 - np.exp(-(0.378 * 2 * np.pi * Dres * tau[:, np.newaxis]) ** 1.5) *
         np.cos(0.583 * 2 * np.pi * Dres * tau[:, np.newaxis])) * prob

    dq = A * np.trapz(y, Dres, axis=1)

    return dq

#Transverse relaxation
def T2_decay(tau,T2,A=1,beta=1): 
    return A*np.exp(-(tau/T2)**beta)

#DQ model curve = abragam multiplied by T2 decay
def single_Dres_dq(tau,Dres,T2,A,beta):
    return abragam(tau,Dres)*T2_decay(tau,T2,A,beta)

#Normal distribution
def gaussian(x,mu=0,sigma=1,normalize=True):
    sigma = 0.1* mu * sigma
    prob = (sigma*np.sqrt(2*np.pi))**(-1) * np.exp((-1/2)*(((x-mu)/sigma)**2))
    if normalize==True: prob = prob / np.trapz(prob,x)
    return prob

#Lognormal distribution
def lognormal(x,mu=0,sigma=1,normalize=True):
    mu = np.log(mu)
    prob = (x*sigma*np.sqrt(2*np.pi))**(-1) * np.exp((-1/2)*(((np.log(x)-mu)/sigma)**2))
    if normalize==True: prob = prob / np.trapz(prob,x)
    return prob

#Distribution with normal distribution of log(x) 
def lognormal2(x,mu=1,sigma=1,normalize=True):
    prob = gaussian(np.log(x),np.log(mu),sigma,normalize)
    if normalize==True: prob = prob / np.trapz(prob,x)
    return prob

#Get the list of all folder in a location which has a file with fixed ending
def file_list(ending,root_folder=None):
    file=[]
    if root_folder is None:
        root_folder=os.getcwd()
    for root,dirs, files in os.walk(root_folder):
        for each in files:
            if each.endswith(ending):
                file.append(each[:-len(ending)])
    return file


#Get the foldername and the file with Baum-Pines data
def rawdata(filename="BP_303.txt",sample=None):
    if sample == None: sample= os.getcwd().split('\\')[-1]
    df = pd.read_csv(filename, sep='\t',header=None,names=['Time','I_ref','I_DQ','Im'])
    return df,sample

#Normalization to the first point of the data and cutoff extra data
def clean(df,cutoff=None,norm_factor=None):
    if norm_factor==None: norm_factor = df['I_DQ'][0] + df['I_ref'][0]
    df['I_DQ'] = df['I_DQ'] / norm_factor
    df['I_ref'] = df['I_ref'] / norm_factor
    df['I_tot'] = df['I_DQ'] + df['I_ref']

    if cutoff==None:
        plotmq(df['Time'],df['I_DQ'],df['I_tot'])
    
    if cutoff==None:
        finish=1
        while finish ==1:
            cutoff = float(input('Enter the time cutoff for further calculations'))
            df_check = df[df['Time'] <= cutoff].copy()
            plotmq(df_check['Time'],df_check['I_DQ'],df_check['I_tot'])
            finish = int(input('Press 1 if you want another cutoff'))

    df_new = df[df['Time'] <= cutoff].copy()

    if cutoff==None:
        plotmq(df_new['Time'],df_new['I_DQ'],df_new['I_tot'])
    return df_new

#Scatter plot of multiple files
def plotmq(tau,*args,y_axis='linear',save=None,show=True,**kwargs):
    for I in args:
        plt.scatter(tau,I)
    for key,values in kwargs.items():
        plt.plot(tau,values,label=key)
        plt.legend(loc='upper right')
    plt.yscale(y_axis)
    xmin, xmax, ymin, ymax = plt.axis()
    plt.ylim(bottom=0.001,top=min(ymax,1))
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
def grab(parameter,fraction,string,extra=False,forward=True):
    dictionary=parameter.valuesdict()
    connect=fraction.copy()
    if extra != False: connect.append(extra)
    if forward != True: connect.reverse()
    new=[]
    for a in connect:
        new.append(dictionary[a+string])
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


###################################################
# Creation of fitting models
###################################################

#Tail sustraction
def tail(df,tau_start=None,tail_model=None,params=None,vary_beta=False):
    #Ready the model   
    if tail_model==None: tail_model=lm.Model(T2_decay)

    #Ready the parameters
    if params==None:
        params = lm.Parameters() 
        params.add('A', value=0.1,min=0.01,max=0.9) #Fraction of tail
        params.add('T2', value=max(df['Time'])/3,min=5 ,max=500) #T2 of tail
        params.add('beta', value=1,vary=vary_beta) #Stretching exponent

    a = tau_start
    
    finish = 1
    while finish == 1:
    #Ready the data 
        if a==None: tau_start = float(input('Enter the starting time for tail fitting'))
        df_tail = df[df['Time'] >= tau_start ].copy()
        tau = df_tail['Time']
        I_tot = df_tail['I_tot']

        
    ##Fit the data to the model
        tail_fit=tail_model.fit(I_tot,params,tau=tau,method='basinhopping')
        fitted= tail_fit.eval(tail_fit.params,tau=df['Time'])
        print(tail_fit.fit_report())

        values=tail_fit.best_values
        
        df['I_SumMQ'] = df['I_DQ'] + df['I_ref'] - T2_decay(df['Time'],values['A'],values['T2'],values['beta']) 
        df['I_nDQ'] = df['I_DQ'] / df['I_SumMQ']

   
        #plotmq(df['Time'],df['I_tot'],df['I_DQ'],y_axis='log',line=fitted)

        
        plotmq(df['Time'],df['I_tot'],df['I_DQ'],df['I_nDQ'],y_axis='log',line=fitted)

    ##Plot the result and graph

        
        
        if a!=None: return tail_fit
        
        finish = int(input('Press 1 if you want another cutoff'))    

    return tail_fit

def components(comp=None,distribution=False):
    if comp==None:
        if distribution==True:
            connectivity=["Ideal_","Defect_"]
        else:
            connectivity=["Single_","Double_","Higher_"]
    if isinstance(comp,list):
        connectivity=comp
    if isinstance(comp,int):        
        connectivity=[input("Prefix for each component") for a in range(comp)]
    return connectivity
        
##Switch between single Dres and distribution
def model_dq(comp=None,distribution=False):
    if distribution==False:
        dq = model_dq_no_distribution(comp)
    else:
        dq = model_dq_distribution(comp)
    return dq

##Model building for dq curve with distribution 
def model_dq_no_distribution(connectivity):
         
    temp = [lm.Model(single_Dres_dq,prefix=fraction) for fraction in connectivity]
    for a in temp: 
        try:
            dq = dq + a
        except UnboundLocalError:
            dq = a
    return dq

##Model building for dq curve with distribution    
def model_dq_distribution(connectivity):
            
    temp = [lm.Model(distribution_Dres_dq,prefix=fraction) for fraction in connectivity]
    for a in temp: 
        try:
            dq = dq + a
        except UnboundLocalError:
            dq = a
    return dq


##Model building for total curve
def model_tot(connectivity):
    temp = [lm.Model(T2_decay,prefix=fraction) for fraction in connectivity]
    for a in temp: 
        try:
            tot = tot + a
        except UnboundLocalError:
            tot = a
    tot = tot + lm.Model(T2_decay,prefix="Tail_")
    return tot

##Update old parameter values to new values
def update_params(parameter,value_dict):
    for key,values in value_dict.items():
        parameter[key].set(value=values)
    return parameter

##Update bounds to new parameter values
def update_bounds(parameter,value_dict,change=None,width=0.2):
    if change==None:
        change=["Dres" , "T2", "sigma"]
    for key,values in value_dict.items():
        if any(a == key for a in change) :
            parameter[key].set(min=values*(1-width),max=values*(1+width))
    return parameter

##Fit DQ curve only
def fit_DQ_only(df,parameter,model,connectivity,**kwagrs):
    dq_fitted=model.fit(df['I_DQ'],parameter,tau=df['Time'],**kwagrs)
    
    dq_fitted.plot_fit(show_init=True,title='I dq only fit')

    comp_dq=dq_fitted.eval_components()
    plotmq(df['Time'],**comp_dq)
    return dq_fitted

##Fit tot curve only
def fit_tot_only(df,parameter,model,connectivity,**kwagrs):

    tot_fitted=model.fit(df['I_tot'],parameter,tau=df['Time'],**kwagrs)
    
    tot_fitted.plot_fit(show_init=True,title='I total only fit')

    comp_tot=tot_fitted.eval_components()
    plotmq(df['Time'],y_axis='log',**comp_tot)
    return tot_fitted

#Definition of DQ curve fitting model
def fit_simultaneous(parameter,df,model_dq,model_tot,connectivity,T2_penalty,Dres_penalty,tail_cutoff,allow_overlap=False):
    df_truncated=df[df['Time']<tail_cutoff].copy()
    residual_dq = (df_truncated['I_DQ'] - model_dq.eval(parameter,tau=df_truncated['Time']))  #*0.5/ max(df_truncated['I_DQ'])

    residual_tot = df['I_tot'] - model_tot.eval(parameter,tau=df['Time'])
    
    residual = pd.concat([residual_dq,residual_tot])
    
    if allow_overlap==False:
        residual = residual * extract_and_penalize(parameter,connectivity,T2_penalty,Dres_penalty)

    err_A = 1-sum(grab(parameter,connectivity,"A","Tail_",False))
    residual = residual * (1+abs(err_A))**4
    return residual #Minimization parameter

#Extract T2 and Dres difference
def extract_diff(parameter,connectivity):
    diff_T2 = np.diff(grab(parameter,connectivity,"T2","Tail_",False))

    diff_Dres = np.diff(grab(parameter,connectivity,"Dres"))    
    return diff_T2,diff_Dres

#Extract and penalize T2 and Dres difference
def extract_and_penalize(parameter,connectivity,T2_penalty,Dres_penalty):
    
    diff_T2, diff_Dres = extract_diff(parameter,connectivity)
    a = 1 * overlap_penalty(diff_T2,**T2_penalty) * overlap_penalty(diff_Dres,**Dres_penalty)
    return a

#This section contains non standard codes
###################################################
#DQ model curve = abragam multiplied by T2 decay
def distribution_Dres_dq(tau,Dres,sigma,T2,A,beta):
    tau_s = tau/Dres_domain['scaling']
    T2_s = T2/Dres_domain['scaling']
    Dres_Hz,prob =Dres_prob_distribution(Dres,sigma,**Dres_domain)
    DQ=abragam_dist(tau_s,Dres_Hz,prob)*T2_decay(tau_s,T2_s,A,beta)
    return DQ

def Dres_prob_distribution(mean,sigma,start, end, division, scaling, dist):
    mean=mean*scaling
    if dist in [lognormal,lognormal2]:
        Dres=np.logspace(np.log10(start), np.log10(end),num=division)
    elif dist in [gaussian]:
        Dres=np.linspace(start, end,division)
    prob = dist(Dres,mean,sigma)
    return Dres,prob

###################################################
# Declaration of global variables
###################################################

Dres_domain={'start': 1,
                'end': 200,
                'division': 2000,
                'scaling': 1000,
                'dist':lognormal2
        }
#This section is for testing purpose only
###################################################

##x=np.linspace(1,100,1000)
##y=lognormal(x,np.log(50),0.1)
##y1=lognormal(x,np.log(153.7),0.1)
##
##
##tau = np.linspace(0.00001,0.025,2500)
##
##T2=0.005
##
##Idq= abragam_dist(tau,x,y)* T2_decay(tau,T2)
##Idq1=abragam_dist(tau,x,y1) * T2_decay(tau,T2)
##plt.plot(x,y1)
##plt.show()
##plotmq(x,y+y1,mu1=y,mu2=y1)
##plotmq(tau,Idq+Idq1,mu1=Idq,mu2=Idq1)

###################################################
