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
        penalty = penalty * heaviside(x, amount, width)
    return penalty

#Abragam like function with a single Dres
def abragam(tau,Dres,A=0.5): 
    return A*(1- np.exp(-(0.378*2*np.pi*Dres*tau)**1.5)*np.cos(0.583*2*np.pi*Dres*tau))

#Abragam like function with a Dres distribution
def abragam_dist(tau_input,Dres, prob, A=0.5): 
    dq = np.empty_like(tau_input)
    #A = A * len(prob)
    for index, tau in np.ndenumerate(tau_input):
        y=(1- np.exp(-(0.378*2*np.pi*Dres*tau)**1.5)*np.cos(0.583*2*np.pi*Dres*tau)) * prob
        dq[index] = A * np.trapz(y,Dres)
    return dq

#Transverse relaxation
def T2_decay(tau,T2,A=1,beta=1): 
    return A*np.exp(-(tau/T2)**beta)

#DQ model curve = abragam multiplied by T2 decay
def single_Dres_dq(tau,Dres,T2,A,beta):
    return abragam(tau,Dres)*T2_decay(tau,T2,A,beta)


#Normal distribution
def gaussian(x,mu=0,sigma=1,normalize=True):
    prob= (sigma*np.sqrt(2*np.pi))**(-1) * np.exp((-1/2)*(((x-mu)/sigma)**2))
    if normalize==True: prob = prob / np.trapz(prob,x)
    return prob

#Lognormal distribution
def lognormal(x,mu=0,sigma=1,normalize=True):
    prob= (x*sigma*np.sqrt(2*np.pi))**(-1) * np.exp((-1/2)*(((np.log(x)-mu)/sigma)**2))
    if normalize==True: prob = prob / np.trapz(prob,x)
    return prob

#Distribution with normal distribution of log(x) 
def lognormal2(x,mu=1,sigma=1,normalize=True):
    prob = gaussian(np.log(x),np.log(mu),sigma,normalize)
    if normalize==True: prob = prob / np.trapz(prob,x)
    return prob

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

    plotmq(df['Time'],df['I_DQ'],df['I_ref'],df['I_tot'])
    
    if cutoff==None:
        finish=1
        while finish ==1:
            cutoff = float(input('Enter the time cutoff for further calculations'))
            df_check = df[df['Time'] <= cutoff].copy()
            plotmq(df_check['Time'],df_check['I_DQ'],df_check['I_ref'],df_check['I_tot'])
            finish = int(input('Press 1 if you want another cutoff'))
    return df[df['Time'] <= cutoff]

#Scatter plot of multiple files
def plotmq(tau,*args,y_axis='linear',**kwargs):
    for I in args:
        plt.scatter(tau,I)
    for key,values in kwargs.items():
        plt.plot(tau,values,label=key)
        plt.legend(loc='upper right')
    plt.yscale(y_axis)
    xmin, xmax, ymin, ymax = plt.axis()
    plt.ylim(bottom=0.001,top=min(ymax,1))
    plt.show()




##Stop program because it is being tested
def test(status=None):
    if status == None: raise ValueError('Exiting because program is set to test')

###################################################
# Creation of fitting models
###################################################

#Tail sustraction
def tail(df,tau_start=None,tail_model=None,params=None):
    #Ready the model   
    if tail_model==None: tail_model=lm.Model(T2_decay)

    #Ready the parameters
    if params==None:
        params = lm.Parameters() 
        params.add('A', value=0.1,min=0.05,max=0.9) #Fraction of tail
        params.add('T2', value=max(df['Time'])/3,min=50,max=500) #T2 of tail
        params.add('beta', value=1,vary=False) #Stretching exponent

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

        tail_fit.plot()
        plt.show()
        
        
        plotmq(df['Time'],df['I_tot'],df['I_DQ'],y_axis='log',line=fitted)

        
        plotmq(df['Time'],df['I_tot'],df['I_DQ'],df['I_nDQ'],line=fitted)

    ##Plot the result and graph

        
        
        if a!=None: return tail_fit
        
        finish = int(input('Press 1 if you want another cutoff'))    

    return tail_fit
        
##Prameter initialization for dq curve without distribution
def model_dq_no_distribution(comp=3):
    if comp==3:
        connectivity=["Single_","Double_","Higher_"]
    else:
        connectivity=[input("Prefix for each component") for a in range(comp)]
        
    temp = [lm.Model(single_Dres_dq,prefix=fraction) for fraction in connectivity]
    for a in temp: 
        try:
            dq = dq + a
        except UnboundLocalError:
            dq = a
    return dq,connectivity

##Prameter initialization for total curve without distribution
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

##Fit DQ curve only
def fit_DQ_only(df,parameter,model,connectivity,**kwagrs):

    dq_fitted=model.fit(df['I_DQ'],parameter,tau=df['Time'],**kwagrs)
    print(dq_fitted.fit_report())
    dq_fitted.plot(show_init=True,title='I dq only fit')

    fit_dq=dq_fitted.eval()
    comp_dq=dq_fitted.eval_components()
    plotmq(df['Time'],**comp_dq)
    return dq_fitted

##Fit tot curve only
def fit_tot_only(df,parameter,model,connectivity,**kwagrs):

    tot_fitted=model.fit(df['I_tot'],parameter,tau=df['Time'],**kwagrs)
    print(tot_fitted.fit_report())
    tot_fitted.plot(show_init=True,title='I total only fit')

    fit_tot=tot_fitted.eval()
    comp_tot=tot_fitted.eval_components()
    plotmq(df['Time'],y_axis='log', Total=fit_tot,**comp_tot)
    return tot_fitted



#This section contains non standard codes
###################################################


    

#This section is for testing purpose only
###################################################
##Tail fitting model
##tail = lm.Model(T2_decay)
##
##x=np.linspace(1,100,10000)
##y=gaussian(x,60,20)
##y=lognormal(x,np.log(60),0.1)
##y=lognormal2(x,60,0.1)
##
##tau = np.linspace(0.00001,0.025,2500)
##
##T2=0.005
##
##Idq= abragam_dist(tau,x,y)* T2_decay(tau,T2)
##Idq1=abragam(tau, 60) * T2_decay(tau,T2)
##
##
##print(tail.param_names,tail.independent_vars)
##
##
##plt.plot(x,y)
##plt.plot(x,y1)
##plt.plot(x,y2)
##plt.show()
##
##plt.plot(tau,Idq)
##plt.plot(tau,Idq1)
##plt.show()

###################################################
