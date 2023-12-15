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

###################################################
# Creation of fitting models
###################################################



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
