###################################################
# This is my personal collection of mathematical functions
###################################################

import numpy as np
#import latexify
import matplotlib.pyplot as plt
from scipy.stats import lognorm

#Smooth Heaviside funtion
def heaviside(x, amplitude=1, steepness=1, base=1, center=0):
    return base + amplitude * (0.5 + 0.5 * np.tanh((x-center) * steepness))


#Abragam like function with a single Dres
def abragam(tau,Dres,A=0.5): 
    intensity = A*(1- np.exp(-(0.378*2*np.pi*Dres*tau)**1.5)*np.cos(0.583*2*np.pi*Dres*tau))
    return intensity

#Abragam like function with a Dres distribution
def abragam_dist(tau_input,Dres_med,Dres_sigma, A=0.5):
    tau = np.array(tau_input)

    lower,upper=lognorm.interval(0.99, Dres_sigma,scale=Dres_med)
    Dres =np.linspace(lower,upper,100)
    prob=lognorm.pdf(Dres,Dres_sigma,scale=Dres_med)
    prob=prob/np.trapz(prob,Dres)
    
    y = (1 - np.exp(-(0.378 * 2 * np.pi * Dres * tau[:, np.newaxis]) ** 1.5) *
         np.cos(0.583 * 2 * np.pi * Dres * tau[:, np.newaxis])) * prob

    intensity = A * np.trapz(y, Dres, axis=1)

    return intensity

#DQ intensity with single Dres and single T2
def DQ_1_Dres_1_T2(tau,Dres,T2,A,beta):
    intensity = abragam(tau,Dres)*T2_decay(tau,T2,A,beta)
    return intensity

#DQ intensity with a Dres distrubution and single T2
def DQ_dist_Dres_1_T2(tau,Dres_med,Dres_sigma,T2,A,beta):
    intensity = abragam_dist(tau,Dres_med,Dres_sigma)*T2_decay(tau,T2,A,beta)
    return intensity

#DQ intensity with a Dres distrubution and a T2 distribution
def DQ_dist_Dres_dist_T2(tau,Dres,prob_Dres,T2,prob_T2,A,beta):
    intensity = abragam_dist(tau,Dres,prob_Dres)*T2_decay(tau,T2,prob_T2,A,beta)
    return intensity


#Transverse relaxation with a single T2
def T2_decay(tau,T2,A=1,beta=1):
    intensity = A*np.exp(-(tau/T2)**beta)
    return intensity

#Transverse relaxation with a T2 distribution
def T2_decay_dist(tau,T2,prob,A=1,beta=1):
    tau = np.array(tau)

    y = np.exp(-(tau[:, np.newaxis]/T2) ** beta) * prob

    intensity = A * np.trapz(y, T2, axis=1)
    return intensity


#Normal distribution
def gaussian(x,mu=0,sigma=1,normalize=True):
    sigma = 0.1* mu * sigma
    prob = (sigma*np.sqrt(2*np.pi))**(-1) * np.exp((-1/2)*(((x-mu)/sigma)**2))
    if normalize==True: prob = prob / np.sum(prob)
    return prob

#Lognormal distribution
def lognormal(x,mu=1,sigma=0.1,normalize=True):
    mu = np.log(mu)
    prob = (x*sigma*np.sqrt(2*np.pi))**(-1) * np.exp((-1/2)*(((np.log(x)-mu)/sigma)**2))
    if normalize==True: prob = prob / np.sum(prob)
    return prob

#Distribution with normal distribution of log(x) 
def lognormal2(x,mu=1,sigma=1,normalize=True):
    prob = gaussian(np.log(x),np.log(mu),sigma,normalize)
    # print(x,prob,mu,sigma)
    if normalize==True: prob = prob / np.sum(prob)
    return prob

#Intensity decay due to diffusion in a field gradient
def diffusion_decay(gsquared,D,k,A=1):
    intensity = A * np.exp(-k*D*gsquared)
    return intensity

#Anderson Weiss Power Law DQ intensity
def AWPL_dq(tau,Dres,k,t0):
    
    
    exponential= np.exp(-1*(((1/5)*Dres**2)/((k-2)*(k-1)))*((k-k**2)*t0**2)+((2*k**2-4*k)*tau*t0+2*tau**(2-k)*t0**k))
    print(exponential)
    hyposinosuidal=np.sinh((((1/5)*Dres**2)/(2*(k-2)*(k-1)))*((k**2-k)*t0**2)+((2**(3-k)-4)*tau**(2-k)*t0**k))
    intensity = exponential * hyposinosuidal
    return intensity

def AWPL_dq_decay(tau,Dres,T2,k,t0):
    return AWPL_dq(tau,Dres,k,t0) * T2_decay(tau,T2)

#Anderson Weiss Power Law MQ intensity
def AWPL_MQ(tau,Dres,k,t0):
    intensity = np.exp(-1*(((1/5)*Dres**2)/((k-2)*(k-1)))*((3/2)*(k-k**2)*t0**2)+((2*k**2-4*k)*tau*t0+(4-2**(2-k))*tau**(2-k)*t0**k))
    return intensity

def AWPL_MQ_decay(tau,Dres,T2,k,t0):
    intensity = AWPL_MQ(tau,Dres,k,t0) * T2_decay(tau,T2)
    
    return intensity
# a=np.linspace(0.1,100,100000)
# y=AWPL_dq_decay(a, 0.1, 1,1.2, 0.099)
# plt.plot(a,y)
# plt.show()

#Fitting of first points of a few numbers

