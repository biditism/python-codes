# importing modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lmfit as lm
import bidit as my
import os
import shutil
from datetime import datetime


#############################################
#Definitions of constants and functions

pi = np.pi


Dres_domain={'start': None,
                'end': None,
                'division': 1000,
                'scaling': 1000,
                'dist':my.lognormal2
        }
my.Dres_domain = Dres_domain
#############################################

#Import the data
df,file = my.rawdata()


df_main = df.copy() #Backup of original table

#Cleaning of data

df = my.clean(df,1500)



##############################################
#Tail substraction
##############################################
tail_cutoff=600
tail_fitted = my.tail(df,tail_cutoff,vary_beta=False)


tail_result=tail_fitted.best_values

tail_freedom=0.05 #Percentage freedeom for tail to vary later

###############################################
#Initialize and define parameters and models for first component
###############################################
Idq_model= lm.Model(my.distribution_Dres_dq,prefix='first_')


Itot_model= lm.Model(my.T2_decay,prefix='first_')

dq_params = lm.Parameters()


dq_params.add('first_Dres',0.05,vary=True,min=0.01,max=0.2)
dq_params.add('first_sigma',0.5,vary=True,min=0.01,max=1.0)
dq_params.add('first_T2',10,vary=True,min=0,max=40)
dq_params.add('first_A',0.5,vary=True,min=0.01,max=0.6)
dq_params.add('first_beta',1.1,vary=True,min=1,max=2)

###############################################
#Define parameters and models for second component
###############################################
Idq_model= Idq_model + lm.Model(my.distribution_Dres_dq,prefix='second_')


Itot_model= Itot_model + lm.Model(my.T2_decay,prefix='second_')

dq_params.add('second_Dres',0.05,vary=True,min=0.005,max=0.1)
dq_params.add('second_sigma',0.5,vary=True,min=0.01,max=1.0)
dq_params.add('second_T2',21,vary=True,min=20,max=100)
dq_params.add('second_A',0.5,vary=True,min=0.01,max=0.6)
dq_params.add('second_beta',1.1,vary=True,min=1,max=2)

###############################################
#Define parameters and models for tail
###############################################

Itot_model= Itot_model + lm.Model(my.T2_decay,prefix='tail_')


dq_params.add('tail_T2',tail_result['T2'],vary=True,
              min=tail_result['T2']*(1-tail_freedom),
              max=tail_result['T2']*(1+tail_freedom))
dq_params.add('tail_A',tail_result['A'],vary=True,
              min=tail_result['A']*(1-tail_freedom),
              max=tail_result['A']*(1+tail_freedom))
dq_params.add('tail_beta',tail_result['beta'],vary=False,min=0)


###############################################
#Define parameters and models for third component
###############################################

Idq_model= Idq_model + lm.Model(my.single_Dres_dq,prefix='third_')


Itot_model= Itot_model + lm.Model(my.T2_decay,prefix='third_')


dq_params.add('third_Dres',0.005,vary=True,min=0.001,max=0.01)
dq_params.add('third_T2',75,vary=True,min=75,max=tail_result['T2'])
dq_params.add('third_A',0.5,vary=True,min=0.01,
              expr='1-(first_A+second_A+tail_A)')
dq_params.add('third_beta',1,vary=True,min=1,max=2)                        



###############################################
#Define other necessary variables
###############################################
connectivity=['first_','second_','third_']


#Define overlap penalty
T2_penalty=dict(amount=1, width=1)
Dres_penalty=dict(amount=1, width=0.005)


##############################################
#Preliminary seperate fit to determine initial values
##############################################

  
#Fitting of DQ curve
dq_fitted= my.fit_DQ_only(df,dq_params,Idq_model,connectivity,
                          method='leastsq')


#Update the parameters based on the above fit
dq_params= my.update_params(dq_params,dq_fitted.best_values)
    
print(dq_params.pretty_print())
        
        
#Fitting of Total curve
tot_fitted= my.fit_tot_only(df,dq_params,Itot_model,connectivity,method='leastsq')


#Update the parameters based on the above fit
dq_params= my.update_params(dq_params,tot_fitted.best_values)

    # print(dq_params.pretty_print())

    #Update bounds based on the initial fitting
    #dq_params= my.update_bounds(dq_params,tot_fitted.best_values)


#dump parameters to a file
f = open("initial_parameters.json", "w")
dq_params.dump(f)
f.close()  



#Simultaneous fitting of DQ and Total curve
dq_cutoff=300
sim_fit=lm.Minimizer(
    my.fit_simultaneous,dq_params,
    fcn_args=(df,Idq_model,Itot_model,connectivity,T2_penalty,Dres_penalty,dq_cutoff),
    max_nfev=10000000)

search=False
repeat=5

if search is True:
    A=np.arange(0.1, 0.9,0.05)
    parameter= 'first_A'
    search_result=[]

    for x in A:
        dq_params[parameter].set(value=x,vary=False)
        lowest=sim_fit.minimize(method='leastsq',params=dq_params)
        for x in range(repeat-1):
            temp =sim_fit.minimize(method='leastsq',params=dq_params)
            if temp.chisqr < lowest.chisqr:
                lowest=temp
        search_result.append(lowest)
    chi_sqr =[a.chisqr for a in search_result]

    plt.plot(A, chi_sqr)
    plt.savefig(file+'_chisqr.pdf', format="pdf", bbox_inches="tight")
    plt.savefig(file+'_chisqr.png', format="png", bbox_inches="tight")
    plt.show()

    #Pickel the minimizer result object
    my.write_object(search_result,file+'_search.pckl')


    sim_fitted= search_result[chi_sqr.index(min(chi_sqr))]

else:

    sim_fitted= sim_fit.minimize(method='basinhopping',params=dq_params)


print(sim_fitted.params.pretty_print())


fitted_points_dq=dict(
    **Idq_model.eval_components(params=sim_fitted.params,tau=df["Time"]),
    Total_Fit_=Idq_model.eval(params=sim_fitted.params,tau=df["Time"]))

fitted_points_tot=dict(
    **Itot_model.eval_components(params=sim_fitted.params,tau=df["Time"]),
    Total_Fit_=Itot_model.eval(params=sim_fitted.params,tau=df["Time"]))

plt.xlim(right=tail_cutoff)

##############################################
#Plot and save graphs
##############################################

my.plotmq(df["Time"],df['I_DQ'], y_axis='log',
          save=file+'_IDQcomp',**fitted_points_dq)


my.plotmq(df["Time"],df["I_tot"], y_axis='log',
          save=file+'_Itotcomp',**fitted_points_tot)

plt.plot(df["Time"],df['I_DQ']-fitted_points_dq['Total_Fit_'],label='DQ Residual')

plt.plot(df["Time"],df['I_tot']-fitted_points_tot['Total_Fit_'],label='Total Residual')

plt.savefig(file+'_residuals.pdf', format="pdf", bbox_inches="tight")
plt.savefig(file+'_residuals.png', format="png", bbox_inches="tight")
plt.show()


##############################################
#Write outputs to file
##############################################

#dump final parameters to a file
f = open(file+"_fit_parameters.json", "w")
sim_fitted.params.dump(f)
f.close()  


#Write the fitted data points to file
df_dq = pd.DataFrame(fitted_points_dq).add_suffix('dq')
df_tot = pd.DataFrame(fitted_points_tot).add_suffix('tot')
df_result = df_dq.copy().assign(**df_tot)
df_result.to_csv(file+'_fit_value.csv',index=False)

print(lm.fit_report(sim_fitted))

#Write fit report to the file
file1 = open(file+"_fit_report.txt", "w")
print(lm.fit_report(sim_fitted),file=file1)
file1.close()

#Pickel the minimizer result object
my.write_object(sim_fitted,file+'_minimized.pckl')

##############################################
#Callculate confidence interval
##############################################

calculate_ci=False

if calculate_ci is True:
    for p in sim_fitted.params:
        if sim_fitted.params[p].stderr is None:
            sim_fitted.params[p].stderr = abs(sim_fitted.params[p].value * 0.1)

    ci,trace=lm.conf_interval(sim_fit, sim_fitted,trace=True)

    lm.printfuncs.report_ci(ci)

    #Pickel the confidence interval object
    my.write_object(ci,file+'_ci.pckl')
    my.write_object(trace,file+'_ci-trace.pckl')




now=datetime.now()
path='./temp/'+ now.strftime('%Y%m%d%H%M%S')+'/'
os.makedirs(path,exist_ok=True)

for f in os.listdir(os.curdir):
    if f.startswith(file):
        shutil.copy2(f, path)
    
##################################################
#Plot the Dres distribution
##################################################
prob=dict()
for a in ['first_','second_']:
    domain,prob[a]=my.Dres_prob_distribution(sim_fitted.params[a+"Dres"],
                                             sim_fitted.params[a+"sigma"],
                                             **Dres_domain)
    plt.plot(domain,prob[a])
    #plt.xlim(0, 200)                        
    plt.xscale('log')
    #plt.yscale('log')
for a in ['third_']:
    plt.axvline(sim_fitted.params[a+"Dres"].value*Dres_domain['scaling'])
plt.show()