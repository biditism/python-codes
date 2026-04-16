#!/usr/bin/env python
# coding: utf-8

# importing modules

# In[1]:


import sys
import csv
import nmrglue as ng
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import other_functions as oth


# Read file and experiment parameter

# In[2]:


dic,fid = ng.bruker.read('./')
exp_info = oth.read_exp_parameters()


# # =============================================================================
# # Define conditions
# # =============================================================================

# In[3]:


dwell_time=exp_info['dwell_time']*1e-6
dead_time=dic['acqus']['DE']*1e-6
p1=dic['acqus']['P'][1]*1e-6


# In[ ]:


try:
	grpdly=exp_info['grpdly']
except:
	if dwell_time<0.11 * 1e-6: #Groupdelay for 200 MHz
		grpdly=9
	else:
		grpdly=20
print(f'{exp_info['EXP']}: Dead time = {dead_time}, dwell time = {dwell_time}, 90 pulse = {p1}, group delay = {grpdly}')


# In[4]:


TD=len(fid)


# for a in fid[:,]:
#     plt.plot(a.real)
#     plt.plot(a.imag)
# plt.show()

# In[5]:


phase = {"p0": np.zeros(TD), "p1":np.zeros(TD)}

fid_phase_corrected=fid.copy()

read_phase=exp_info.get('read_phase_correction',False)#Do phase correction if not stated

# In[6]:


if read_phase==True:
        try:
                phase['p0'],phase['p1']=np.loadtxt('phase.txt')
        except:
                print("No phase.txt file. Phase correction from TopSpin will be used.")
                phase['p0'],phase['p1']=phase['p0']+dic['procs']['PHC0'],phase['p1']+dic['procs']['PHC1']
        for i in range(TD):
            fid_phase_corrected[i]=ng.proc_base.ps(fid[i], p0=phase['p0'][i] , p1=phase['p1'][i])
else:
    next=0
    for i in range(TD):
        if i==int(next):
                p0,p1=oth.phasecorr_time_domain(fid[i,:])
                next=input(f"Next FID with different phase correction (current  is {i}):")
        fid_phase_corrected[i]=ng.proc_base.ps(fid[i], p0=p0 , p1=p1)
        phase['p0'][i],phase['p1'][i]=p0,p1
        

# In[7]:


np.savetxt('phase.txt',(phase['p0'],phase['p1']))


# for a in fid_phase_corrected:
#     plt.plot(a[22:200].real)
#     plt.plot(a[22:200].imag)
# plt.show()

# In[8]:


fid_group_delay_corrected= fid_phase_corrected[:,grpdly:]
spectra=ng.proc_base.fft(fid_group_delay_corrected)


# for a in fid_group_delay_corrected:
#     plt.plot(a[:200].real)
#     plt.plot(a[:200].imag)
# plt.show()
# 
# for a in spectra:
#     plt.plot(a[470:530].real)
# plt.show()

# In[9]:


time_axis= p1/2 + dead_time + dwell_time * (np.arange(len(fid_group_delay_corrected[0]))+1)
print(time_axis)


# In[10]:


df_fid=pd.DataFrame()
df_fid["time"]=time_axis*1e3
for i,a in enumerate(fid_group_delay_corrected):
    df_fid["real_intensity"]=a.real
    df_fid["imaginary_intensity"]=a.imag
    df_fid["magnitude"]=np.sqrt(df_fid["real_intensity"]**2+df_fid["imaginary_intensity"]**2)
    df_fid["TD"]=i
    df_fid.to_csv(f"clean_fid_{i}.txt",index=False,header=False)


    plt.plot(df_fid['time']+i*max(df_fid['time']),df_fid['real_intensity'])
    plt.plot(df_fid['time']+i*max(df_fid['time']),df_fid['imaginary_intensity'])
plt.show()


