#!/usr/bin/env python
# coding: utf-8

# In[3]:


# importing modules

import sys
import csv
import nmrglue as ng
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import other_functions as oth


# In[4]:


dic,fid = ng.bruker.read('./')
exp_info = oth.read_exp_parameters()

# In[5]:


# =============================================================================
# Define conditions
# =============================================================================

dwell_time=exp_info['dwell_time']*1e-6
dead_time=dic['acqus']['DE']*1e-6
p1=dic['acqus']['P'][1]*1e-6

try:
	grpdly=exp_info['grpdly']
except:
	if dwell_time<0.11 * 1e-6: #Groupdelay for 200 MHz
		grpdly=9
	else:
		grpdly=20
print(f'{exp_info['EXP']}: Dead time = {dead_time}, dwell time = {dwell_time}, 90 pulse = {p1}, group delay = {grpdly}')

# In[6]:

#NMR glue's method of correcting digital filter artifact
#fid_dig_filter_removed=ng.bruker.remove_digital_filter(dic,fid,truncate=True)


# In[9]:


#plt.plot(fid.real)
#plt.plot(fid.imag)
#plt.show()


phase = {"p0": 0, "p1":0}


read_phase=exp_info.get('read_phase_correction',False)#Do phase correction if not stated

if read_phase==True:

	try:
	        with open("phase.txt") as f:
	                phase['p0']=float(f.readline())
	                phase['p1']=float(f.readline())
	except:
		print("No phase.txt file. Phase correction from TopSpin will be used.")
		p0,p1=dic['procs']['PHC0'],dic['procs']['PHC1']
	fid_phase_corrected=ng.proc_base.ps(fid, p0=phase['p0'] , p1=phase['p1'])

# In[10]:
	
else:

	phase['p0'],phase['p1']=oth.phasecorr_time_domain(fid)
	# In[11]:


	fid_phase_corrected=ng.proc_base.ps(fid, p0=phase['p0'] , p1=phase['p1'])
	#plt.plot(fid_phase_corrected[22:200].real)
	#plt.plot(fid_phase_corrected[22:200].imag)
	#plt.grid()
	#plt.show()

with open('phase.txt', 'w') as file: file.write(f'{phase['p0']}\n{phase['p1']}')



# In[12]:


fid_group_delay_corrected= fid_phase_corrected[grpdly:]
#plt.plot(fid_group_delay_corrected[:200].real)
#plt.plot(fid_group_delay_corrected[:200].imag)
#plt.show()
spectra=ng.proc_base.fft(fid_group_delay_corrected)
#plt.plot(spectra)
#plt.show()


# In[13]:


time_axis= p1/2 + dead_time + dwell_time * (np.arange(len(fid_group_delay_corrected))+1)
print(time_axis)


# In[15]:


df_fid=pd.DataFrame()
df_fid["time"]=time_axis*1e3
df_fid["real_intensity"]=fid_group_delay_corrected.real
df_fid["imaginary_intensity"]=fid_group_delay_corrected.imag
df_fid["magnitude"]=np.sqrt(df_fid["real_intensity"]**2+df_fid["imaginary_intensity"]**2)
#plt.plot(df_fid['time'],df_fid['real_intensity'])
#plt.plot(df_fid['time'],df_fid['imaginary_intensity'])
#plt.show()


# In[16]:


fit_fid=df_fid[(df_fid['time']<=0.2)]#&(df_fid['time']>=2)
#plt.plot(fit_fid['time'],fit_fid['real_intensity'])
#plt.plot(fit_fid['time'],fit_fid['imaginary_intensity'])
#plt.plot(fit_fid['time'],fit_fid['magnitude'])
#plt.show()


# In[17]:


fit_fid.to_csv("clean_fid.txt",index=False,header=False)

