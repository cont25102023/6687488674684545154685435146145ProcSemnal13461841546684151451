# Lab 8

# module
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as s1
import scipy.signal as s2
import sounddevice as s3
import cmath
import time
import pickle # pentru salvare variabile pe disc
import os
import shutil
from scipy import misc, ndimage
import statsmodels as sm

#%%

# 1

N=1000
t=np.linspace(0,N-1,N).astype(np.uint32)
f=lambda x: 2*x**2/N
trend=f(t)
f1=lambda x: N/2*np.sin(2*np.pi*x*15/N)
f2=lambda x: N*np.sin(2*np.pi*x*8/N)
sezon=f1(t)+f2(t)
varMici=np.random.normal(0,N/2,N)
serie=trend+sezon+varMici
fig,ax=plt.subplots(2,2,num=1,figsize=(20,20),clear=True)
ax[0][0].plot(t,serie,label='Serie de timp',lw=1/2)
ax[0][0].legend(loc='best')
ax[0][1].plot(t,trend,label='Trend',lw=1/2)
ax[0][1].legend(loc='best')
ax[1][0].plot(t,sezon,label='Sezon',lw=1/2)
ax[1][0].legend(loc='best')
ax[1][1].plot(t,varMici,label='Variatii mici',lw=1/2)
ax[1][1].legend(loc='best')

#%%

# 2

a=1/2
s=np.zeros(N)
s[0]=serie[0]
for i in range(1,N):
    s[i]=a*serie[i]+(1-a)*s[i-1]
fig,ax=plt.subplots(4,num=2,figsize=(20,20),clear=True)
ax[0].plot(t,serie,label='Serie originala',lw=1/2)
ax[0].legend(loc='best')
ax[1].plot(t,s,label='Serie generata',lw=1/2)
ax[1].legend(loc='best')
interval=np.arange(0,1,0.01)
A=np.zeros_like(interval)
ss=np.zeros(N)
ss[0]=serie[0]
for j in range(len(interval)):
    a=interval[j]
    for i in range(1,N):
        ss[i]=a*serie[i]+(1-a)*ss[i-1]
    A[j]=sum((ss[:N-1]-serie[1:])**2)
ax[2].plot(interval,A,label='y=eroare(a)',lw=1/2)
ax[2].legend(loc='best')
a=interval[A.argmin()]
for i in range(1,N):
    s[i]=a*serie[i]+(1-a)*s[i-1]
ax[3].plot(t,serie,label=f'a={a}',lw=1/2)
ax[3].legend(loc='best')

#%%

# 3

