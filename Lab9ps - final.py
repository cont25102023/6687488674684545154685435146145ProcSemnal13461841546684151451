# Lab 9

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
from statsmodels.tsa.arima.model import ARIMA

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
ax[3].plot(t,ss,label=f'a={a}',lw=1/2)
ax[3].legend(loc='best')

#%%

# 3

q=4
Q=np.zeros(q)
Q[0]=a # a calculat la 2
for i in range(1,q):
    Q[i]=Q[i-1]*(1-a)
Q[-1]/=a # Q este creat din primii parametri ai modelului exponential
med=sum(serie/N)
er=np.zeros(N)
ss=np.zeros(N)
ss[:q]=med
er[:q]=serie[:q]-med
for i in range(q,N):
    ss[i]=sum(er[i-q:i]*Q)+med
    er[i]=serie[i]-ss[i]
fig,ax=plt.subplots(3,num=3,figsize=(20,20),clear=True)
ax[0].plot(t,serie,label='Serie originala',lw=1/2)
ax[0].legend(loc='best')
ax[1].plot(t,ss,label='Serie generata',lw=1/2)
ax[1].legend(loc='best')
ax[2].plot(t,er,label='Eroare',lw=1/2)
ax[2].legend(loc='best')

#%%

# 4

n=20
p0=1
q0=1
model=ARIMA(serie,order=(p0,0,q0))
mfit=model.fit()
ss0=mfit.predict()
er0=sum(((serie-ss0)/N)**2)
for p in range(1,n+1):
    for q in range(1,n+1):
        try:
            model=ARIMA(serie,order=(p,0,q))
            mfit=model.fit()
            ss=mfit.predict()
            er=sum(((serie-ss)/N)**2)
            if er<er0:
                p0=p
                q0=q
                er0=er
                ss0=np.copy(ss)
            print(p,q)
        except:
            pass
print('\nParametri:\np =',p0,'\nq =',q0)
fig,ax=plt.subplots(3,num=4,figsize=(20,20),clear=True)
ax[0].plot(t,serie,label='Serie originala',lw=1/2)
ax[0].legend(loc='best')
ax[1].plot(t,ss0,label='Serie generata',lw=1/2)
ax[1].legend(loc='best')
er=serie-ss0
ax[2].plot(t,er,label='Eroare',lw=1/2)
ax[2].legend(loc='best')