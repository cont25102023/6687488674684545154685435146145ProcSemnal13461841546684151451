# Lab 4

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

#%%

# 1

fig,ax=plt.subplots(1,figsize=(20,20))
l=[128,256,512,1024,2048,4096,8192]
for n in l:
    x=np.random.sample(n)
    t1=time.perf_counter_ns()
    fr=np.zeros((n,n)).astype(complex)
    for i in range(n):
        for j in range(n):
            fr[i][j]=cmath.exp(-2*np.pi*i*j/n*1j)
    X=fr@x
    t2=time.perf_counter_ns()
    Y=np.fft.fft(x)
    t3=time.perf_counter_ns()
    ax.plot(n/128,np.log(t2-t1)-np.log(t3-t2),marker='o')
    print(t2-t1,t3-t2)

#%%

# 2

fig,ax=plt.subplots(4,figsize=(20,20))
f1=lambda x: np.sin(2*np.pi*37*x) # frecventa sinusoidei e 37 Hz
f2=lambda x: np.sin(2*np.pi*21*x)
f3=lambda x: np.sin(2*np.pi*5*x)
t=np.arange(0,0.5,0.0625) # frecventa de esantionare e 16 Hz
s=np.arange(0,0.5,0.001)
ax[0].plot(s,f1(s))
ax[1].plot(s,f1(s))
ax[1].plot(t,f1(t),linestyle='None',marker='o')
ax[2].plot(s,f2(s))
ax[2].plot(t,f2(t),linestyle='None',marker='o')
ax[3].plot(s,f3(s))
ax[3].plot(t,f3(t),linestyle='None',marker='o')

#%%

# 3

fig,ax=plt.subplots(4,figsize=(20,20))
f1=lambda x: np.sin(2*np.pi*37*x) # frecventa sinusoidei e 37 Hz
f2=lambda x: np.sin(2*np.pi*21*x)
f3=lambda x: np.sin(2*np.pi*5*x)
t=np.arange(0,0.5,0.0125) # frecventa de esantionare e 80 Hz
s=np.arange(0,0.5,0.001)
ax[0].plot(s,f1(s))
ax[1].plot(s,f1(s))
ax[1].plot(t,f1(t),linestyle='None',marker='o')
ax[2].plot(s,f2(s))
ax[2].plot(t,f2(t),linestyle='None',marker='o',label='esantioanele calculate prin functia f2')
ax[2].legend(loc='best')
ax[2].plot(t,f1(t),linestyle='None',marker='o',label='esantioanele calculate prin functia f1')
ax[2].legend(loc='best')
ax[3].plot(s,f3(s))
ax[3].plot(t,f3(t),linestyle='None',marker='o',label='esantioanele calculate prin functia f3')
ax[3].legend(loc='best')
ax[3].plot(t,f1(t),linestyle='None',marker='o',label='esantioanele calculate prin functia f1')
ax[3].legend(loc='best')

#%%

# 4

# fs>=2*max(frecvente)=2*200=400

#%%

# 7

p=90
snrdb=80
pzgomot=p/10**(snrdb/10)
print(pzgomot,'dB')