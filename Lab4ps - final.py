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
import os
import shutil

shutil.rmtree('Lab4psFig',ignore_errors=True)
os.mkdir('Lab4psFig')
# os.makedirs('Lab4psFig',exist_ok=True)
filnum='Lab4psFig\\'
nr=0

#%%

# 1

fig,ax=plt.subplots(1,figsize=(20,20))
l=[128,256,512,1024,2048,4096,8192]

# v=[]
# for n in l:
#     x=np.random.sample(n)
#     t1=time.perf_counter_ns()
#     fr=np.zeros((n,n)).astype(complex)
#     for i in range(n):
#         for j in range(n):
#             fr[i][j]=cmath.exp(-2*np.pi*i*j/n*1j)
#     X=fr@x
#     t2=time.perf_counter_ns()
#     Y=np.fft.fft(x)
#     t3=time.perf_counter_ns()
#     v.append(np.log10(t2-t1)-np.log10(t3-t2))
#     print(t2-t1,t3-t2)
# ax.plot(l,v,marker='o')

# NU VA PARALELIZA NIMIC
def f(n):
    x=np.random.sample(n)
    t=list(range(n))
    t1=time.perf_counter_ns()
    fr=np.array(list(map(lambda i: list(map(lambda j: 
        cmath.exp(-2*np.pi*i*j/n*1j),t)),t)))
    X=fr@x
    t2=time.perf_counter_ns()
    Y=np.fft.fft(x)
    t3=time.perf_counter_ns()
    print(t2-t1,t3-t2)
    return np.log10(t2-t1)-np.log10(t3-t2)
v=list(map(f,l))
ax.plot(l,v,marker='o')

plt.savefig(filnum+str(nr)+'.png')
plt.savefig(filnum+str(nr)+'.pdf')
nr=nr+1

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

plt.savefig(filnum+str(nr)+'.png')
plt.savefig(filnum+str(nr)+'.pdf')
nr=nr+1

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

plt.savefig(filnum+str(nr)+'.png')
plt.savefig(filnum+str(nr)+'.pdf')
nr=nr+1

#%%

# 4

# fs>=2*max(frecvente)=2*200=400

# 5

# Nu se pot distinge vocalele pe baza spectogramei
# Vezi Lab4VizAudacity.png

#%%

# 6

# a
rate,x=s1.read('voc.wav')
# Prelucrat cu Audacity pentru a fi compatibil cu scipy.io.wavfile.
# Sunetul original se afla in vocale.wav
# b
l=[[]]*199
n=len(x)
n0=n//199
t=list(range(n0))
for i in range(199):
    l[i]=list(map(lambda j: x[i*n0+j],t))
# c
l=list(map(lambda i:np.fft.fft(i)/n0,l))
# d
l=np.transpose(np.array(list(map(lambda i:i*np.conjugate(i),l)))).astype(float)
# e
plt.figure(figsize=(12.5,8))
plt.imshow(l,aspect=0.2)

plt.savefig(filnum+str(nr)+'.png')
plt.savefig(filnum+str(nr)+'.pdf')
nr=nr+1

#%%

# 7

p=90
snrdb=80
pzgomot=p/10**(snrdb/10)
print(pzgomot,'dB')