# Lab 6

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

#%%

# 1

N=100
x=np.random.uniform(0,1,N)
fig,ax=plt.subplots(4,figsize=(10,20),num=1,clear=True)
for i in range(4):
    ax[i].plot(range(len(x)),x)
    x=np.convolve(x,x)
# Se apropie de gaussiana (seamana cu densitatea normalei, scalata).

#%%

# 2

N=100
p=np.random.randint(0,10,N)
q=np.random.randint(0,10,N)
r1=np.zeros(2*N-1).astype(int)
for i in range(N):
    for j in range(N):
        r1[i+j]+=p[i]*q[j]
# simulare convolutie cu fft
# p1=np.zeros(2*N-1)
# q1=np.zeros(2*N-1)
# p1[:N]=p[:N]
# q1[:N]=q[:N]
# P=np.fft.fft(p1)
# Q=np.fft.fft(q1)
# R=P*Q
# r2=np.abs(np.fft.ifft(R)).round()
r2=np.convolve(p,q).round() # inmulteste polinoamele p si q
print('Ex2: A dat la fel:',(r1==r2).all())
fig,ax=plt.subplots(2,figsize=(10,20),num=2,clear=True)
ax[0].plot(range(2*N-1),r1,marker='o',linewidth=1,markersize=2)
ax[1].plot(range(2*N-1),r2,marker='o',linewidth=1,markersize=2)

#%%

# 3

def fer_dr(i,j,n):
    x=np.zeros(n)
    x[i:j]=np.ones(j-i)
    return x

def fer_Han(i,j,n):
    x=np.zeros(n)
    f=lambda k: np.ones(np.shape(k)[0])-np.cos(2*np.pi*k/n)
    x[i:j]=0.5*np.floor(f(np.array(range(i,j))))
    return x

f=100
A=1
phi=0

func=lambda t,f,A,phi: A*np.sin(phi+2*np.pi*f*t+phi)

n=1000
i=250
Nw=200
fig,ax=plt.subplots(3,figsize=(10,20),num=3,clear=True)
t=np.linspace(0,1,n)
x=func(t,f,A,phi)
ax[0].plot(t,x)
ax[1].plot(t,x*fer_dr(i,i+Nw,n))
ax[2].plot(t,x*fer_Han(i,i+Nw,n))

#%%

# 4

# a
file=open('Train.csv','r')
lst=list(map(lambda x: x.split(','),file.readlines()))
R=3
zileR=lst[1:2+24*R]
x=list(map(lambda i: int(i[2]),zileR))

# b
w=[5,9,13,17]
fig,ax=plt.subplots(len(w),figsize=(10,20),num='4b',clear=True)
for i in range(len(w)):
    y=np.convolve(x,np.ones(w[i]),'valid')/w[i]
    ax[i].plot(range(len(y)),y)

# c
'''
Aleg perioada de 10 ore (pe primul grafic de la 4b apar blocuri de aproximativ 10 ore).
frecventa_Hz = 1/10ore = 1/36000
frecventa normalizata: frecventa_Nyquist = 1/1ora/2 = 1/7200
wn = frecventa_Hz/frecventa_Nyquist = 1/5
'''
print('Ex4c:')
print('Frecventa in Hz: %.19f'%(1/36000))
print('Frecventa Nyquist in Hz:',1/7200)
print('Frecventa normalizata:',1/5)

# d
n=len(x)
u=range(n)
wn=1/5
f1=s2.butter(5,wn,btype='low')
f2=s2.cheby1(5,5,wn,btype='low')
filtre=[s2.cheby1(5,i,wn,btype='low') for i in range(1,11)]

# e
z1=s2.filtfilt(f1[0],f1[1],x)
z2=s2.filtfilt(f2[0],f2[1],x)
z=[s2.filtfilt(i[0],i[1],x) for i in filtre]
fig,ax=plt.subplots(3,figsize=(10,20),num='4e',clear=True)
ax[0].plot(u,x,label='semnal brut')
ax[0].legend(loc='best')
ax[0].plot(u,z1,label='semnal filtrat butter')
ax[0].legend(loc='best')
ax[1].plot(u,x,label='semnal brut')
ax[1].legend(loc='best')
ax[1].plot(u,z2,label='semnal filtrat Chebyshev')
ax[1].legend(loc='best')
i=np.random.randint(1,11)
ax[2].plot(u,x,label='semnal brut')
ax[2].legend(loc='best')
ax[2].plot(u,z[i-1],label=f'Chebyshev({i}dB)')
ax[2].legend(loc='best')

# f
r=np.random.randint(1,11)
f1=s2.butter(r,wn,btype='low')
f2=s2.cheby1(r,5,wn,btype='low')
filtre=[s2.cheby1(r,i,wn,btype='low') for i in range(1,11)]
z1=s2.filtfilt(f1[0],f1[1],x)
z2=s2.filtfilt(f2[0],f2[1],x)
z=[s2.filtfilt(i[0],i[1],x) for i in filtre]
fig,ax=plt.subplots(3,figsize=(10,20),num='4f',clear=True)
ax[0].plot(u,x,label='semnal brut')
ax[0].legend(loc='best')
ax[0].plot(u,z1,label=f'butter-atenuare({r})')
ax[0].legend(loc='best')
ax[1].plot(u,x,label='semnal brut')
ax[1].legend(loc='best')
ax[1].plot(u,z2,label=f'Chebyshev-atenuare({r})')
ax[1].legend(loc='best')
i=np.random.randint(1,11)
ax[2].plot(u,x,label='semnal brut')
ax[2].legend(loc='best')
ax[2].plot(u,z[i-1],label=f'Chebyshev({i}dB)-atenuare({r})')
ax[2].legend(loc='best')
# Valori optime ordin atenuare: 1,3,5,7,9
# Valori optime rp: 1,2,3,4,5,6,7,8,9,10