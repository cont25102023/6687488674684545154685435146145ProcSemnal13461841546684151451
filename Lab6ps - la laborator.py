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
# Se apropie de gaussiana.

#%%

# 2

N=100
p=np.random.randint(0,10,N)
q=np.random.randint(0,10,N)
r1=np.zeros(2*N-1).astype(int)
for i in range(N):
    for j in range(N):
        r1[i+j]+=p[i]*q[j]
p1=np.zeros(2*N-1)
q1=np.zeros(2*N-1)
p1[:N]=p[:N]
q1[:N]=q[:N]
P=np.fft.fft(p1)
Q=np.fft.fft(q1)
R=P*Q
r2=np.abs(np.fft.ifft(R)).round()
print('A dat la fel:',(r1==r2).all())
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