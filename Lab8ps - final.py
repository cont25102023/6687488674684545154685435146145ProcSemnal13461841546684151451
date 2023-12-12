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

#%%

# 1

# a
N=1000
t=np.array(list(range(N)))
f=lambda x: 2*x**2/N
trend=f(t)
f1=lambda x: N/2*np.sin(2*np.pi*x*15/N)
f2=lambda x: N*np.sin(2*np.pi*x*8/N)
sezon=f1(t)+f2(t)
varMici=np.random.normal(0,N/2,N)
serie=trend+sezon+varMici
fig,ax=plt.subplots(2,2,num='a',figsize=(20,20),clear=True)
ax[0][0].plot(t,serie,label='Serie de timp',lw=1/2)
ax[0][0].legend(loc='best')
ax[0][1].plot(t,trend,label='Trend',lw=1/2)
ax[0][1].legend(loc='best')
ax[1][0].plot(t,sezon,label='Sezon',lw=1/2)
ax[1][0].legend(loc='best')
ax[1][1].plot(t,varMici,label='Variatii mici',lw=1/2)
ax[1][1].legend(loc='best')

#%%

# b

t1=range(2*N-1)
x=serie-sum(serie)/N
x1=np.correlate(x,x,'full')
x2=s2.correlate(x,x,'full')
fig,ax=plt.subplots(3,num='b',figsize=(20,20),clear=True)
ax[0].plot(t1,x1,color='b',label='Fara FFT')
ax[0].plot(t1,x2,color='r',label='Cu FFT')
ax[0].legend(loc='best')
ax[1].plot(t1,x1,color='b',label='Fara FFT')
ax[1].legend(loc='best')
ax[2].plot(t1,x2,color='r',label='Cu FFT')
ax[2].legend(loc='best')
# x1 si x2 sunt vectori de autocorelatie pentru serie.
# Asa cum se vede in primul subplot, valorile lor coincid.
# x1[i](=x2[i]) reprezinta covarianta dintre serie siftata la dreapta cu i-N si serie.

#%%

# c

p=10
m=25
serie1=np.zeros(N)
serie1[:m+p-1]=serie[:m+p-1]
for n in range(m+p-1,N-1):
    y=serie[n-m+1:n+1]
    Y=np.zeros((m,p))
    for i in range(p):
        Y[:,i]=serie[n-m-i:n-i]
    G=np.transpose(Y)@Y
    g=np.transpose(Y)@y
    x=np.linalg.solve(G,g)
    serie1[n+1]=np.sum(x*y[:p])
serie2=np.zeros(N)
serie2[:m+p-1]=serie[:m+p-1]
for n in range(m+p-1,N-1):
    y=serie[n-m+1:n+1]
    Y=np.zeros((m,p))
    for i in range(p):
        Y[:,i]=serie2[n-m-i:n-i]
    G=np.transpose(Y)@Y
    g=np.transpose(Y)@y
    x=np.linalg.solve(G,g)
    serie2[n+1]=np.sum(x*y[:p])
fig,ax=plt.subplots(num='c',figsize=(20,20),clear=True)
ax.plot(t,serie,label='Serie initiala',lw=1/2,color='b')
ax.plot(t,serie1,label='Serie aproximata',lw=1/2,color='r')
ax.plot(t,serie2,label='Serie aproximata2',lw=1/2,color='g')
ax.legend(loc='best')

#%%

# d

MP=[]
m0=1
p0=1
err0=0
for n in range(m0+p0-1,N-1):
    y=serie[n-m0+1:n+1]
    Y=np.zeros((m0,p0))
    for i in range(p0):
        Y[:,i]=serie[n-m0-i:n-i]
    G=np.transpose(Y)@Y
    g=np.transpose(Y)@y
    x=np.linalg.solve(G,g)
    err0+=abs(np.sum(x*y[:p0])-serie[n+1])**2/(N-m0-p0)
for m in range(2,N-11):
    for p in range(1,min(m,N-12-m)):
        err=0
        for n in range(m+p-1,N-1):
            y=serie[n-m+1:n+1]
            Y=np.zeros((m,p))
            for i in range(p):
                Y[:,i]=serie[n-m-i:n-i]
            G=np.transpose(Y)@Y
            g=np.transpose(Y)@y
            x=np.linalg.solve(G,g)
            err+=abs(np.sum(x*y[:p])-serie[n+1])**2/(N-m-p)
            if err>=err0:
                break
        if err<err0:
            err0=err
            m0=m
            p0=p
            MP.append((m0,p0))
        print(m,p)
print(m0,p0)
# fis1=open('Lab8data\serie','wb')
# pickle.dump(serie,fis1)
# fis1.close()
# fis2=open('Lab8data\m','wb')
# pickle.dump(m0,fis2)
# fis2.close()
# fis3=open('Lab8data\p','wb')
# pickle.dump(p0,fis3)
# fis3.close()
# fis4=open('Lab8data\MP','wb')
# pickle.dump(MP,fis4)
# fis4.close()
#%%
fis1=open('Lab8data\serie','rb')
ser0=pickle.load(fis1)
fis1.close()
fis2=open('Lab8data\m','rb')
m=pickle.load(fis2)
fis2.close()
fis3=open('Lab8data\p','rb')
p=pickle.load(fis3)
fis3.close()
N=len(ser0)
t=np.array(list(range(N)))
serie1=np.zeros(N)
serie1[:m+p-1]=ser0[:m+p-1]
for n in range(m+p-1,N-1):
    y=ser0[n-m+1:n+1]
    Y=np.zeros((m,p))
    for i in range(p):
        Y[:,i]=ser0[n-m-i:n-i]
    G=np.transpose(Y)@Y
    g=np.transpose(Y)@y
    x=np.linalg.solve(G,g)
    serie1[n+1]=np.sum(x*y[:p])
serie2=np.zeros(N)
serie2[:m+p-1]=ser0[:m+p-1]
for n in range(m+p-1,N-1):
    y=serie2[n-m+1:n+1]
    Y=np.zeros((m,p))
    for i in range(p):
        Y[:,i]=serie2[n-m-i:n-i]
    G=np.transpose(Y)@Y
    g=np.transpose(Y)@y
    x=np.linalg.solve(G,g)
    serie2[n+1]=np.sum(x*y[:p])
fig,ax=plt.subplots(num='d',figsize=(20,20),clear=True)
ax.plot(t[m+p:],ser0[m+p:],label='Serie initiala',lw=1/2,color='b')
ax.plot(t[m+p:],serie1[m+p:],label='Serie aproximata',lw=1/2,color='r')
ax.plot(t[m+p:],serie2[m+p:],label='Serie aproximata2',lw=1/2,color='g')
ax.legend(loc='best')