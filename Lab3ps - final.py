# Lab 3

# module
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as s1
import scipy.signal as s2
import sounddevice as s3
import cmath
import os
import shutil

shutil.rmtree('Lab3psFig',ignore_errors=True)
os.mkdir('Lab3psFig')
# os.makedirs('Lab3psFig',exist_ok=True)
filnum='Lab3psFig\\'
nr=0

#%%

# 1

n=8
fr=np.zeros((n,n)).astype(complex)
for i in range(n):
    for j in range(n):
        fr[i][j]=cmath.exp(2*np.pi*i*j/n*1j)
t=list(range(n))
fig,a=plt.subplots(n,figsize=(10,3*n))
for i in range(n):
    a[i].plot(t,fr[i].real)
plt.savefig(filnum+str(nr)+'.png')
plt.savefig(filnum+str(nr)+'.pdf')
nr=nr+1
fig,a=plt.subplots(n,figsize=(10,3*n))
for i in range(n):
    a[i].plot(t,fr[i].imag)
plt.savefig(filnum+str(nr)+'.png')
plt.savefig(filnum+str(nr)+'.pdf')
nr=nr+1
frt=np.zeros((n,n)).astype(complex)
for i in range(n):
    for j in range(n):
        frt[i][j]=np.conj(fr[j][i])
b=(fr@frt)-n*np.identity(n)
print(b)

#%%

# 1

n=8
fr=np.zeros((n,n)).astype(complex)
for i in range(n):
    for j in range(n):
        fr[i][j]=cmath.exp(2*np.pi*i*j/n*1j)
b=fr@np.transpose(np.conjugate(fr))-n*np.identity(n)
print(b)
s=0+0j
for i in range(n):
    for j in range(n):
        s+=b[i][j]*np.conj(b[i][j])
print(s+1==1) # s~=0

#%%

# 2

f=lambda x: np.sin(2*np.pi*10*x)

t=np.arange(0,1,1/1000)
x=f(t)
# Figura 1
fig=plt.figure(21)
ax=plt.axes()
ax.plot(t,x)
ax.plot(t[127],x[127],color='r',marker='o')
plt.savefig(filnum+str(nr)+'.png')
plt.savefig(filnum+str(nr)+'.pdf')
nr=nr+1
# Figura 2
fig=plt.figure(22)
y=np.array([x[i]*cmath.exp(-2*np.pi*t[i]*1j) for i in range(len(t))])
ax=plt.axes()
ax.plot(y.real,y.imag)
ax.plot(y[127].real,y[127].imag,color='r',marker='o')
plt.savefig(filnum+str(nr)+'.png')
plt.savefig(filnum+str(nr)+'.pdf')
nr=nr+1
# z[omega]
for l in range(10): # omega=2*l
    fig=plt.figure(220+l)
    y=np.array([x[i]*cmath.exp(-2*np.pi*t[i]*1j*2*l) for i in range(len(t))])
    ax=plt.axes()
    ax.plot(y.real,y.imag)
    ax.plot(y[127].real,y[127].imag,color='r',marker='o')
    plt.savefig(filnum+str(nr)+'.png')
    plt.savefig(filnum+str(nr)+'.pdf')
    nr=nr+1
# Figurile 1 si 2 avand culoarea graficului o functie de destanta fata de origine
def plcol(x,y,ax):
    for i in range(len(x)):
        dist=int(100*np.sqrt(x[i]**2+y[i]**2))%100
        col='#00'+str(dist)+str(dist)
        col=col+'0'*(7-len(col))
        ax.plot(x[i],y[i],color=col,marker='o',markersize=2)
    ax.plot(x[127],y[127],color='r',marker='o')
t=np.arange(0,1,1/10000)
x=f(t)
fig=plt.figure(2101)
ax=plt.axes()
plcol(t,x,ax)
plt.savefig(filnum+str(nr)+'.png')
plt.savefig(filnum+str(nr)+'.pdf')
nr=nr+1
y=np.array([x[i]*cmath.exp(-2*np.pi*t[i]*1j) for i in range(len(t))])
fig=plt.figure(2102)
ax=plt.axes()
plcol(y.real,y.imag,ax)
plt.savefig(filnum+str(nr)+'.png')
plt.savefig(filnum+str(nr)+'.pdf')
nr=nr+1

#%%

# 3

f=lambda x: 5*np.sin(2*np.pi*10*x)+2*np.sin(2*np.pi*25*x)+2*np.sin(2*np.pi*40*x)

t=np.arange(0,1,1/200)
x=f(t)
fig=plt.figure(31)
ax=plt.axes()
ax.plot(t,x)
plt.savefig(filnum+str(nr)+'.png')
plt.savefig(filnum+str(nr)+'.pdf')
nr=nr+1
N=len(t)
# frecventa caracteristica omega:
# cmmmc(10,25,40) = 200 Hz
# perioada = 5 milisecunde
y=np.zeros(N).astype(complex)
for p in range(N):
    for q in range(N):
        y[p]+=x[q]*cmath.exp(-2*np.pi*1j*q*p/N)
fig=plt.figure(32)
ax=plt.axes()
ax.stem(t,y*np.conjugate(y))
plt.savefig(filnum+str(nr)+'.png')
plt.savefig(filnum+str(nr)+'.pdf')
nr=nr+1