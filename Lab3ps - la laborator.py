# Lab 3

# module
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as s1
import scipy.signal as s2
import sounddevice as s3
import cmath

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
fig,a=plt.subplots(n,figsize=(10,3*n))
for i in range(n):
    a[i].plot(t,fr[i].imag)
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

# fig1

t=np.arange(0,1,1/1000)
x=f(t)
fig=plt.figure(21)
ax=plt.axes()
ax.plot(t,x)
ax.plot(t[127],x[127],color='r',marker='o')
fig=plt.figure(22)
y=np.array([x[i]*cmath.exp(-t[i]*np.pi*2*1j) for i in range(len(t))])
ax=plt.axes()
ax.plot(y.real,y.imag)
ax.plot(y[127].real,y[127].imag,color='r',marker='o')



#%%

# 3


