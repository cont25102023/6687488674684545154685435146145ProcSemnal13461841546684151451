# Lab 10

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
from sympy import Matrix

#%%

# 1

N=10000
print('\n1:')

m=5
var=4
x=np.random.normal(0,1,N)*np.sqrt(var)+m
print(sum(x)/N-m)

mu=np.array([1,1]).astype(float) # media
A=Matrix([[1,3/5],[3/5,2]]) # matricea de covarianta
(U,J)=A.jordan_form() # (U,J), J e forma normala Jordan a lui A, A = UJU^-1
L=np.copy(J).astype(float)
L[0][0]=np.sqrt(L[0][0])
L[1][1]=np.sqrt(L[1][1])
L=np.array(U)@L
n=np.random.normal(0,1,(2,N))
y=L@n+np.transpose(np.full((N,2),mu))
print(sum(np.transpose(y))/N-mu)

fig,ax=plt.subplots(1,2,num=1,figsize=(20,20),clear=True)
ax[0].hist(x,25)
ax[0].set_title('1-d')
ax[1].plot(y[0],y[1],linestyle='none',marker='o',markersize=1)
ax[1].plot(*mu,marker='o',markersize=4) # ax[1].contour(...)
ax[1].set_title('2-d')

#%%

# 2



#%%

# 3

file=open('co2_mm_mlo.csv','r')
l=file.readlines()
