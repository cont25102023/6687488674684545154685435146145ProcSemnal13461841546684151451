# Lab 5

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

# a
file=open('Train.csv','r')
lst=list(map(lambda x: x.split(','),file.readlines()))
# frecventa de esantionare este 1/1ora = 1/3600.
print('Frecventa de esantionare:',1/3600,'Hz.')

# b
# Timpul total (in ore) este cu 2 mai mic decat numarul de linii din fisier.
print('Interval acoperit:',len(lst)-2,'ore =',3600*(len(lst)-2),'secunde.')

# c
fresantion=1/3600
frmax=fresantion/2
print('Frecventa maxima posibila in semnal:',frmax,'Hz.')

# d
fr=np.fft.fft(list(map(lambda x: float(x[2]), lst[2:])))
n=len(fr)
frabs=np.abs(fr/n)
plt.figure(figsize=(20,20))
plt.plot(range(n),frabs)
frj=frabs[:n//2]
f=1/3600*np.linspace(0,n//2,n//2)/n
f1=1/3600*np.linspace(0,n//2,n//2)
plt.figure(figsize=(20,20))
plt.plot(range(n//2),frabs[:n//2])

# e
# componenta continua <=> media !=0.
val=list(map(lambda x: float(x[2]), lst[2:]))
medie=sum(val)/len(val) # len(val) = n
print('Componenta continua in semnal:',medie!=0)

# f
temp=list(frabs[:n//2])
index=[0,0,0,0]
for i in range(4):
    index[i]=temp.index(max(temp))
    temp[index[i]]=0
print('Fourrier maxim la frecventele:')
for i in range(4):
    print('    Frecventa',index[i]/3600,', |Fourrier| =',frabs[index[i]])