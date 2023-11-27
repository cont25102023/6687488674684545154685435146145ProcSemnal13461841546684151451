# Lab 5

# module
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as s1
import scipy.signal as s2
import cmath
import time
import datetime
import pickle # pentru salvare variabile pe disc
import os
import shutil

#%%

# 1

# a
file=open('..\Train.csv','r')
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
# fr=np.fft.fft(list(map(lambda x: x[2], np.genfromtxt('..\Train.csv',delimiter=',')[2:])))
fr=np.fft.fft(list(map(lambda x: float(x[2]), lst[1:])))
n=len(fr)
frabs=np.abs(fr/n)
plt.figure(figsize=(20,20))
plt.plot(range(n),frabs)
frj=frabs[:n//2]
f=1/3600*np.linspace(0,n//2,n//2)/n
f1=1/3600*np.linspace(0,n//2,n//2)
plt.figure(figsize=(20,20))
plt.plot(range(n//2),frabs[:n//2]/3600)

# e
# componenta continua <=> media !=0.
val=list(map(lambda x: float(x[2]), lst[1:]))
medie=sum(val)/len(val) # len(val) == n
print('Componenta continua in semnal:',medie!=0)
# eliminare componenta continua:
val0=list(map(lambda x: x-medie, val))

# f
temp=list(frabs[:n//2])
index=[0,0,0,0]
for i in range(4):
    index[i]=temp.index(max(temp))
    temp[index[i]]=0
print('Fourrier maxim la frecventele:')
for i in range(4):
    print('    Frecventa',index[i]/3600,', |Fourrier| =',frabs[index[i]])
print('Frecventa 0 reprezinta media.')

# g
es=1001
while datetime.datetime.strptime(lst[es][1],'%d-%m-%Y %H:%M').weekday()!=0: # luni
    es=es+1
fin=es
d0=datetime.datetime.strptime(lst[es][1],'%d-%m-%Y %H:%M')
d1=datetime.datetime.strptime(lst[fin][1],'%d-%m-%Y %H:%M')
while 1:
    fin=fin+1
    d1=datetime.datetime.strptime(lst[fin][1],'%d-%m-%Y %H:%M')
    if d1.month==1+d0.month or d1.month==d0.month-11: # diferenta de o luna
        if d1.day==d0.day or (d1+datetime.timedelta(days=1)).day==1: # aceeasi zi a lunii
            if d1.hour>=d0.hour: # aceeasi ora
                break
plt.figure(figsize=(20,20))
plt.plot(range(es-1,fin),val[es-1:fin]) # val calculat la pct. e

# h
'''
Se partitioneaza semnalul in N bucati de lungimi egale.
Pentru fiecare bucata se calculeaza media, m=[.,.,.,...], len(m) = N
Se calculeaza polinomul de interpolare Lagrange pentru 
range(0,n,N) si m.
Se evalueaza polinomul in -1, -2, ... pana cand:
Cazul 1: se obtine o valoare negativa, caz in care pentru acel
indice se calculeaza data, sau
Cazul 2: valorile incep sa creasca, fara sa fi devenit negative,
caz in care metoda nu aproximeaza corect data de inceput.
Neajunsuri: se poate intampla cazul 2.
Acuratetea depinde de N.
'''

# i
frecmax=50
filtrat=list(map(lambda i: fr[i] if i<frecmax or n-i<frecmax else 0,range(n)))
x=np.fft.ifft(filtrat)
plt.figure(figsize=(20,20),num='Semnal original si semnal filtrat (frecvente joase)')
plt.plot(range(n),x,linewidth=2.5,color='b',label=' semnal filtrat')
plt.plot(range(n),val,linewidth=0.16,color='r',label='semnal original')
plt.legend(loc='best')