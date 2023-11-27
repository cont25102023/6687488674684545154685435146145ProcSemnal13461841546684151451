# Lab 7

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
# Dimensiunea imaginii: 512*512 px
m=512
n=512
#%%
# Primul punct
x=np.zeros((m,n))
for i in range(m):
    for j in range(n):
        x[i][j]=math.sin(2*math.pi*i+3*math.pi*j)
plt.imshow(x)
plt.show() # Nimic de observat
plt.imshow(20*np.log10(np.abs(np.fft.fft2(x))))
plt.show() # Linii verticale, orizontale si oblice
#%%
# Punctul al doilea
x=np.zeros((m,n))
for i in range(m):
    for j in range(n):
        x[i][j]=math.sin(4*math.pi*i)+math.cos(6*math.pi*j)
        # matrice plina cu 1 pentru ca sin(4*math.pi*n1)=0, cos(6*math.pi*n2)=1
        # pentru orice n1, n2 numere naturale
plt.imshow(x)
plt.show()
plt.imshow(abs(np.fft.fft2(x)))
# np.fft.fft2(x) contine in pozitia 0,0 suma elementelor din x, iar in rest 0
# pentru ca suma radacinilor de ordin n ale unitatii este 0 pentru n>=2,
# deci abs(np.fft.fft2(x)) nu poate fi afisat pe scala logaritmica.
plt.show()
#%%
# Punctul al treilea
Y=np.zeros((m,n)) # Y este spectrul unei imagini
Y[0][5]=1
Y[0][-5]=1
plt.imshow(abs(np.fft.ifft2(Y))) # Imaginea ; obs: benzi verticale
# motiv: spectrul contine pe orizontala o singura frecventa (frecventa 5)
# si pe verticala frecventa 0
plt.show()
plt.imshow(Y)
plt.show()
#%%
# Punctul al patrulea
Y=np.zeros((m,n)) # Y este spectrul unei imagini
Y[5][0]=1
Y[-5][0]=1
plt.imshow(abs(np.fft.ifft2(Y))) # Imaginea ; obs: benzi orizontale
# motiv: spectrul contine pe verticala o singura frecventa (frecventa 5)
# si pe orizontala frecventa 0
plt.show()
plt.imshow(Y)
plt.show()
#%%
# Punctul al cicilea
Y=np.zeros((m,n)) # Y este spectrul unei imagini
Y[5][5]=1
Y[-5][-5]=1
plt.imshow(abs(np.fft.ifft2(Y))) # Imaginea ; obs: benzi oblice paralele cu diagonala secundara
# motiv: spectrul contine si pe verticala si pe orizontala aceeasi frecventa (5)
plt.show()
plt.imshow(Y)
plt.show()

#%%

# 2
X=misc.face(gray=True)
plt.imshow(X,cmap=plt.cm.gray)
plt.title('Original')
plt.show()
Y=np.fft.fft2(X)
m,n=np.shape(X)
freq_cutoff=64
Y_cutoff=np.copy(Y)
for i in range(m):
    for j in range(n):
        if min(i,m-i)+min(j,n-j)>2*freq_cutoff:
            Y_cutoff[i][j]=0
X_cutoff=np.fft.ifft2(Y_cutoff)
X_cutoff=np.abs(X_cutoff)
plt.imshow(X_cutoff,cmap=plt.cm.gray)
plt.title('Trunchiat1')
plt.show()
# Stocarea unei forme comprimate
comprimat={
    'parametri':np.array([m,n,freq_cutoff]),
    'medie':Y[0][0],
    'prima linie':Y[0,1:2*freq_cutoff+1],
    'prima coloana':Y[1:2*freq_cutoff+1,0],
    'stanga sus':np.zeros(freq_cutoff*(2*freq_cutoff-1)).astype(complex),
    'dreapta sus':np.zeros(freq_cutoff*(2*freq_cutoff-1)).astype(complex)}
k=0
for i in range(1,2*freq_cutoff):
    for j in range(1,2*freq_cutoff-i+1):
        comprimat['stanga sus'][k]=Y[i][j]
        comprimat['dreapta sus'][k]=Y[i][n-j]
        k=k+1
# Creare imagine din forma comprimata
m1,m2,u=comprimat['parametri']
Z=np.zeros((m1,m2)).astype(complex)
Z[0][0]=comprimat['medie']
Z[0,1:2*u+1]=np.copy(comprimat['prima linie'])
Z[0,m2-1:m2-2*u-1:-1]=np.conj(comprimat['prima linie'])
Z[1:2*u+1,0]=np.copy(comprimat['prima coloana'])
Z[m1-1:m1-2*u-1:-1,0]=np.conj(comprimat['prima coloana'])
k=0
for i in range(1,2*u):
    for j in range(1,2*u-i+1):
        Z[i][j]=comprimat['stanga sus'][k]
        Z[m1-i][m2-j]=np.conj(comprimat['stanga sus'][k])
        Z[i][m2-j]=comprimat['dreapta sus'][k]
        Z[m1-i][j]=np.conj(comprimat['dreapta sus'][k])
        k=k+1
X1=np.fft.ifft2(Z)
plt.imshow(abs(X1),cmap=plt.cm.gray)
plt.title('Trunchiat2')
plt.show()
# Calcul SNR
inalt=X-X1
t1=0
t2=0
for i in range(m):
    for j in range(n):
        t1+=abs(X1[i][j])**2
        t2+=abs(inalt[i][j])**2
print('SNR compresie =',t1/t2)

#%%

# 3
pixel_noise=200
noise=np.random.randint(-pixel_noise,pixel_noise+1,size=X.shape)
X_noisy=X+noise
plt.imshow(X,cmap=plt.cm.gray)
plt.title('Original')
plt.show()
plt.imshow(X_noisy,cmap=plt.cm.gray)
plt.title('Noisy')
plt.show()
# Eliminare frecvente inalte
Y=np.fft.fft2(X_noisy)
m,n=np.shape(X_noisy)
freq_cutoff=80
Y_cutoff=np.copy(Y)
for i in range(m):
    for j in range(n):
        if min(i,m-i)+min(j,n-j)>2*freq_cutoff:
            Y_cutoff[i][j]=0
X_cutoff=np.fft.ifft2(Y_cutoff)
plt.imshow(abs(X_cutoff),cmap=plt.cm.gray)
plt.title('Frecvente joase')
plt.show()
# Calcul SNR
t1=0.0
t2=0.0
t3=0
t4=0
Y=np.fft.fft2(X)
for i in range(m):
    for j in range(n):
        if min(i,m-i)+min(j,n-j)>2*freq_cutoff:
            Y[i][j]=0
X1=np.fft.ifft2(Y)
noise1=X_cutoff-X1
for i in range(m):
    for j in range(n):
        t1+=int(X[i][j])**2
        t2+=noise[i][j]**2
        t3+=abs(X1[i][j])**2
        t4+=abs(noise1[i][j])**2
print('SNR inainte =',t1/t2)
print('SNR dupa =',t3/t4)

#%%

# 4
rate,x=s1.read('sunet.wav')
rate1,x1=s1.read('drums.wav') #modificat cu Audacity pentru a fi compatibil cu Python
n=len(x)
y=x[(n+1)//2:n//46*33]
z=x1[(n+1)//2:n//46*33]
n=len(y)
s3.play(y-z,rate)