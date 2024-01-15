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
import scipy
from sympy import Matrix,sin,exp,pi,lambdify,symbols

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
(U,J)=A.LDLdecomposition()
# J nu contine pe diagonala valorile proprii (este, totusi, o matrice diagonala)
# U si J pot fi calculate numai daca A este simetrica si pozitiv definita
# (este implicat algoritmul Cholesky, nu are nucio treaba cu valorile proprii)
# U este o matrice inferior triunghiulara, cu 1 pe diagonala principala;
# daca am calcula U si J din descompunerea Jordan (cu valori proprii),
# U transpus nu ar fi inversa lui U
L=np.copy(J).astype(float)
L[0][0]=np.sqrt(L[0][0])
L[1][1]=np.sqrt(L[1][1])
L=np.array(U).astype(float)@L
n=np.random.normal(0,1,(2,N))
y=L@n+np.transpose(np.full((N,2),mu))
# y=np.transpose(np.random.multivariate_normal(mu,np.array(A).astype(float),(N)))
# ar genera y direct
print(sum(np.transpose(y))/N-mu)

fig,ax=plt.subplots(1,2,num='1',figsize=(20,10),clear=True)
ax[0].hist(x,25)
ax[0].set_title('1-d')
ax[1].plot(y[0],y[1],linestyle='none',marker='o',markersize=1)
ax[1].plot(*mu,marker='o',markersize=4)
# contur
m0=min(y[0])
M0=max(y[0])
d0=(M0-m0)/1000
m1=min(y[1])
M1=max(y[1])
d1=(M1-m1)/1000
xx,yy=np.mgrid[m0:M0:d0,m1:M1:d1] # puncte pe grid
grid=np.dstack((xx,yy)) # grid; dimensiune *np.shape(xx)+(2,)
dens=scipy.stats.multivariate_normal(mu,np.array(A).astype(float))
z=dens.pdf(grid)
T=z[150][500]
ax[1].contourf(xx,yy,z,levels=[0.95*T,T],cmap='viridis')
ax[1].set_title('2-d')

#%%

# 2 - conform curs
# cine sunt x si y ? Nu scrie in curs.

print('\n2 - conform curs:')
N=250
x=np.linspace(-1,1,N)
fig,ax=plt.subplots(3,2,num='2 - conform curs',figsize=(20,9),clear=True)

# liniar
y=np.random.normal(0,abs(x))
ax[0][0].plot(x,y,marker='o',markersize=1.6,lw=0.5)
ax[0][0].set_title('Liniar')

# miscare browniana
y=np.cumsum(np.random.normal(0,abs(x)))
ax[0][1].plot(x,y,marker='o',markersize=1.6,lw=0.5)
ax[0][1].set_title('Miscare Browniana')

# exponentiala patrata
a=3
y=np.random.normal(0,np.exp(-a*x**2)) # conform curs
ax[1][0].plot(x,y,marker='o',markersize=1.6,lw=0.5)
ax[1][0].set_title('Exponentiala patrata')

# Ornstein-Uhlenbeck
a=3
y=np.random.normal(0,np.exp(-a*abs(x))) # conform curs
ax[1][1].plot(x,y,marker='o',markersize=1.6,lw=0.5)
ax[1][1].set_title('Ornstein-Uhlenbeck')
# Ornstein-Uhlenbeck 2
y=np.zeros(N)
theta,mu,sigma=0.25,0,0.25
for t in range(1, N):
    y[t]=y[t-1]+theta*(mu-y[t-1])+sigma*np.random.normal(0,1)
fig1,ax1=plt.subplots(num='Ornstein-Uhlenbeck',figsize=(10,5),clear=True)
ax1.plot(x,y,marker='o',markersize=1.6,lw=0.5)

# periodic
a=3
b=2
y=np.random.normal(0,np.exp(-a*np.sin(b*np.pi*x)**2)) # conform curs
ax[2][0].plot(x,y,marker='o',markersize=1.6,lw=0.5)
ax[2][0].set_title('Periodic')

# simetric
# conform curs:
# pentru ca nu se stie cine sunt z si y, pres x= axa Ox si y=0
# in acest caz, va da similar cu exponentiala patrata (aceeasi repartitie)
a=3
y=np.random.normal(0,np.exp(-a*min(abs(x),abs(x),key=lambda x:np.sum(x))**2))
ax[2][1].plot(x,y,marker='o',markersize=1.6,lw=0.5)
ax[2][1].set_title('Simetric')

fig.tight_layout()

#%%

# 2 - de pe internet
# cautare pe internet => alte variante

print('\n2 - cautare pe internet:')
N=250
fig,ax=plt.subplots(3,2,num='2 - de pe internet',figsize=(20,9),clear=True)

# liniar
liniar=lambda x1,x2,par: par[0]+par[1]*np.dot(x1,x2.T)
def PG(X,kernel,par,nr):
    K=kernel(X,X,par)
    samples=np.random.multivariate_normal(np.zeros(X.shape[0]),K,nr)
    return samples
par=[1.0,1.0]
X=np.linspace(-1,1,N).reshape(-1,1)
samples=PG(X,liniar,par,5)
for i in samples:
    ax[0][0].plot(X,i)
ax[0][0].set_title('Liniar')

# miscare browniana
t=np.linspace(1,N+1,N)
K=np.minimum.outer(t,t) # matricea de covarianta
L=np.linalg.cholesky(K)
r=np.random.normal(0,1,N)
B=np.dot(L,r) # B reprezinta un rezultat dintr-o normala N-dimensionala
# cu medie 0 si matrice de covarianta K
ax[0][1].plot(t,B,marker='o',markersize=1.6,lw=0.5)
ax[0][1].set_title('Miscare Browniana')

# exponentiala patrata
def PG(t,a):
    K=np.exp(-a*np.abs(t[:,None]-t[None,:])**2)
    L=np.linalg.cholesky(K)
    x=np.dot(L,np.random.normal(size=len(t)))
    return x
a=0.1
t=np.linspace(1,N+1,N)
process=PG(t,a)
ax[1][0].plot(t,process,marker='o',markersize=1.6,lw=0.5)
ax[1][0].set_title('Exponentiala patrata')

# Ornstein-Uhlenbeck
def PG(t,a):
    K=np.exp(-a*np.abs(t[:,None]-t[None,:]))
    L=np.linalg.cholesky(K)
    x=np.dot(L,np.random.normal(size=len(t)))
    return x
a=0.1
t=np.linspace(1,N+1,N)
process=PG(t,a)
ax[1][1].plot(t,process,marker='o',markersize=1.6,lw=0.5)
ax[1][1].set_title('Ornstein-Uhlenbeck')

# periodic
x,y,a,b=symbols('x y a b')
kernel=exp(-a*sin(b*pi*(x-y))**2)
kernel_func=lambdify((x,y,a,b),kernel,'numpy')
def PG(x,a,b,kernel_func):
    K=kernel_func(x,x[:,None],a,b)
    return np.random.multivariate_normal(np.zeros(len(x)),K) # sau cholesky, ca mai sus
a=1
b=5
x=np.linspace(-1,1,N)
y=PG(x,a,b,kernel_func)
ax[2][0].plot(x,y,marker='o',markersize=1.6,lw=0.5)
ax[2][0].set_title('Periodic')

# simetric
kernel=lambda x,y,a: np.exp(-a*np.minimum(np.abs(x-y),np.abs(x+y))**2)
def PG(x,y,a):
    K=kernel(x[:,None],y,a)
    return np.random.multivariate_normal(np.zeros(len(x)),K)
x=np.linspace(-5,5,N)
y=PG(x,x,1)
ax[2][1].plot(x,y,marker='o',markersize=1.6,lw=0.5)
ax[2][1].set_title('Simetric')

fig.tight_layout()

#%%

# 3

print('\n3:')
file=open('co2_daily_mlo.csv','r')
l=list(filter(lambda x:len(x)>0 and x[0] in '0123456789' and len(x.split(','))==5\
,file.readlines()))
n=len(l)
l=np.transpose(list(map(lambda x: [float(i) for i in x.split(',')],l)))

# a
lunar=[]
s=0
nr=0
last=l[1][0]
for i in range(n):
    s+=l[4][i]
    nr+=1
    if i<n-1 and l[1][i+1]!=l[1][i]:
        lunar.append(s/nr)
        s=0
        nr=0
lunar.append(s/nr)
fig,ax=plt.subplots(num='3 a',figsize=(20,10),clear=True)
ax.plot(lunar)

# b
# presupunem ca trendul este de forma y=ax+b; regresie liniara a tuturor punctelor
y=np.array(lunar)
m=len(y)
c=np.array(list(range(m)))
A=np.ones((m,2))
A[:,0]=c
x=np.linalg.solve(np.transpose(A)@A,np.transpose(A)@y)
lunar2=y-A@x
fig,ax=plt.subplots(num='3 b',figsize=(20,10),clear=True)
ax.plot(lunar2)

# c
# conform cursului, prezicerea este suma a doi vectori aleatori (Z si epsilon),
# prin urmare, este complet aleatoare, nu va avea nicio legatura cu seria,
# inafara de faptul ca initial are media egala cu serie[-nr-1];
# nu ma astept la comportament periodic, ci mai degraba la o crestere liniara
# a marginii superioare si la o scadere liniara a marginii inferioare
nr=12
A=lunar2[:-nr]
B=lunar2[-nr:]
pmin=np.zeros(nr)
pmax=np.zeros(nr)
nt=50 # numarul de teste la fiecare pas
# hardcodez toti parametrii a caror provenienta nu a fost specificata in curs
mu1=A[-1]
mu2=A[-1]
mub1=A[-1]
mub2=A[-1]
C=np.array([[0.01,0.001],[0.001,0.01]])
for i in range(nr):
    D=abs(C[0][0]-C[0][1]/C[1][1]*C[1][0])
    mub1=mu1+C[0][1]/C[1][1]*(mu1-mub1)
    mub2=mu2+C[0][1]/C[1][1]*(mu2-mub2)
    C[0][0]=D
    test=np.random.normal(mu1,np.sqrt(D),nt)+np.random.normal(mu2,np.sqrt(D),nt)
    mu1=pmin[i]=min(np.random.normal(mu1,np.sqrt(D),nt))
    mu2=pmax[i]=max(np.random.normal(mu2,np.sqrt(D),nt))
fig,ax=plt.subplots(num='3 c',figsize=(20,10),clear=True)
ax.plot(lunar2,color='blue')
ax.plot(c[-nr:],pmin,color='green')
ax.plot(c[-nr:],pmax,color='red')