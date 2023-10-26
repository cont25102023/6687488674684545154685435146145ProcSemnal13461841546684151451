# module
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as s1
import scipy.signal as s2
import sounddevice as s3


#%%
# 1.
def f(x): return np.sin(2*np.pi*5*x)
def g(x): return np.cos(2*np.pi*5*x-np.pi/2)

x=np.linspace(0,1,100)
fig,axes = plt.subplots(2)

axes[0].plot(x,f(x))
axes[0].set_title('f')

axes[1].plot(x,g(x))
axes[1].set_title('g')

plt.show()


#%%
# 2.
def f(x,faz): return np.sin(2*np.pi*5*x+faz)
def nr(x): return sum([i**2 for i in x])

t=np.linspace(0,1,1000)
for j in range(4):
    x=f(t,np.pi*j/3)
    z=np.random.normal(0,1,len(t))
    gam=np.sqrt(nr(x)/nr(z))
    val=[0.1,1,10,100]
    fig,axes = plt.subplots(len(val),figsize=(20,20))
    for i in range(len(val)):
        z=gam/val[i]*z
        axes[i].plot(t,x+z)
        axes[i].set_title('lambda='+str(val[i]))
    plt.show()


#%%
# 3.
#%%
# 2a
t=np.sin(400*np.pi*2*np.arange(0,1600/400,1/1600))
s3.play(t,5000)
#%%
# 2b
t=np.sin(800*np.pi*2*np.arange(0,3,1/(3*800)))
s3.play(t,5000)
#%%
# 2c
t=list(map(lambda x:-1+2*math.modf(240*x)[0],np.arange(0,0.2,1/24000)))
s3.play(t,5000)
#%%
# 2d
t=np.sign(np.sin(300*np.pi*2*np.arange(0,0.2,1/30000)))
s3.play(t,5000)
#%%
# Salvarea lui 2a.
t=np.sin(400*np.pi*2*np.arange(0,1600/400,1/1600))
rate=int(10e5)
s1.write('Lab2psEx3semnal.wav',rate,t)
r,x=s1.read('Lab2psEx3semnal.wav')
s3.play(x,5000)


#%%
# 4.
def f(x): return np.sin(2*np.pi*5*x)
def g(x): return -1+2*(5*x-int(5*x))

x = np.linspace(0,1,100)
fig,axes=plt.subplots(2,figsize=(20,20))

z=[sum([f(x[i]) for i in range(j+1)])for j in range(len(x))]
axes[0].plot(x,f(x))
axes[0].plot(x,z)
axes[0].set_title('f')

y=list(map(lambda i:g(i),x))
z=[sum([g(x[i]) for i in range(j+1)])for j in range(len(x))]
axes[1].plot(x,y)
axes[1].plot(x,z)
axes[1].set_title('g')

plt.show()


#%%
# 5.
def f(x): return np.sin(2*np.pi*10*x)
def g(x): return np.sin(2*np.pi*25*x)
t=np.linspace(0,50,5000)
x=list(f(t))+list(g(t))
s3.play(x,5000)
# Observatie: un sunet mai jos urmat de un sunet mai inalt.


#%%
# 6.
fs=20
def f(x,fr): return np.sin(2*np.pi*fr*x)
x=np.linspace(0,1,1000)
fig,axes=plt.subplots(3,figsize=(20,20))
# a
axes[0].plot(x,f(x,fs/2))
axes[0].set_title('fs/2')
# b
axes[1].plot(x,f(x,fs/4))
axes[1].set_title('fs/4')
# c
axes[2].plot(x,f(x,0))
axes[1].set_title('0')
plt.show()
# Observatie: b) are frecventa de doua ori mai mica decat a), iar c) e constant.


#%%
# 7.
# a
t1=np.arange(0,0.2,1/1000)
def f(x): return np.sin(2*np.pi*20*x)
t2=[t1[i] for i in range(0,len(t1),4)]
x1=f(t1)
x2=[x1[i] for i in range(0,len(x1),4)]
fig=plt.figure(figsize=(20,5))
ax=plt.axes()
ax.plot(t1,x1,color='g')
ax.plot(t2,x2,color='r')
# Observatie: Semnalul al doilea este mai neregulat decat primul.
# b
t3=[t2[i] for i in range(0,len(t2),4)]
x3=[x2[i] for i in range(0,len(x2),4)]
ax.plot(t3,x3,color='magenta')
# Observatie: Semnalul de la b) este si mai neregulat decat anterioarele.


#%%
# 8.
def Pade(x): return (x-7*x**3/60)/(1+x**2/20)
x=np.arange(-np.pi/2,np.pi/2,1/1000)
fig,axes=plt.subplots(4,figsize=(20,20))
axes[0].plot(x,np.sin(x))
axes[0].plot(x,x)
axes[0].set_title('f(x)=x')
axes[1].plot(x,np.abs(x-np.sin(x)))
axes[1].set_title('|f(x)=x-sin(x)|')
axes[2].plot(x,np.sin(x))
axes[2].plot(x,Pade(x))
axes[2].set_title('f(x)=Pade(x)')
axes[3].plot(x,np.abs(Pade(x)-np.sin(x)))
axes[3].set_title('f(x)=|Pade(x)-sin(x)|')
plt.show()