# module
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as s1
import scipy.signal as s2
#import sounddevice as s3


# 1.
def f(x): return np.sin(2*np.pi*5*x)
def g(x): return np.cos(2*np.pi*5*x-np.pi/2)

x = np.linspace(0, 1, 100)
fig, axes = plt.subplots(2)

axes[0].plot(x, f(x))
axes[0].set_title('f')

axes[1].plot(x, g(x))
axes[1].set_title('g')

plt.show()


# 2.
def f(x): return np.sin(2*np.pi*5*x)
def nr(x): return sum([i**2 for i in x])

t = np.linspace(0, 10, 1000)
x=f(t)
z=np.random.normal(0,1,len(t))
gam=np.sqrt(nr(x)/nr(z))
val=[0.1,1,10,100]
fig, axes = plt.subplots(len(val))
for i in range(len(val)):
    z=gam/val[i]*z
    axes[i].plot(t,x+z)
    axes[i].set_title('lambda='+str(val[i]))
plt.show()


# 4.
def f(x): return np.sin(2*np.pi*5*x)
def g(x): return -1+2*(5*x-int(5*x))

x = np.linspace(0, 1, 100)
fig, axes = plt.subplots(2)

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