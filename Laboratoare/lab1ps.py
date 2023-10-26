# Module
import numpy as np
import matplotlib.pyplot as plt
import math

#%%
# 1 a
t=np.arange(0,0.03,0.0005)
fig,axs=plt.subplots(1,1,figsize=(8,6))
axs.plot(t,0*t)
plt.tight_layout()
plt.show()

#%%
# 1 b
t=np.arange(0,0.03,0.0005)
fig,axs=plt.subplots(3,1)
axs[0].plot(t,np.cos(520*np.pi*t+np.pi/3))
axs[0].set_title('x(t)')
axs[1].plot(t, np.cos(280*np.pi*t-np.pi/3))
axs[1].set_title('y(t)')
axs[2].plot(t,np.cos(120*np.pi*t+np.pi/3))
axs[2].set_title('z(t)')
plt.tight_layout()
plt.show()

#%%
# 1 c
n=np.arange(0,0.03,1/200)
fig,axs=plt.subplots(3,1)
axs[0].plot(n,np.cos(520*np.pi*n+np.pi/3))
axs[0].set_title('x(n)')
axs[1].plot(n,np.cos(280*np.pi*n-np.pi/3))
axs[1].set_title('y(n)')
axs[2].plot(n,np.cos(120*np.pi*n+np.pi/3))
axs[2].set_title('z(n)')
plt.tight_layout()
plt.show()

#%%
# 2 a
t=np.arange(0,1600/400,1/1600)
fig,axs=plt.subplots(1,1,figsize=(100,5))
axs.plot(t,np.sin(400*np.pi*2*t))
plt.tight_layout()
plt.show()

#%%
# 2 b
t=np.arange(0,3,1/(3*800))
fig,axs=plt.subplots(1,1,figsize=(100,5))
axs.plot(t,np.sin(800*np.pi*2*t))
plt.tight_layout()
plt.show()

#%%
# 2 c
t=np.arange(0,0.02,1/24000)
fig,axs=plt.subplots(1,1)
v=list(map(lambda x:-1+2*math.modf(240*x)[0],t))
axs.plot(t,v)
plt.tight_layout()
plt.show()

#%%
# 2 d
t=np.arange(0,0.02,1/30000)
fig,axs=plt.subplots(1,1)
axs.plot(t,np.sign(np.sin(300*np.pi*2*t)))
plt.tight_layout()
plt.show()

#%%
# 2 e
t=np.random.rand(128,128)
plt.imshow(t)

#%%
# 2 f
t1=np.zeros((128,128))
plt.imshow(t1)
t2=np.ones((128,128))
plt.imshow(t2)
t3=np.zeros((128,128))
for i in range(128):
    for j in range(128):
        t3[i][j]=128*i+j
plt.imshow(t3)

#%%
# 3 a
# interval = 1/2000 = 0.0005 secunde
print(1/2000)
# 3 b
# dimensiune = 2000 * 4 * 3600 / 8 = 3600000 octeti
print(2000*4*3600/8)