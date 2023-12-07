# Lab 8

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

# a
N=1000
t=np.array(list(range(N)))
f=lambda x: x**2/N
trend=f(t)
f1=lambda x: N*np.sin(2*np.pi*x*15/N)
f2=lambda x: 2*N*np.sin(2*np.pi*x*8/N)
sezon=f1(t)+f2(t)
varMici=np.random.normal(0,N/2,N)
serie=trend+sezon+varMici
plt.plot(t,serie)