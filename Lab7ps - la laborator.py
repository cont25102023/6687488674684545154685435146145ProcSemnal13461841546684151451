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

#%%

from scipy import misc, ndimage
X = misc.face(gray=True)
plt.imshow(X, cmap=plt.cm.gray)
plt.show()

m,n=np.shape(X)
x=np.zeros((m,n))
for i in range(m):
    for j in range(n):
        