# Tema 1

# module
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as s1
import scipy.signal as s2
import cmath
import time
import pickle # pentru salvare variabile pe disc
import os
import shutil
from scipy import misc,ndimage
from scipy.fft import dctn,idctn
import cv2
import warnings
import huffman

warnings.filterwarnings('ignore')

#%%
# Compresie huffman

def hcod(X):
    sh=np.shape(X)
    X0=X.flatten()
    sample=X0[0] # pentru a retine tipul
    X0=X0.view(np.uint8) # reinterpreteaza fiecare float64 ca 8 uint8
    sw=list(zip(*np.unique(X0,return_counts=True)))
    tabel=huffman.codebook(sw)
    X0=np.array(list(''.join([tabel[i] for i in X0]))).astype(np.uint8)
    X0=np.packbits(X0)
    return sw,X0,sh,sample

def hdec(sw,X0,sh,sample):
    rad=huffman.huffman.Tree(sw).root
    X=np.unpackbits(X0)
    l=[]
    nod=rad
    for i in X:
        if i==0:
            nod=nod.left
        else:
            nod=nod.right
        if type(nod)==huffman.huffman.Leaf:
            l.append(nod.symbol)
            nod=rad
    l=np.array(l).astype(np.uint8)[:np.prod(sh)*len(np.array([sample]).view(np.uint8))].view(type(sample)).reshape(sh)
    return l

def toImg(Y0,m,n):
    d=np.shape(Y0)
    if len(d)==3:
        Y0=np.array([Y0])
        d=np.shape(Y0)
    elif len(d)!=4:
        return 0
    X0=np.zeros_like(Y0)
    for i in range(d[0]):
        for j in range(d[3]):
            for p in range(0,d[1],8):
                for q in range(0,d[2],8):
                    X0[i,p:p+8,q:q+8,j]=idctn(Y0[i,p:p+8,q:q+8,j]) # des-dct
    X0=X0[:,:m,:n,:]
    X=np.zeros_like(X0)
    for i in range(d[0]):
        X[i]=cv2.cvtColor(X0[i].astype(np.float32)/256,cv2.COLOR_YCR_CB2RGB).astype(float)*256
    return X

#%%
# 1
# JPEG pe o imagine grayscale. Exemplificere pe scipy.misc.ascent.

Q_jpeg=[[16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 28, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]]

X=misc.ascent()

def JPEG1(X,Q):
    m,n=np.shape(X)
    X1=np.zeros((int(8*np.ceil(m/8)),int(8*np.ceil(n/8))))
    X0=np.zeros_like(X1)
    X0[:m,:n]=np.copy(X)
    Y1=np.zeros_like(X1)
    Y0=np.zeros_like(X1)
    for i in range(0,m,8):
        for j in range(0,n,8):
            X1[i:i+8,j:j+8]=dctn(X0[i:i+8,j:j+8])
            Y1[i:i+8,j:j+8]=Q*np.round(X1[i:i+8,j:j+8]/Q)
            Y0[i:i+8,j:j+8]=idctn(Y1[i:i+8,j:j+8])
    nz=np.count_nonzero(X1)
    nz_jpeg=np.count_nonzero(Y1)
    return Y1,Y0[:m,:n],nz,nz_jpeg

Y0,Y,nz,nz_jpeg=JPEG1(X,Q_jpeg)
fig,ax=plt.subplots(1,2,num=1,figsize=(20,10),clear=True)
ax[0].imshow(X,cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[1].imshow(Y,cmap=plt.cm.gray)
ax[1].set_title('JPEG')

print('\n1:\nComponente în frecvență: '+str(nz)+
      '\nComponente în frecvență după cuantizare: '+str(nz_jpeg)+'\n')

#%%
# 2
# Jpeg pe o imagine color rgb trecand prin ycbcr. Exemplificare pe scipy.misc.face.

X=misc.face()

def JPEG2(X,Q):
    A=np.array([[0.2126,0.7152,0.0722],
                [-0.1146,-0.3854,0.5],
                [0.5,-0.4542,-0.0458]]) # rgb to ycbcr
    B=np.array([[1,0,1.5748],
                [1,-0.1873,-0.4681],
                [1,1.8556,0]]) # ycbcr to rgb
    try:
        m,n,p=np.shape(X)
        if p!=3:
            return 0,0,0,0
        temp0=np.zeros((m,n,p))
        for i in range(m):
            for j in range(n):
                temp0[i][j]=A@X[i][j] # temp0 este de tip ycbcr
        Y0=np.zeros((int(8*np.ceil(m/8)),int(8*np.ceil(n/8)),p))
        Y=np.zeros((m,n,p))
        temp1=np.zeros((m,n,p))
        nz=0
        nz_jpeg=0
        for i in range(p):
            Y0[...,i],temp1[...,i],nr1,nr2=JPEG1(temp0[...,i],Q)
            nz+=nr1 # temp1 este in format ycbcr
            nz_jpeg+=nr2 # nz si nz_jpeg numara frecventele nenule din versiunile ycbcr
        for i in range(m):
            for j in range(n):
                Y[i][j]=B@temp1[i][j] # Y este de tip rgb, comprimat jpeg
        return Y0,Y,nz,nz_jpeg
    except:
        return -1,-1,-1,-1 # daca np.shape(X) nu are exact 3 componente

Y0,Y,nz,nz_jpeg=JPEG2(X,Q_jpeg) # Y0 contine blocuri dct, Y este imagine rgb
fig,ax=plt.subplots(1,2,num=2,figsize=(20,10),clear=True)
ax[0].imshow(X)
ax[0].set_title('Original')
ax[1].imshow(Y.astype(np.uint8)) # Y este de tip float64
ax[1].set_title('JPEG')

print('\n2:\nComponente în frecvență: '+str(nz)+
      '\nComponente în frecvență după cuantizare: '+str(nz_jpeg)+'\n')

#%%
# 3
# Compresie JPEG pana la un prag MSE impus.
# Pe tot parcursul algoritmului, MSE va fi calculat conform formulei din cursul 8.

prag=0.001 # prag MSE de 0.1%

def jpgaux3(X,Q,m,n):
    temp0=cv2.cvtColor(X.astype(np.float32)/256,cv2.COLOR_RGB2YCR_CB)*256
    Y0=np.zeros((int(8*np.ceil(m/8)),int(8*np.ceil(n/8)),3))
    temp1=np.zeros((m,n,3))
    nz=0
    nz_jpeg=0
    for i in range(3):
        Y0[...,i],temp1[...,i],nr1,nr2=JPEG1(temp0[...,i],Q)
        nz+=nr1 # temp1 este in format ycbcr / ycrcb
        nz_jpeg+=nr2 # nz si nz_jpeg numara frecventele nenule din versiunile ycbcr
    Y=cv2.cvtColor(temp1.astype(np.float32)/256,cv2.COLOR_YCR_CB2RGB).astype(float)*256
    return Y0,Y,nz,nz_jpeg

def JPEG3(X0,Q,prag):
    d=np.shape(X0)
    if len(d)!=3:
        return -1,-1,-1,-1,-1 # daca np.shape(X0) nu are exact 3 componente
    if d[-1]!=3:
        return 0,0,0,0,0
    inf=0.5
    sup=2
    m,n,p=np.shape(X0)
    X=X0.astype(float)
    Q=np.array(Q).astype(float)
    Y0,Y,nz,nz_jpeg=jpgaux3(X0,Q*inf,m,n)
    s=' '*1000
    while np.sum((X-Y)**2)/np.sum(X**2)>prag:
        inf/=2
        Y0,Y,nz,nz_jpeg=jpgaux3(X0,Q*inf,m,n)
        print('\rinf:',inf,end=s)
    Y0,Y,nz,nz_jpeg=jpgaux3(X0,Q*sup,m,n)
    while np.sum((X-Y)**2)/np.sum(X**2)<prag:
        sup*=2
        Y0,Y,nz,nz_jpeg=jpgaux3(X0,Q*sup,m,n)
        print('\rsup:',sup,end=s)
    act=(inf+sup)/2
    Y0,Y,nz,nz_jpeg=jpgaux3(X0,Q*act,m,n)
    while act!=sup and act!=inf:
        if np.sum((X-Y)**2)/np.sum(X**2)<prag:
            inf=act
        else:
            sup=act
        act=(inf+sup)/2
        print('\ract:',act,'; eroare:',sup-inf,
              '; procesat:',-np.log2(sup-inf)/52*100,'%',end=s)
        Y0,Y,nz,nz_jpeg=jpgaux3(X0,Q*act,m,n)
    print('\r',s,end='\r')
    return Y0,Y,nz,nz_jpeg,act

print('\n3:')
Y0,Y,nz,nz_jpeg,act=JPEG3(X,Q_jpeg,prag) # Y0 contine blocuri dct, Y este imagine rgb
fig,ax=plt.subplots(1,2,num=3,figsize=(20,10),clear=True)
ax[0].imshow(X)
ax[0].set_title('Original')
ax[1].imshow(Y.astype(np.uint8))
ax[1].set_title('JPEG')

print('Componente în frecvență: '+str(nz)+
      '\nComponente în frecvență după cuantizare: '+str(nz_jpeg)+
      '\nFactor MSE: '+str(prag)+'\n')

#%%
# exemplificare huffman
sw,X0,sh,sample=hcod(Y0.astype(np.float32))
m,n=np.shape(X)[:2] # imaginea se poate reconstrui din sw,X0,sh,sample,m,n
Y1=hdec(sw,X0,sh,sample)
X1=toImg(Y1,m,n)[0].astype(np.uint8)
fig,ax=plt.subplots(1,2,num=30,figsize=(20,10),clear=True)
ax[0].imshow(X)
ax[0].set_title('Original')
ax[1].imshow(X1)
ax[1].set_title('JPEG si huffman')

#%%
# 4
# Extindere pentru compresie video

video=[]
path='original.mkv'
cap=cv2.VideoCapture(path)
ret=True
while ret:
    ret,img=cap.read()  # citește un frame din obiectul 'cap'; img este (H,W,C), in formatul BGR
    if ret:
        video.append(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) # conversie de la BGR la RGB
# video are dimensiuni (T,H,W,C) si este in format RGB
# original.mkv are dimensiunea 1.5 MB (dureaza 10 secunde, e deja comprimat cu un algoritm pentru video)
# array-ul de cadre video are dimensiunea 3.77 GB (de 2600 de ori mai mult)
# deci, din lipsa de ram (ar fi nevoie de cel putin 64 GB de ram)
# nu se vor pastra transformatele dct ale tuturor cadrelor si
# nu se va aplica huffman pe intregul array
# de asemenea, algoritmul va dura o ora;
# cv2.VideoWriter nu va reusi sa impacheteze cadrele atat de comprimate
# ca formatul .mkv (de 2600 de ori), de aceea transformat.mkv va fi vizibil mai mare
# decat original.mkv si vizibil mai neclar
d=tuple([len(video)])+np.shape(video[0])
nz=0
nz_jpeg=0
s=' '*100
fourcc=cv2.VideoWriter_fourcc(*'mp4v')
out=cv2.VideoWriter('transformat.mkv',fourcc=fourcc,fps=cap.get(cv2.CAP_PROP_FPS),frameSize=(d[2],d[1]))
print('\n4:')
for i in range(d[0]):
    temp=jpgaux3(video[i],Q_jpeg,d[1],d[2])
    nz+=temp[2]
    nz_jpeg+=temp[3]
    video[i]=cv2.cvtColor(temp[1].astype(np.uint8),cv2.COLOR_RGB2BGR)
    out.write(video[i])
    video[i]=0
    print('\rcadrul',i+1,'din',d[0],end=s)
print('\rComponente în frecvență: '+str(nz)+
      '\nComponente în frecvență după cuantizare: '+str(nz_jpeg)+'\n')
out.release() # De remarcat ce mult scade calitatea scrisului