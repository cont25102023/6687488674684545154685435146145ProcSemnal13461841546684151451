import numpy as np
import matplotlib.pyplot as plt

# generarea unei serii de timp cu o anomalie
def ser(length):
    time = np.arange(length)
    trend = (time-length/2)**4*2/length**3
    noise = np.random.normal(0, 1, length)
    time_series = trend + noise
    return time_series # serie cu trend
n=100 # lungimea seriei
v=ser(n)
for i in range(50,70):
    v[i]+=3+min(i-50,70-i)/5 # produce de anomalii
plt.figure()
plt.plot(v,label='Serie initiala')
plt.legend()

#%%

# determinarea trendului
def dt(serie,d): # d = gradul polinomului
    t=np.arange(0,len(serie))
    A=np.vander(t,d+1,True).astype(float)
    A1=np.transpose(A)
    c=np.linalg.solve(A1@A,A1@v)
    return c,sum(c[i]*t**i for i in range(d+1))

#%%

# metoda z-score

plt.figure()
plt.plot(v,label='Serie initiala')
plt.legend()

anm=np.zeros(n)
grade=5
for d in range(2,grade+1):
    c,trend=dt(v,d)
    r=v-trend
    plt.figure()
    plt.plot(r,label=f'Eliminare trend de ordin {d}')
    plt.legend()
    m=np.sum(r)/n
    sd=np.sqrt(sum((r-m)**2)/n)
    temp=np.where(abs(r-m)>=sd/4,r-m,0)
    sd=np.sqrt(sum(temp**2)/np.count_nonzero(temp))
    z=(r-m)/sd
    anm+=np.where(abs(z)>=1.6,d,0)
M=np.max(anm)
plt.figure()
plt.plot(v,label='serie',color='b')
x,y=[],[]
for i in range(n):
    if anm[i]>=0.7*M:
        x.append(i)
        y.append(v[i])
plt.plot(x,y,label='anomalii',linestyle='none',marker='o',markerfacecolor='r')
plt.legend()

#%%

# metoda de deviatie medie absoluta

plt.figure()
plt.plot(v,label='Serie initiala')
plt.legend()

anm=np.zeros(n)
grade=5
for d in range(2,grade+1):
    c,trend=dt(v,d)
    r=v-trend
    plt.figure()
    plt.plot(r,label=f'Eliminare trend de ordin {d}')
    plt.legend()
    m=np.sort(r)[n//2]
    sd=sum(abs(r-m))/n
    z=(r-m)/sd
    anm+=np.where(abs(z)>=2.2,d,0)
M=np.max(anm)
plt.figure()
plt.plot(v,label='serie',color='b')
x,y=[],[]
for i in range(n):
    if anm[i]>=0.7*M:
        x.append(i)
        y.append(v[i])
plt.plot(x,y,label='anomalii',linestyle='none',marker='o',markerfacecolor='r')
plt.legend()