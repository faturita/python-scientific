'''

Estimation based on https://www.tandfonline.com/doi/abs/10.1080/00949657808810232


'''


from scipy.special import psi
from scipy.special import polygamma
from scipy.optimize import root_scalar
from numpy.random import beta
import numpy as np

def ipsi(y):
    if y >= -2.22:
        x = np.exp(y) + 0.5
    else:
        x = - 1/ (y + psi(1))  
    for i in range(5):
        x = x - (psi(x) - y)/(polygamma(1,x))
    return x
        
#%%
# q satisface
# psi(q) - psi(ipsi(lng1 - lng2 + psi(q)) + q) -lng2 = 0 
# O sea, busco raíz de 
# f(q) = psi(q) - psi(ipsi(lng1 - lng2 + psi(q)) + q) -lng2
# luego:
# p = ipsi(lng1 - lng2 + psi(q))
def f(q,lng1,lng2):
    return psi(q) - psi(ipsi(lng1 - lng2 + psi(q)) + q) -lng2

#%%
def ml_beta_pq(sample):
    lng1 = np.log(sample).mean()
    lng2 = np.log(1-sample).mean()
    def g(q):
        return f(q,lng1,lng2)
    q=root_scalar(g,x0=1,x1=1.1).root
    p = ipsi(lng1 - lng2 + psi(q))
    return p, q

#%%
p = 2
q = 5
n = 1500
sample = beta(p,q,n)
ps,qs = ml_beta_pq(sample) #s de sombrero

print(f'Estimación de parámetros de una beta({p}, {q}) \na partir de una muestra de tamaño n = {n}')
print(f'\nn ={n:5d} |   p   |   q')
print(f'---------+-------+------')
print(f'original | {p:2.3f} | {q:2.3f}')
print(f'estimado | {ps:2.3f} | {qs:2.3f}')

from scipy.stats import beta
from scipy.special import gamma as gammaf
import matplotlib.pyplot as plt

plt.hist(sample,bins=30,normed=True)
fitted=lambda x,a,b:gammaf(a+b)/gammaf(a)/gammaf(b)*x**(a-1)*(1-x)**(b-1) #pdf of beta

xx=np.linspace(0,max(sample),len(sample))
plt.plot(xx,fitted(xx,ps,qs),'g')

plt.show()