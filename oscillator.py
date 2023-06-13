'''

This is based on https://medium.com/analytics-vidhya/understanding-oscillators-python-2813ec38781d

Implementation of a forced damped harmonic oscillator.

'''


import numpy as np 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


t = np.linspace(0,15,10000)
freq = 4
omega = 2 * np.pi * freq
y = [0,2] #y[0]=x and y[1]=v


def harmonic(t,y):
    solution = [y[1],-omega*omega*y[0]]
    return solution
sho = solve_ivp(harmonic, [0,1000], y0 = y, t_eval = t)

plt.plot(t,sho.y[0])
plt.ylabel("Position")
plt.xlabel("Time")
plt.title('SHO', fontsize = 20)

plt.show()


t = np.linspace(0,15,1000)
y = [1,1]
gamma = 4
freq = 4
omega = 2 * np.pi *  freq
def sho(t,y):
    solution = (y[1],(-gamma*y[1]-omega * omega *y[0]))
    return solution
solution = solve_ivp(sho, [0,1000], y0 = y, t_eval = t)
plt.plot(t,solution.y[0])
plt.ylabel("Position")
plt.xlabel("Time")
plt.title('Damped Oscillator', fontsize = 20)


plt.show()



# example of numerical integration of 1D driven oscilla
from math import sin, sqrt
import matplotlib.pyplot as plt

# constants to give resonant frequency of about 1 /s
k = 2.0
m = 0.5
g = 0.05 # light damping
F = 1.0
w0 = sqrt(k/m)

w = 0.99 * w0
tmax = 150
dt = 0.01
t = 0
x = 0
v = 0
a = 0

# equation of motion
# m d2x/dt2 - 2 g dx/dt + kx = F sin wt

print('w0 = %f\n'%w0)
def force(x, v, t):
    return -g*v - k*x + F*sin(w*t)

X = [0.]
T = [0.]
FF = [1.]

while t<tmax:
    f = force(x,v, t)
    a = f / m
    x = x + v * dt + 0.5*a *dt*dt
    v = v + a * dt
    FF.append(F*sin(w*t))
    t = t + dt
    X.append(x)
    T.append(t)

plt.figure()
plt.plot(T,X)
plt.plot(T, FF)
plt.title('amplitude response of lightly damped driven oscillator')
plt.legend(('displacement','force'))
plt.xlabel('time')
plt.show()