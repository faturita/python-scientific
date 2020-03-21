"""
==================
Filtros Espectrales
==================

Los filtros espectrales permiten filtrar información en el espacio de frecuencias.

Pueden ser Pasa-bajo, Pasa-alto o pasa-banda.

"""
print(__doc__)
import numpy as np

import sys, select

import time
import datetime
import os

from scipy.fftpack import fft

import math

from scipy.signal import firwin, remez, kaiser_atten, kaiser_beta
from scipy.signal import butter, filtfilt, buttord

from scipy.signal import butter, lfilter

import matplotlib.pyplot as plt

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def psd(y):
    # Number of samplepoints
    N = 128
    # sample spacing
    T = 1.0 / 128.0
    # From 0 to N, N*T, 2 points.
    #x = np.linspace(0.0, 1.0, N)
    #y = 1*np.sin(10.0 * 2.0*np.pi*x) + 9*np.sin(20.0 * 2.0*np.pi*x)


    # Original Bandpass
    fs = 128.0
    fso2 = fs/2
    #Nd,wn = buttord(wp=[9/fso2,11/fso2], ws=[8/fso2,12/fso2],
    #   gpass=3.0, gstop=40.0)
    #b,a = butter(Nd,wn,'band')
    #y = filtfilt(b,a,y)

    y = butter_bandpass_filter(y, 8.0, 15.0, fs, order=6)


    yf = fft(y)
    #xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
    #import matplotlib.pyplot as plt
    #plt.plot(xf, 2.0/N * np.abs(yf[0:N/2]))
    #plt.axis((0,60,0,1))
    #plt.grid()
    #plt.show()

    return np.sum(np.abs(yf[0:int(N/2)]))

# Frecuencia de sampleo
N = 128
# sample spacing
T = 1.0 / 128.0

# Linspace me arma una secuencia de N números igualmente espaciados entre 0.0 y 1.0
x = np.linspace(0.0, 1.0, N)
# A esa secuencia le agrego una señal pura de 10 Hz y una de 20 Hz de mucha mayor amplitud, emulando un ruído no deseado sobre la señal.
y = 1*np.sin(10.0 * 2.0*np.pi*x) + 9*np.sin(20.0 * 2.0*np.pi*x)


plt.plot(x, y)
plt.grid()
plt.title(r'Original Signal')
plt.axis((0,1,-20,20))
plt.show()

# Aplico la transformada de fourier y compongo el eje X de las frecuencias para visualizar la señal transformada.
yf = fft(y)
xf = np.linspace(0.0, int(1.0/(2.0*T)), int(N/2))

plt.plot(xf, 2.0/N * np.abs(yf[0:int(N/2)]))
plt.grid()
plt.title(r'Signal spectrum.')
plt.axis((0,60,0,9))
plt.show()

# Le aplico un filtro pasabanda entre 8 y 15 Hz.  El resto se intenta planchar a cero.
y = butter_bandpass_filter(y, 8.0, 15.0, 128.0, order=6)

yf = fft(y)
xf = np.linspace(0.0, int(1.0/(2.0*T)), int(N/2))

plt.plot(xf, 2.0/N * np.abs(yf[0:int(N/2)]))
plt.grid()
plt.title(r'Output filtered signal spectrum.')
plt.axis((0,60,0,9))
plt.show()


plt.plot(x, y)
plt.grid()
plt.title(r'Output filtered signal.')
plt.axis((0,1,-20,20))
plt.show()

# Este bloque de código permite hacer el mismo análisis sobre la señal completa de EEG, expandiendo un segundo para cubrir toda la señal.
shamsignal = False
if (shamsignal):
    t = np.linspace(0, 1.0, 6430)
    T = 1.0 / 128.0
    N = 128.0
    tt=np.asarray([])
    for i in range(51):
        t = np.linspace(0.0, N*T, N) * N 
        t = t + i * N
        tt=np.concatenate((tt,t), axis=0)
        
    plt.plot(tt, 200 * np.sin(2*np.pi*50*tt),'b')