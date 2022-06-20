"""
==================
Spectral Filters
==================

Spectral filters allow to filter information in the spectral domain, based on their frecuencies.

They could be low-pass, band-pass, and high-pass.
Their objective is to enhance SNR, by filtering out the noise contained in the signal.
Spectral Noise could be:

- White Noise: the power of the noise is the same in all frequencies. 
- Pink Noise: the power of the noise is inversily proportional to the frequency (higher frequency, less noise)

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


# Longitud de la señal (10 segundos de señal)
N = 1280
sr = 128.0

# Linspace me arma una secuencia de N números igualmente espaciados entre 0.0 y el largo a la frecuencia de sampleo
x = np.linspace(0.0, N, int(N*sr))
# A esa secuencia le agrego una señal pura de 10 Hz y una de 20 Hz de mucha mayor amplitud, emulando un ruído no deseado sobre la señal.
y = 1*np.sin(10.0 * 2.0*np.pi*x) + 9*np.sin(20.0 * 2.0*np.pi*x)

plt.plot(x, y)
plt.grid()
plt.title(r'Original Signal')
plt.axis((0,N,-20,20))
plt.show()


X = fft(y)
N = len(X)
n = np.arange(N)
# get the sampling rate
T = N/sr
freq = n/T 

# Get the one-sided specturm
n_oneside = N//2
# get the one side frequency
f_oneside = freq[:n_oneside]

plt.figure(figsize = (12, 6))
plt.plot(f_oneside, np.abs(X[:n_oneside]), 'b')
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')
plt.show()

from scipy.fft import rfft, rfftfreq



import pandas as pd
signals = pd.read_csv('data/blinking.dat', delimiter=' ', names = ['timestamp','counter','eeg','attention','meditation','blinking'])
data = signals.values
eeg = data[:,2]

Fs = 128.0

normalized_signal = eeg

N = len(normalized_signal)


# Creo una secuencia de N puntos (el largo de EEG), de 0 hasta el largo de la secuencia en segundos (N/Fs).
x = np.linspace(0.0, int(N/Fs), N)   

# A esa secuencia de EEG le agrego una señal pura de 30 Hz.  Estoy ayuda a visualizar bien que la relación espectral está ok.
normalized_signal +=  100*np.sin(30.0 * 2.0*np.pi*x)

yf = rfft(normalized_signal)
xf = rfftfreq(N, 1 / Fs)

plt.figure(figsize=(14,7))
plt.title('Frequency Spectrum')
plt.plot(xf, np.abs(yf), color='green')
plt.ylabel('Amplitude')
plt.xlabel('Frequency (Hertz)')
plt.show()





