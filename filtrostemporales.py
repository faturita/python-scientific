"""
==================
Filtros Temporales
==================

Algunos filtros temporales

El más básico es mediante 

"""
print(__doc__)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

signals = pd.read_csv('data/blinking.dat', delimiter=' ', names = ['timestamp','counter','eeg','attention','meditation','blinking'])

data = signals.values

eeg = data[:,2]

# Filtro de todos los valores solo aquellos que son efectivamente mayores a 50
eegf1 = eeg[eeg>50]

# Filtro los valores que son mayores a 10 y menores que -40
eegf2 = eeg[np.logical_or(eeg>10,eeg<-40)] 

print("Largo 1 %2d" % len(eeg))
print("Largo 2 %2d" % len(eegf1))
print("Largo 3 %2d" % len(eegf2))

convolvedsignal = np.convolve([1,2,3],[-1,1,-1], 'same')

plt.plot(eeg,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
#plt.title(r'Plot of CT signal $x(t)=325 \sin(2\pi 50 t)$');
plt.title(r'EEG Signal')     # r'' representa un raw string que no tiene caracteres especiales
plt.ylim([-2000, 2000]);
plt.xlim([0,len(eeg)])
plt.show()

# La operación de convolución permite implementar el suavizado del Moving Average
windowlength = 10
avgeeg = np.convolve(eeg, np.ones((windowlength,))/windowlength, mode='same')

plt.plot(avgeeg,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
#plt.title(r'Plot of CT signal $x(t)=325 \sin(2\pi 50 t)$');
plt.title(r'EEG Signal')     # r'' representa un raw string que no tiene caracteres especiales
plt.ylim([-2000, 2000]);
plt.xlim([0,len(avgeeg)])
plt.show()

