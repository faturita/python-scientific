#coding: latin-1
#
# STEM - Blinking Counter

# Este programa es un ejemplo de utilizacion de python para implementar un simple
# contador de penstaneos basados en una senal de EMG/EMG/EOG.
#
# Frecuencia de sampleo Fs = 128
#

import csv
import numpy as np


results = []

# Esta primera linea, abre el archivo 'blinking.dat' que se grabó
# al establecerse la conexión con el servidor.
with open('data/blinking.dat') as inputfile:
    for row in csv.reader(inputfile):
        rows = row[0].split(' ')
        results.append(rows[1:])

print ('Longitud del archivo:'+str(len(results)))

# Convert the file into numpy array of ints.
results = np.asarray(results)
results = results.astype(int)

# Strip from the signal anything you want


# La primer columna corresponde a el largo del archivo a considerar
# en relación a las muestras (1:100 serian las muestras) representante
# del tiempo.
# La segunda columna, corresponde a: eeg, attention y meditation.
eeg = results[1:,1]

print (eeg)

#eeg = np.zeros((64))

#eeg = np.arange(64)

#print eeg.shape

#eeg[32] = -60

#eeg[43] = -130

#eeg = eeg - baseline_als(eeg,10000,0.5)


import matplotlib.pyplot as plt
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.plot(eeg,'r', label='EEG')
plt.legend(loc='upper left');
plt.show()


# El threshold corresponde al limite en amplitud a considerar para discriminar
# que es un pestañeo de qué no lo es.
signalthreshold = 420



boolpeaks = np.where( eeg > signalthreshold  )
print (boolpeaks)
dpeaks = np.diff( eeg )
print (dpeaks)
pdpeaks = np.where( dpeaks > 0)
print (pdpeaks)
print (pdpeaks != 0)
a = np.in1d(pdpeaks,boolpeaks)
print (a)
blinkings = a.sum()

print ('Blinkings: %d' % blinkings)

import matplotlib.pyplot as plt
from scipy.signal import find_peaks

peaks, _ = find_peaks(eeg, height=200)
plt.plot(eeg)
plt.plot(peaks, eeg[peaks], "x")
plt.plot(np.zeros_like(eeg), "--", color="gray")
plt.show()