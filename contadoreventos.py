"""
=====================
STEM Blinking Counter
=====================

Contador de pestañeos.

Fs = 128

"""
print(__doc__)

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

import matplotlib.pyplot as plt
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.plot(eeg,'r', label='EEG')
plt.legend(loc='upper left')
plt.show()


# El threshold corresponde al limite en amplitud a considerar para discriminar
# que es un pestañeo de qué no lo es.  Idealmente debería utilizarse algún umbralizador automático (detección de outliers)
signalthreshold = 420

# Primero filtramos los valores de la señal que superan un umbral hardcoded
boolpeaks = np.where( eeg > signalthreshold  )
print (boolpeaks)

# Por otro lado, calculamos la derivada de la señal.
dpeaks = np.diff( eeg )
print (dpeaks)

# De la derivada, identificamos los valores positivos que corresponden a las curvas crecientes
pdpeaks = np.where( dpeaks > 0)

peaksd = pdpeaks[0] 
# boolpeaks y pdpeaks son indices. Chequeo cuales de los valores que tienen derivada creciente en peaksd, son tambien picos en boolpeaks
finalresult = np.in1d(peaksd,boolpeaks)

print (finalresult)     # Finalresult es una lista de valores booleanos que indican si cada valor de peaksd matchea o no la clausula.
blinkings = finalresult.sum()

peaks1 = peaksd[finalresult]

print ('Blinkings: %d' % blinkings)
print ('Locations:');print(peaks1)

import matplotlib.pyplot as plt
from scipy.signal import find_peaks

peaks2, _ = find_peaks(eeg, height=200)
plt.plot(eeg)
plt.plot(peaks2, eeg[peaks2], "x")
plt.plot(peaks1, eeg[peaks1], "o")
plt.plot(np.zeros_like(eeg), "--", color="gray")
plt.show()