"""
=====================
Umbralizador
=====================

Pongo el de Otsu en 1-D por ahora: https://en.wikipedia.org/wiki/Otsu%27s_method

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

# La siguiente función es la implementación del método de Otsu en una dimensión.
# Esto permite encontrar uno (o más) valores que permiten segmentar la información de la entrada.
def otsu(signal):
    h = np.histogram(signal, bins=100)
    bins = h[1]
    p = h[0] / len(h[0])

    def w1(h,t):
        s = h[0:t].sum()
        return s

    def w2(h,t):
        s = h[t:].sum()
        return s

    def mu1(h,t): 
        val = 0 
        for i in range(0,t): 
            val = val + (i * h[i])/(w1(h,t))   
        return val

    def mu2(h,t): 
        val = 0 
        for i in range(t,len(h)): 
            val = val + (i * h[i])/(w2(h,t))  
        return val

    def sigma1(h,t):
        val = 0
        for i in range(0,t):
            val = val + ((i - mu1(h,t))**2) * h[i] / w1(h,t)
        return val

    def sigma2(h,t):
        val = 0
        for i in range(0,t):
            val = val + ((i - mu2(h,t))**2) * h[i] / w2(h,t)
        return val

    maxval = 0
    maxt = 0
    for t in range(1,len(p)-1):
        print(t)
        val = w1(p,t)*sigma1(p,t) + w2(p,t)*sigma2(p,t)
        print(val)
        if (val > maxval):
            maxval = val
            maxt = t

    print('Threshold value:' + str(maxt))

    return maxt

# El método de Otsu me da el umbral 

val = [1,10,20,22,12,23,12,23,23,89,99,102,104,102,109,97,79]
signal = np.asarray(val)

otsuvalue = otsu(signal)

reval = signal[signal>otsuvalue]

print (reval)

threshold = otsu(eeg)

