"""
=====================
STEM Blinking Counter
=====================

Contador de pestañeos.

Este código intenta encontrar los picos de la señal que generalmente corresponden a pestañeos en
señales de EOG, Electrooculografía.

Fs = 128

"""
print(__doc__)

import csv
import numpy as np
import seaborn as sns
import math
import scipy
from scipy import stats
import matplotlib.pyplot as plt

results = []

# Esta primera linea, abre el archivo 'blinking.dat' que se grabó
# al establecerse la conexión con el servidor.
with open('data/blinking.dat') as inputfile:
    for row in csv.reader(inputfile):
        rows = row[0].split(' ')
        results.append(rows[1:])


# Convert the file into numpy array of ints.
results = np.asarray(results)
results = results.astype(int)


# La primer columna corresponde a el largo del archivo a considerar
# en relación a las muestras (1:100 serian las muestras) representante
# del tiempo.
# La segunda columna, corresponde a: eeg, attention y meditation.
eeg = results[1:,1]


print ('Longitud del archivo:'+str(len(results)))
print("Primeros valores del detaset:\n")
print(results[0:10,])
print("Dimensiones de la matriz results: {}".format(results.shape))
print("Algunos valores del vector egg\n")
print("Longitud: {}".format(len(eeg)))
print("Máximo valor: {}".format(eeg.max()))
print("Mínimo valor: {}".format(eeg.min()))
print("Rango: {}".format(eeg.max()-eeg.min()))
print("Valor promedio: {}".format(eeg.mean()))
print("Varianza: {}".format(eeg.var()))
print("Desvío standard: {}".format(math.sqrt(eeg.var())))
plt.figure(figsize=(12,5))
plt.plot(eeg,color="green")
plt.ylabel("Medición",size=10)
plt.xlabel("Número de medición",size=10)
plt.title("Serie temporal de eeg",size=20)
plt.show()


# Prueba de normalidad
print('normality = {}'.format(scipy.stats.normaltest(eeg)))
sns.distplot(eeg)
plt.title("Supuestos de normalidad del vector eeg")
plt.show()
sns.boxplot(eeg,color="red")
plt.title("Supuestos de normalidad del vector eeg V2")
plt.show()
res = stats.probplot(eeg, plot = plt)
plt.title("Supuestos de normalidad V3") 
plt.show()


#Obtenemos nuestros umbrales para distinguir un parpadeo respecto a lo que no lo es
umbral_superior=int(eeg.mean()+3*eeg.std())
print("Umbral superior: {}".format(umbral_superior))
umbral_inferior=int(eeg.mean()-3*eeg.std())
print("Umbral inferior: {}".format(umbral_inferior))
plt.figure(figsize=(12,5))
plt.plot(eeg,color="green")
plt.plot(np.full(len(eeg),umbral_superior),'r--')
plt.plot(np.full(len(eeg),umbral_inferior),'r--')
plt.ylabel("Medición",size=10)
plt.xlabel("Número de medición",size=10)
plt.title("Serie temporal de eeg con límites de control",size=20)
plt.annotate("Umbral superior",xy=(500,umbral_superior+10),color="red")
plt.annotate("Umbral inferior",xy=(500,umbral_inferior+10),color="red")
plt.show()

"""
Aplicaremos filtros a nuestros datos para transformarlos en una terna según si están por encima del umbral
superior (asignar valor 1), por debajo del umbral inferior (asignar valor -1) o entre los 2 umbrales (asignar
valor 0). Luego para determinar la cantidad de parpadeos, se contará la cantidad de ocasiones en las cuales la
serie pasa de valor cero a valor uno, es decir la cantidad de ocasiones que desde un estado de reposo las
mediciones de eeg superan el umbral superior.
"""

filtro_eeg=[]
contador=0
for i in range(len(eeg)):
    if i==0:
        filtro_eeg.append(0)
    elif eeg[i]>umbral_superior:
        filtro_eeg.append(1)
        if eeg[i-1]<=umbral_superior:
            print(i)
            contador=contador+1
    elif eeg[i]<umbral_inferior:
        filtro_eeg.append(-1)
    else:
        filtro_eeg.append(0)
print("Cantidad de parpadeos: {}".format(contador))
filtro_eeg=np.asarray(filtro_eeg)
plt.figure(figsize=(16,5))
plt.plot(filtro_eeg,color="blue")
plt.title("Filtro temporal de parpadeos",size=20)
plt.ylabel("Clase ternaria",size=10)
plt.xlabel("Número de medición",size=10)
plt.show()


# Otro approach

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