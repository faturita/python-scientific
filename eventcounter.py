"""
=====================
STEM Blinking Counter
=====================

Easy blinking counter.

Use me to find the peaks on the EEG signal obtained with Mindwave.
They are marks that in general correspond to blinking events.

= Pragmatic Signal Processing

Analytical signal processing:  There is a richful set of techniques and tool that can be used
to perform basic analysis of signals, trying to find peaks, troughs, valleys, and so on.  These
tools are very widespread in analytical chemestry.

This website is the best ever which contains a wonderful description of these tools:

https://terpconnect.umd.edu/~toh/spectrum/index.html

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

# Open blinking.dat file.
with open('data/blinking.dat') as inputfile:
    for row in csv.reader(inputfile):
        rows = row[0].split(' ')
        results.append(rows[1:])


# Convert the file into numpy array of ints.
results = np.asarray(results)
results = results.astype(int)


# Pick the EEG signal.
eeg = results[1:,1]


print('File Length:'+str(len(results)))
print("Some values from the dataset:\n")
print(results[0:10,])
print("Matrix dimension: {}".format(results.shape))
print("EEG Vector Metrics\n")
print("Length: {}".format(len(eeg)))
print("Max value: {}".format(eeg.max()))
print("Min value: {}".format(eeg.min()))
print("Range: {}".format(eeg.max()-eeg.min()))
print("Average value: {}".format(eeg.mean()))
print("Variance: {}".format(eeg.var()))
print("Std: {}".format(math.sqrt(eeg.var())))
plt.figure(figsize=(12,5))
plt.plot(eeg,color="green")
plt.ylabel("Amplitude",size=10)
plt.xlabel("Timepoints",size=10)
plt.title("Serie temporal de eeg",size=20)
plt.show()


# Prueba de normalidad
print('normality = {}'.format(scipy.stats.normaltest(eeg)))
sns.distplot(eeg)
plt.title("Normality-1 Analysis on EEG vector")
plt.show()
sns.boxplot(eeg,color="red")
plt.title("Normality-2 Analysis on EEG vector")
plt.show()
res = stats.probplot(eeg, plot = plt)
plt.title("Normality-3 Analysis on EEG vector") 
plt.show()


#Find the threshold values to determine what is a blinking and what is not
umbral_superior=int(eeg.mean()+3*eeg.std())
print("Upper Threshold: {}".format(umbral_superior))
umbral_inferior=int(eeg.mean()-3*eeg.std())
print("Lower Threshold: {}".format(umbral_inferior))
plt.figure(figsize=(12,5))
plt.plot(eeg,color="green")
plt.plot(np.full(len(eeg),umbral_superior),'r--')
plt.plot(np.full(len(eeg),umbral_inferior),'r--')
plt.ylabel("Amplitude",size=10)
plt.xlabel("Timepoint",size=10)
plt.title("EEG Series with control limits",size=20)
plt.annotate("Upper Threshold",xy=(500,umbral_superior+10),color="red")
plt.annotate("Lower Threshold",xy=(500,umbral_inferior+10),color="red")
plt.show()

'''
Now the EEG data is filtered to produce a new output, assigning 1, greater than the upper limit, 0 between lower and upper
limit, and -1, under the lower limit.  In order to determine the number of valid events, changes from 0-1 will be counted
as a possible blinking event.
'''


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
        
print("Blinking counter: {}".format(contador))
filtro_eeg=np.asarray(filtro_eeg)
plt.figure(figsize=(16,5))
plt.plot(filtro_eeg,color="blue")
plt.title("Blinking Filter",size=20)
plt.ylabel("Class",size=10)
plt.xlabel("Timepoint",size=10)
plt.show()


# Alternative method

# The threshold is hardcoded, visually estimated.
signalthreshold = 420

# Filter the values above the threshold
boolpeaks = np.where( eeg > signalthreshold  )
print (boolpeaks)

# Pick the derivative
dpeaks = np.diff( eeg )
print (dpeaks)

# Identify those values where the derivative is ok
pdpeaks = np.where( dpeaks > 0)

peaksd = pdpeaks[0] 

# boolpeaks and peaksd are indexes.
finalresult = np.in1d(peaksd,boolpeaks)

print (finalresult)     
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

# In[2]
# Alumno: Francisco Seguí https://github.com/fseguior/
# Propongo una forma alternativa de delimitar dinámicamente los umbrales de detección de pestañeo
# Calculo los límites inferiores y superiores utilizando una medida de posición
# En este caso uso el percentil 1 y el 99, con lo cual se consideran como picos el 2% de los valores
# De esta forma el filtro es dinámico, y se adapta a los valores de la muestra.
# Vemos en el gráfico que el criterio funciona adecuadamente

lowerbound=int(np.percentile(eeg, 1))
upperbound=int(np.percentile(eeg, 99))

plt.plot(eeg, color="steelblue")
plt.plot(np.full(len(eeg),lowerbound), color="goldenrod", ls="--")
plt.plot(np.full(len(eeg),upperbound), color="goldenrod", ls="--")
plt.ylabel("Amplitude",size=10)
plt.xlabel("Timepoint",size=10)
plt.title("EEG Series with control limits",size=20)
plt.ylim([min(eeg)*1.1, max(eeg)*1.1 ])  ## dinamizo los valores del eje así se adapta a los datos que proceso
plt.annotate("Lower Bound",xy=(500,lowerbound+10),color="goldenrod")
plt.annotate("Upper Bound",xy=(500,upperbound+10),color="goldenrod")
plt.savefig('blinks.png')
plt.show()

# Grafico el filtro de pestañeos/blinking
# Utilizo una función lambda para marcar los pestañeos

blinks = list((map(lambda x: 1 if x >upperbound else ( -1 if x < lowerbound else 0), eeg)))
blinks = np.asarray(blinks)

plt.plot(blinks, color="darksalmon")
plt.title("Blinking Filter",size=20)
plt.ylabel("Class",size=10)
plt.xlabel("Timepoint",size=10)
plt.savefig('blinkingfilter.png')
plt.show()

# Encuentro picos positivos. Filtro los valores donde blink==1, y luego analizo que haya habido un salto realmente (para no contar dos veces puntos consecutivos).
# Con un map y una funcion lambda obtengo una lista con booleanos para los valores donde hay picos realmente.
# Luego los filtro con una función filter y otra lambda
peak=np.where(blinks == 1)[0]

peakdiff=np.diff(np.append(0,peak))

boolpeak=list(map(lambda x : x > 100, peakdiff))

peakslocation=list(filter(lambda x: x, boolpeak*peak))

# Repito para los valles, mismo algoritmo pero busco blinks == -1
valley=np.where(blinks == -1)[0]

valleydiff=np.diff(np.append(0,valley))

boolvalley=list(map(lambda x : x > 100, valleydiff))

valleylocation=list(filter(lambda x: x, boolvalley*valley))

# Hago un append de los valles y los picos, y los ordeno. Luego los cuento para imprimir tanto la cantidad de pestañeos, como la localización de los mismos

blinklocations=np.sort(np.append(peakslocation,valleylocation))

blinkcount=np.count_nonzero(blinklocations)

print(f'Count of Blinks: {blinkcount}')
print('Location of Blinks');print(blinklocations)

# %%
# Grafico los valores de attention (rojos) y de meditation (azules)
# Vemos que en la primera parte ambos valores son bajos
# En el pico del timestamp 597 la attention sube repentinamente y alcanza un pico, que despues desciende gradualmente para volver hacia el final de la serie a los valores bajos.
# En el pico del timestamp 597 la mediation sube, pero alcanza su maximo en el pico del timestamp 600. Luego baja un poco pero se mantienen en valores altos.

import seaborn as sns
import pandas as pd
signals = pd.read_csv('data/blinking.dat', delimiter=' ', names = ['timestamp','counter','eeg','attention','meditation','blinking'])

sns.set(style="darkgrid")
sns.lineplot(x="timestamp", y="eeg", hue="attention", data=signals, palette="Reds")
plt.show()

sns.lineplot(x="timestamp", y="eeg", hue="meditation", data=signals, palette="Blues")
plt.show()
