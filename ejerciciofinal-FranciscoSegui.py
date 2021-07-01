# In[1]
"""
==================
Final Assignment
==================

Alumno: Francisco Seguí

"""
print('ECD')
print('Este ejercicio tiene dos maneras de resolverse.')
print('Lo tiene que tener listo para el fin de la Cuarentena')

print('Opción B: elijan una (al menos) pregunta e intentan implementar una solución, codificando en R, Java o python.')

print('0 - Construyan una alternativa para detectar pestañeos (blinking.dat) y trabajen sobre el dataset de pestañeos para simular y testear el abordaje propuesto.')


# %%
# Importo las liberías y leo el array

from numpy.core import numeric
from numpy.testing._private.utils import print_assert_equal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


signals = pd.read_csv('data/blinking.dat', delimiter=' ', names = ['timestamp','counter','eeg','attention','meditation','blinking'])

data = signals.values

eeg = data[:,2]

# %%

# Propongo una forma alternativa de delimitar dinámicamente los umbrales de detección de pestañeo
# Calculo los límites inferiores y superiores utilizando una medida de posición
# En este caso uso el percentil 0.5 y el 99.5, con lo cual se consideran como picos el 1% de los valores
# De esta forma el filtro es dinámico, y se adapta a los valores de la muestra.
# Vemos en el gráfico que el criterio funciona adecuadamente

lowerbound=int(np.percentile(eeg, 0.5))
upperbound=int(np.percentile(eeg, 99.5))

plt.plot(eeg, color="steelblue")
plt.plot(np.full(len(eeg),lowerbound), color="goldenrod", ls="--")
plt.plot(np.full(len(eeg),upperbound), color="goldenrod", ls="--")
plt.ylabel("Amplitude",size=10)
plt.xlabel("Timepoint",size=10)
plt.title("EEG Series with control limits",size=20)
plt.ylim([min(eeg)*1.1, max(eeg)*1.1 ])  ## dinamizo los valores del eje así se adapta a los datos que proceso
plt.annotate("Lower Bound",xy=(500,lowerbound+10),color="goldenrod")
plt.annotate("Upper Bound",xy=(500,upperbound+10),color="goldenrod")
plt.savefig('blinkingpeaks.png')
plt.show()


# %%
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

# %%
# Encuentro picos positivos
peak=np.where(blinks == 1)[0]

peakdiff=np.diff(np.append(0,peak))

boolpeak=list(map(lambda x : x > 100, peakdiff))

peakslocation=list(filter(lambda x: x, boolpeak*peak))

print(peakslocation)

# %%
# Repito para los valles
valley=np.where(blinks == -1)[0]

valleydiff=np.diff(np.append(0,valley))

boolvalley=list(map(lambda x : x > 100, valleydiff))

valleylocation=list(filter(lambda x: x, boolvalley*valley))

print(valleylocation)

# %%

blinklocations=np.append(peakslocation,valleylocation)
print(blinklocations)
blinklocations.shape

# %%
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

# %% 
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
print(filtro_eeg)

# %%
boolpeaks = np.where(eeg > upperbound)
print (boolpeaks)

# Pick the derivative
dpeaks = np.diff( eeg )
print (dpeaks)
plt.plot(dpeaks)

#%%
# Identify those values where the derivative is ok
pdpeaks = np.where( dpeaks > 0)

print(pdpeaks)

peaksd = pdpeaks[0] 

print(peaksd)
#%%
# boolpeaks and peaksd are indexes.
finalresult = np.in1d(peaksd,boolpeaks)

print (finalresult)   

# %%
blinkings = finalresult.sum()

peaks1 = peaksd[finalresult]

a=np.diff(peaks1)
print(a)

print ('Blinkings: %d' % blinkings)
print ('Locations:');print(peaks1)



# %%
a=signals.eeg.rolling(64).max()
print(a)
plt.plot(a)

b=pd.DataFrame(signals.eeg.rolling(64).min(), signals.timestamp)
print(b)
plt.plot(b)

# %% 
a1= np.where(np.diff(a)==0)# and a > upperbound)
plt.plot(a1)

# %%

c=np.where(np.diff(b) > 0)[0] + 1
c.shape
# %%
# Grafico los valores de attention (rojos) y de meditation (azules)
# Vemos que en la primera parte ambos valores son bajos
# En el pico del timestamp 597 la attention sube repentinamente y alcanza un pico, que despues desciende gradualmente para volver hacia el final de la serie a los valores bajos.
# En el pico del timestamp 597 la mediation sube, pero alcanza su maximo en el pico del timestamp 600. Luego baja un poco pero se mantienen en valores altos.

import seaborn as sns
sns.set(style="darkgrid")
sns.lineplot(x="timestamp", y="eeg", hue="attention", data=signals, palette="Reds")
plt.show()

sns.lineplot(x="timestamp", y="eeg", hue="meditation", data=signals, palette="Blues")
plt.show()

