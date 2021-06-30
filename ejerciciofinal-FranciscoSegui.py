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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


signals = pd.read_csv('data/blinking.dat', delimiter=' ', names = ['timestamp','counter','eeg','attention','meditation','blinking'])

data = signals.values

eeg = data[:,2]

# %%

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
plt.savefig('blinkingpeaks.png')
plt.show()


# %%
# Grafico el filtro de pestañeos/blinking

blinks = list((map(lambda x: 1 if x >upperbound else ( -1 if x < lowerbound else 0), eeg)))
blinks = np.asarray(blinks)

plt.plot(blinks, color="darksalmon")
plt.title("Blinking Filter",size=20)
plt.ylabel("Class",size=10)
plt.xlabel("Timepoint",size=10)
plt.savefig('blinkingfilter.png')
plt.show()


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

