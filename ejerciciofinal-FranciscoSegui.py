# In[1]
"""
==================
Final Assignment
==================

Alumno: Francisco Seguí

"""
print(__doc__)
print('Opción B: elijan una (al menos) pregunta e intentan implementar una solución, codificando en R, Java o python.')
print('')
print('Elijo la opción 0')
print('')
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
sns.set(style="darkgrid")
sns.lineplot(x="timestamp", y="eeg", hue="attention", data=signals, palette="Reds")
plt.savefig('attention.png')
plt.show()

sns.lineplot(x="timestamp", y="eeg", hue="meditation", data=signals, palette="Blues")
plt.savefig('meditation.png')
plt.show()
