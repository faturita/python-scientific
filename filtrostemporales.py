"""
==================
Filtros Temporales
==================

Algunos filtros temporales

El más básico es mediante filtros sobre la propia serie temporal con clausulas de comparación.
La operación de convolución es muy útil para la aplicación de filtros.

"""
print(__doc__)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Operacion de Convolucion matemática
print('Original Signal')
print([1,2,3])
print('Kernel:')
print([-1,1,-1])
convolvedsignal = np.convolve([1,2,3],[-1,1,-1], 'same')
print('Output Signal')
print(convolvedsignal)

# Leemos el array con Pandas
signals = pd.read_csv('data/blinking.dat', delimiter=' ', names = ['timestamp','counter','eeg','attention','meditation','blinking'])

data = signals.values

eeg = data[:,2]

# Filtro de todos los valores solo aquellos que son efectivamente mayores a 50
eegf1 = eeg[eeg>50]

# Muchas veces lo que me interesa es saber los índices (que en series de tiempo representan el tiempo) donde el filtro es positivo
# Esto se hace con el comando np.where
idxeeg1f = np.where( eeg > 50 )

# Filtro los valores que son mayores a 10 y menores que -40
eegf2 = eeg[np.logical_or(eeg>10,eeg<-40)] 

print("Largo 1 %2d" % len(eeg))
print("Largo 2 %2d" % len(eegf1))
print("Largo 3 %2d" % len(eegf2))

plt.plot(eeg,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'Original EEG Signal')     # r'' representa un raw string que no tiene caracteres especiales
plt.ylim([-2000, 2000]);
plt.xlim([0,len(eeg)])
plt.show()

# La operación de convolución permite implementar el suavizado del Moving Average
windowlength = 10
avgeeg = np.convolve(eeg, np.ones((windowlength,))/windowlength, mode='same')

# El kernel/máscara está compuesto de 10 valores de 1/10.  Cuando esos valores se suman para cada posición, implica que se reemplaza el valor por el promedio
# de los 5 valores anteriores y 4 posteriores.

plt.plot(avgeeg,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'Smoothed EEG Signal')     
plt.ylim([-2000, 2000]);
plt.xlim([0,len(avgeeg)])
plt.show()

