"""
==================
Baseline Removal
==================

Eliminación del Baseline usando interpolación

"""
print(__doc__)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy.sparse as sparse
from scipy.sparse.linalg  import spsolve

print('Eliminación de Baseline')

signals = pd.read_csv('data/blinking.dat', delimiter=' ', names = ['timestamp','counter','eeg','attention','meditation','blinking'])
data = signals.values
eeg = data[:,2]

# Creo una secuencia de puntos del largo de la señal [0,1,2,...len(eeg)]
# Esta va a ser la señal de drift que distorsiona la señal original.
time = np.linspace(0, len(eeg), len(eeg))

# Modifico la señal original SIMULANDO el agregado del drift
eeg = eeg +  time

plt.plot(eeg,'r', label='EEG')
plt.xlabel('t')
plt.ylabel('eeg(t)')
plt.title(r'EEG Signal with a upward drifting')
plt.ylim([-2000, 6500])
plt.xlim([0,len(eeg)])
plt.show()

from scipy.interpolate import interp1d

# Genero 100 puntos de 0 al valor de len(eeg)  [0,123,340,...,len(eeg)]
x = range(0,len(eeg),100)
y = eeg[x]                                                  # Me fijo el valor de EEG en esos puntos
f = interp1d(x, y,fill_value="extrapolate")                 # Calculo una f de interpolacion sobre esos puntos.  
                                                            # Esto me da una forma posible de la señal de base, solo considerando esos puntos.
f2 = interp1d(x, y, kind='cubic',fill_value="extrapolate")  # Idem con una función cúbica.

# Muestro las interpolaciones en un gráfico.
xnew = np.linspace(0, 6000, num=41, endpoint=True)
import matplotlib.pyplot as plt
plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
plt.title(r'Interpolating signal')
plt.legend(['data', 'linear', 'cubic'], loc='best')
plt.show()

# Ahora necesito regenerar esa función para todos los puntos intermedios interpolandolos.
baseline = f(range(len(eeg)))

# Y finalmente le resto esos puntos interpolados a la señal original.
eeg = eeg - baseline


plt.plot(eeg,'r', label='EEG')
plt.xlabel('t')
plt.ylabel('eeg(t)')
plt.title(r'EEG Signal')
plt.ylim([-2000, 2000])
plt.xlim([0,len(eeg)])
plt.show()

