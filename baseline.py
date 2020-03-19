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


time = np.linspace(0, len(eeg), len(eeg))

eeg = eeg +  time

plt.plot(eeg,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'EEG Signal with a upward drifting')
plt.ylim([-2000, 2000]);
plt.xlim([0,len(eeg)])
plt.show()

from scipy.interpolate import interp1d

x = range(0,len(eeg),100)
y = eeg[x]
f = interp1d(x, y,fill_value="extrapolate")
f2 = interp1d(x, y, kind='cubic',fill_value="extrapolate")

xnew = np.linspace(0, 6000, num=41, endpoint=True)
import matplotlib.pyplot as plt
plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
plt.title(r'Interpolating signal')
plt.legend(['data', 'linear', 'cubic'], loc='best')
plt.show()

baseline = f(range(len(eeg)))

eeg = eeg - baseline


plt.plot(eeg,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'EEG Signal')
plt.ylim([-2000, 2000]);
plt.xlim([0,len(eeg)])
plt.show()

