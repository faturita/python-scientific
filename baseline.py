"""
==================
Baseline Removal
==================

Baseline Removal based on basic interpolation.

"""
print(__doc__)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy.sparse as sparse
from scipy.sparse.linalg  import spsolve

print('Baseline Removal')

signals = pd.read_csv('data/blinking.dat', delimiter=' ', names = ['timestamp','counter','eeg','attention','meditation','blinking'])
data = signals.values
eeg = data[:,2]

# Create a basic linearly spaced sequence to map the axis points [0,1,2,...len(eeg)]
# This points will be used to capture the signal drift.
time = np.linspace(0, len(eeg), len(eeg))

# Artificially modify the EEG signal with a linear drift.
eeg = eeg +  time

plt.plot(eeg,'r', label='EEG')
plt.xlabel('t')
plt.ylabel('eeg(t)')
plt.title(r'EEG Signal with an upward drifting')
plt.ylim([-2000, 6500])
plt.xlim([0,len(eeg)])
plt.show()

from scipy.interpolate import interp1d

# Get 100 points from 0 .. len(eeg)  [0,123,340,...,len(eeg)]
x = range(0,len(eeg),100)
y = eeg[x]                                                  # Get the signal values on those points
f = interp1d(x, y,fill_value="extrapolate")                 # Estimate a function that will interpolate those points. 
                                                            # This will estimate a waveform based solely on those points.
f2 = interp1d(x, y, kind='cubic',fill_value="extrapolate")  # Replicate the same with a cubic function

# Plot the interpolated values
xnew = np.linspace(0, 6000, num=41, endpoint=True)
import matplotlib.pyplot as plt
plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
plt.title(r'Interpolating signal')
plt.legend(['data', 'linear', 'cubic'], loc='best')
plt.show()

# Now regenerate the signal based on the estimated function 'f' and get a signal from that.
baseline = f(range(len(eeg)))

# Finally, substract those points from the original signal.
eeg = eeg - baseline


plt.plot(eeg,'r', label='EEG')
plt.xlabel('t')
plt.ylabel('eeg(t)')
plt.title(r'EEG Signal')
plt.ylim([-2000, 2000])
plt.xlim([0,len(eeg)])
plt.show()

