"""
==================
Baseline Removal
==================

Baseline Removal based on basic interpolation.

Modpoly: Modified Polynomial: iterative approach similar to alpha-prune
ASLS: Assymetrical least square, whittaker smoothing: penalizes the roughtness of the baseline
MOR: Performs morph operations first to throw away outliers and iterate.
SNIP: identify peaks, throw them away and get the baseline iteratively.

- pybaselines: https://pybaselines.readthedocs.io/en/latest/installation.html

"""
print(__doc__)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy.sparse as sparse
from scipy.sparse.linalg  import spsolve

from pybaselines import Baseline, utils


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
eeg2 = eeg - baseline


plt.plot(eeg2,'r', label='EEG')
plt.xlabel('t')
plt.ylabel('eeg(t)')
plt.title(r'EEG Signal')
plt.ylim([-2000, 2000])
plt.xlim([0,len(eeg2)])
plt.show()

# %% PyBaseline routines

x = np.linspace(1, 1000, 1000)
# a measured signal containing several Gaussian peaks
signal = (
    utils.gaussian(x, 4, 120, 5)
    + utils.gaussian(x, 5, 220, 12)
    + utils.gaussian(x, 5, 350, 10)
    + utils.gaussian(x, 7, 400, 8)
    + utils.gaussian(x, 4, 550, 6)
    + utils.gaussian(x, 5, 680, 14)
    + utils.gaussian(x, 4, 750, 12)
    + utils.gaussian(x, 5, 880, 8)
)
# exponentially decaying baseline
true_baseline = 2 + 10 * np.exp(-x / 400)
noise = np.random.default_rng(1).normal(0, 0.2, x.size)

y = signal + true_baseline + noise
y = data[:,2] + time 
x = time
true_baseline = time


baseline_fitter = Baseline(x, check_finite=False)

bkg_1 = baseline_fitter.modpoly(y, poly_order=3)[0]
bkg_2 = baseline_fitter.asls(y, lam=1e7, p=0.02)[0]
bkg_3 = baseline_fitter.mor(y, half_window=30)[0]
bkg_4 = baseline_fitter.snip(
    y, max_half_window=40, decreasing=True, smooth_half_window=3
)[0]

plt.plot(x, y, label='raw data', lw=1.5)
plt.plot(x, true_baseline, lw=3, label='true baseline')
plt.plot(x, bkg_1, '--', label='modpoly')
plt.plot(x, bkg_2, '--', label='asls')
plt.plot(x, bkg_3, '--', label='mor')
plt.plot(x, bkg_4, '--', label='snip')

plt.legend()
plt.show()

eeg1 = eeg - bkg_1 + 1000
eeg2 = eeg - bkg_1 + 2000
eeg3 = eeg - bkg_1 + 3000
eeg4 = eeg - bkg_1 + 4000

plt.plot(eeg1,'r', label='EEG')
plt.plot(eeg2,'b', label='EEG')
plt.plot(eeg3,'g', label='EEG')
plt.plot(eeg4,'y', label='EEG')
plt.xlabel('t')
plt.ylabel('eeg(t)')
plt.title(r'EEG Signal')
plt.ylim([-1, 5000])
plt.xlim([0,len(eeg2)])
plt.show()