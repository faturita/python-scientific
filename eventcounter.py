"""
=====================
STEM Blinking Counter
=====================

Easy blinking counter.

Use me to find the peaks on the EEG signal obtained with Mindwave.
They are marks that in general correspond to blinking events.

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