"""
=====================
Thresholding
=====================

Otsu method: https://en.wikipedia.org/wiki/Otsu%27s_method

Otsu method divides the signal into two classes, the background and the foreground.
It is a very simple method, but it is very effective.  It is based on the histogram
of the signal.  The method is based on the assumption that the signal contains two classes
of values following bi-modal histogram (foreground and background).  The algorithm calculates
the optimum threshold separating the two classes so that their intra-class variance is minimal.

"""
print(__doc__)

# %%
import csv
import numpy as np
import matplotlib.pyplot as plt

results = []

with open('data/blinking.dat') as inputfile:
    for row in csv.reader(inputfile):
        rows = row[0].split(' ')
        results.append(rows[1:])

print ('Length:'+str(len(results)))

# Convert the file into numpy array of ints.
results = np.asarray(results)
results = results.astype(int)

# Strip from the signal anything you want

eeg = results[1:,1]

print (eeg)

plt.plot(eeg,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'EEG Signal')     # r'' representa un raw string que no tiene caracteres especiales
plt.ylim([-2000, 2000]);
plt.xlim([0,len(eeg)])
plt.show()

# Otsu method provides the umbralization vlaue
# %%
val = [-40,-50,-45,-25,-44,-33,-45,-15,-23,-45,96,97,94,92,93,93,92,93,94,87,88]
val = eeg
signal = np.asarray(val)

def w1(p,t):
    s = p[0:t].sum()
    return s

def w2(p,t):
    s = p[t:].sum()
    return s

def mu1(p,t): 
    val = 0 
    for i in range(0,t): 
        val = val + (i * p[i])/(w1(p,t))   
    return val

def mu2(h,t): 
    val = 0 
    for i in range(t,len(p)): 
        val = val + (i * p[i])/(w2(p,t))  
    return val

def sigma1(h,t):
    val = 0
    for i in range(0,t):
        val = val + ((i - mu1(h,t))**2) * h[i] / w1(h,t)
    return val

def sigma2(h,t):
    val = 0
    for i in range(0,t):
        val = val + ((i - mu2(h,t))**2) * h[i] / w2(h,t)
    return val

h = np.histogram(signal,bins=np.ptp(signal))
bins = h[1]
p = h[0] / len(signal)
prob = p

maxval = 0
maxt = 0
sigmas = []
for t in range(len(prob)):   
    sigmab = w1(prob,t) * w2(prob,t) * (mu1(prob,t) - mu2(prob,t))**2
    sigmas.append(sigmab)

maxt = np.where(sigmas == np.amax(sigmas))

#otsuvalue = otsu(signal)
otsuvalue = bins[int(maxt[0].mean())]

print( f'Otsu thresholding:{otsuvalue}')

#reval = signal[signal>otsuvalue]


# %%
import matplotlib.pyplot as plt
plt.plot(bins[:-1], p*20000, linewidth=2, color='r')
plt.plot(bins[:-1], sigmas)
plt.axvline(x=otsuvalue, color='k', linestyle='--')
plt.xlim([bins.min(), bins.max()])
plt.show()
#threshold = otsu(eeg)


# %%
# This is wikipedia's implementation of Otsu's method:
def otsu_intraclass_variance(image, threshold):
	"""
	Otsu’s intra-class variance.
	If all pixels are above or below the threshold, this will throw a warning that can safely be ignored.
	"""
	return np.nansum([
		np.mean(cls) * np.var(image[cls])
		#   weight   ·  intra-class variance
		for cls in [image>=threshold,image<threshold]
	])
	# NaNs only arise if the class is empty, in which case the contribution should be zero, which `nansum` accomplishes.


otsu_threshold = min(
		range( np.min(eeg)+1, np.max(eeg) ),
		key = lambda th: otsu_intraclass_variance(eeg,th)
	)

print( f'Otsu thresholding:{otsuvalue}')

plt.plot(eeg,'r', label='EEG')
plt.axhline(y=otsu_threshold, color='k', linestyle='--')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'EEG Signal')     # r'' representa un raw string que no tiene caracteres especiales
plt.ylim([-500, 500]);
plt.xlim([0,len(eeg)])
plt.show()

# %%
