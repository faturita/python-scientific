"""
==================
Filtros Espaciales
==================

Blind Source Separation Techniques


"""
print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from sklearn.decomposition import FastICA, PCA

# #############################################################################
# Primero clavamos la semilla en un valor fijo para proder reproducir el ejemplo.

np.random.seed(0)
n_samples = 2000

# Generamos 2000 muestras entre 0 y 8 igualmente espaciadas (dividimos 0 a 8 en 2000)
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)  # Señal 1, sinusoidal.
s2 = np.sign(np.sin(3 * time))  # Señal 2, cuadrada
s3 = signal.sawtooth(2 * np.pi * time)  # Señal 3, sawtooth

# Muestro el primer grafico donde aparecen todas las señales crudas, tal como serían en la realidad.
plt.figure(1)
plt.title('Sinusoidal')
plt.subplot(3,1,1)
plt.plot(s1, color='red')
plt.title('Squared')
plt.subplot(3,1,2)
plt.plot(s2, color='steelblue')
plt.title('Sawtooth')
plt.subplot(3,1,3)
plt.plot(s3, color='orange')


# Las señales se concatenan en una unica estructura
S = np.c_[s1, s2, s3]
S += 0.2 * np.random.normal(size=S.shape)  # Agregamos ruido a cada una de 0.2 (alta frecuencia)

# Estas serían las señales tal como se registrarían en los sensores, sin ninguna mezcla.
plt.figure(2)
plt.title('Noisy Signals')
plt.subplot(3,1,1)
plt.plot(S[:,0], color='red')
plt.title('')
plt.subplot(3,1,2)
plt.plot(S[:,1], color='steelblue')
plt.title('')
plt.subplot(3,1,3)
plt.plot(S[:,2], color='orange')


S /= S.std(axis=0)  # Standardize data

# Se mezclan datos segun la matriz A de mezcla.  Luego se multiplican las matrices.
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations

print("Las Observaciones son los datos originales mezclados %d,%d" % X.shape)

plt.figure(2)
plt.title('Observation 1')
plt.subplot(3,1,1)
plt.plot(X[:,0], color='red')
plt.title('Observation 2')
plt.subplot(3,1,2)
plt.plot(X[:,0], color='steelblue')
plt.title('Observation 3')
plt.subplot(3,1,3)
plt.plot(X[:,0], color='orange')


# A partir de aca, se intenta hacer el camino inverso y separar las muestras buscando sus componentes.
# Compute ICA
ica = FastICA(n_components=3)
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix

# We can `prove` that the ICA model applies by reverting the unmixing.
assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

# For comparison, compute PCA
pca = PCA(n_components=3)
H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

# #############################################################################
# Plot results


plt.figure(3)
plt.title('ICA 1')
plt.subplot(3,1,1)
plt.plot(S_[:,0], color='red')
plt.title('ICA 2')
plt.subplot(3,1,2)
plt.plot(S_[:,1], color='steelblue')
plt.title('ICA 3')
plt.subplot(3,1,3)
plt.plot(S_[:,2], color='orange')


import scipy.stats as stats
r, p = stats.pearsonr(S_[:,0], -X[:,1])
print(f"ICA 0 vs 1:Pearson r: {r} and p-value: {p}")

r, p = stats.pearsonr(S_[:,1], X[:,2])
print(f"ICA 1 vs 2:Pearson r: {r} and p-value: {p}")

r, p = stats.pearsonr(S_[:,2], X[:,0])
print(f"ICA 2 vs 0:Pearson r: {r} and p-value: {p}")

plt.figure(4)
plt.title('PCA 1')
plt.subplot(3,1,1)
plt.plot(H[:,0], color='red')
plt.title('PCA 2')
plt.subplot(3,1,2)
plt.plot(H[:,1], color='steelblue')
plt.title('PCA 3')
plt.subplot(3,1,3)
plt.plot(H[:,2], color='orange')

import scipy.stats as stats
r, p = stats.pearsonr(H[:,0], X[:,1])
print(f"PCA 0 vs 1:Pearson r: {r} and p-value: {p}")

r, p = stats.pearsonr(H[:,1], X[:,2])
print(f"PCA 1 vs 2:Pearson r: {r} and p-value: {p}")

r, p = stats.pearsonr(H[:,2], X[:,0])
print(f"PCA 2 vs 0:Pearson r: {r} and p-value: {p}")


plt.figure(5)
models = [X, S, S_, H]
names = ['Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals',
         'PCA recovered signals']
colors = ['red', 'steelblue', 'orange']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(4, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
plt.show()