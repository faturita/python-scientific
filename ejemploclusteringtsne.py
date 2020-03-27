# coding: latin-1
'''
Excelente explicación del método: https://towardsdatascience.com/t-sne-python-example-1ded9953f26

'''

import numpy as np
from sklearn.datasets import load_digits
from scipy.spatial.distance import pdist
from sklearn.manifold.t_sne import _joint_probabilities
from scipy import linalg
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 10)

# Este dataset es muy conocido, se trata de MNIST.
# X corresponde a 1797 imagenes de 8x8 binarias con los trazos manuales de digitos.
# y por otro lado tiene el numero asociado para cada uno de los 1797
X, y = load_digits(return_X_y=True)

# tSNE es una tecnica de reduccion de la dimensionalidad (ver Everitt).  
# Al reducir y transformar las dimensiones, es una tecnica de clustering.
tsne = TSNE()
 
# Se asigna directamente el valor de X y se lo transforma a dos dimensiones.
X_embedded = tsne.fit_transform(X)

# Se usa seaborn para hacer el plot de los puntos.
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend='full', palette=palette)

plt.show()


