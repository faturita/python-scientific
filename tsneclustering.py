"""
===============
tSNE Clustering
===============

Multidimensional Scaling

https://towardsdatascience.com/t-sne-python-example-1ded9953f26

"""
print(__doc__)


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

# the well-known MNIST dataset
# X corresponds to 1797 8x8 binary images of handwritten digit strokes
# y on the other hand contains the corresponding number that the stroke represent
X, y = load_digits(return_X_y=True)

# tSNE is a dimensionality Reduction technique (check Everitt).  
# At the end it is a clustering technique.
tsne = TSNE()
 
# Fit and transform to two dimensiones.
X_embedded = tsne.fit_transform(X)

# Plot with seaborn
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend='full', palette=palette)

plt.show()


