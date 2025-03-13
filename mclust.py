'''

Gaussian finite mixture models fitted via EM algorithm for
model-based clustering, classification, and density estimation,
including Bayesian regularization, dimension reduction for
visualisation, and resampling-based inference.

'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.decomposition import PCA

# Simulated spike waveform data (n_samples x n_features)
np.random.seed(42)
n_samples = 500
n_features = 4  # Example: peak, valley, energy, and width

# Generate synthetic spike features
X = np.vstack([
    np.random.multivariate_normal([10, 5, 3, 2], np.diag([1, 1, 1, 1]), n_samples // 2),
    np.random.multivariate_normal([2, 10, 4, 5], np.diag([1, 1, 1, 1]), n_samples // 2)
])

# Visualize the raw features
plt.figure(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], c='gray', marker='.')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Raw Spike Features')
plt.show()

# Dimensionality reduction using PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Fit a Gaussian Mixture Model using the EM algorithm
n_clusters = 2
gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
gmm.fit(X)
labels_gmm = gmm.predict(X)

# Fit a Bayesian Gaussian Mixture Model
bgmm = BayesianGaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
bgmm.fit(X)
labels_bgmm = bgmm.predict(X)

# Plot the GMM clustering results
plt.figure(figsize=(8, 8))
for i in range(n_clusters):
    plt.scatter(X_pca[labels_gmm == i, 0], X_pca[labels_gmm == i, 1], label=f'GMM Cluster {i + 1}')

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('GMM Spike Sorting Clusters (EM algorithm)')
plt.legend()
plt.show()

# Plot the Bayesian GMM clustering results
plt.figure(figsize=(8, 8))
for i in range(n_clusters):
    plt.scatter(X_pca[labels_bgmm == i, 0], X_pca[labels_bgmm == i, 1], label=f'Bayesian GMM Cluster {i + 1}')

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Bayesian GMM Spike Sorting Clusters')
plt.legend()
plt.show()

# Print cluster means and covariances for both models
print(f"GMM Cluster means:\n{gmm.means_}")
print(f"GMM Cluster covariances:\n{gmm.covariances_}")
print(f"Bayesian GMM Cluster means:\n{bgmm.means_}")
print(f"Bayesian GMM Cluster covariances:\n{bgmm.covariances_}")