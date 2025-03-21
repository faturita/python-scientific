"""
====================================
Toy Sample for clustering techniques
====================================

Based on Phil Roth <mr.phil.roth@gmail.com> Clustering sample

This programs check differences between kmeans clustering and dbscan.

KMeans: requires the number of clusters.  So, you can try different options and check some measurement of "good clustering"
DBScan: does not requires the number of cluster, but instead it needs the eps, the radius, and the minimun number of elements.

"""


print(__doc__)

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

def plotclusters(labels):
    # #############################################################################
    # Plot result


    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

# #############################################################################
# Generate random sample data, three clusters.
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)

# Add extra samples
#X2, labels_true2 = make_blobs( n_samples=30, centers=[[-1,1]],cluster_std=0.4,random_state=0)
#X = np.concatenate((X,X2))
#labels_true = np.concatenate((labels_true, labels_true2))


# ZScoring....
X = StandardScaler().fit_transform(X)


# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# Count the number of labels assigned to -1 which is no cluster, hence noise.
n_noise_ = list(labels).count(-1)



# #############################################################################
# Compute Kmeans
lblsk2 = KMeans(n_clusters=2).fit_predict(X)
print("KMeans 2 clusters: Silhouette Coefficient (-1,1): %0.3f"
      % metrics.silhouette_score(X, lblsk2))
plotclusters(lblsk2)

lblsk3 = KMeans(n_clusters=3).fit_predict(X)
print("KMeans 3 clusters: Silhouette Coefficient (-1,1): %0.3f"
      % metrics.silhouette_score(X, lblsk3))
plotclusters(lblsk3)

lblsk4 = KMeans(n_clusters=4).fit_predict(X)
print("KMeans 4 clusters: Silhouette Coefficient (-1,1): %0.3f"
      % metrics.silhouette_score(X, lblsk4))
plotclusters(lblsk4)

print("DBScan Silhouette Coefficient (-1,1): %0.3f"
      % metrics.silhouette_score(X, labels))

plotclusters(labels)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

# These are metrics that can be calculated for clusters.  Silhouette is the most widespread.
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))

print(__doc__)
