'''

WORK IN PROGRESS

'''


import scipy.stats as stats
r, p = stats.pearsonr(H[:,0], X[:,1])
print(f"PCA 0 vs 1:Pearson r: {r} and p-value: {p}")

