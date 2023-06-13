'''

Sample permutation test

http://www2.stat.duke.edu/~ar182/rr/examples-gallery/PermutationTest.html
http://sites.utexas.edu/sos/guided/inferential/categorical/univariate/binomial/

'''

import numpy as np
z = np.array([94,197,16,38,99,141,23])
y = np.array([52,104,146,10,51,30,40,27,46])


theta_hat = z.mean() - y.mean()
print (theta_hat)


def run_permutation_test(pooled,sizeZ,sizeY,delta):
     np.random.shuffle(pooled)
     starZ = pooled[:sizeZ]
     starY = pooled[-sizeY:]
     return starZ.mean() - starY.mean()

pooled = np.hstack([z,y])
delta = z.mean() - y.mean()
numSamples = 10000
estimates = np.array(map(lambda x: run_permutation_test(pooled,z.size,y.size,delta),range(numSamples)))
diffCount = len(np.where(estimates <=delta)[0])
hat_asl_perm = 1.0 - (float(diffCount)/float(numSamples))

sigma_bar = np.sqrt((  np.sum((z-z.mean())**2) + np.sum((y-y.mean())**2) ) / (z.size + y.size - 2.0))
sigma_bar


import scipy.stats as stats
1.0 - stats.norm.cdf(theta_hat / (sigma_bar * np.sqrt((1.0/z.size)+(1.0/y.size))))



1.0 - stats.t.cdf(theta_hat / (sigma_bar * np.sqrt((1.0/z.size)+(1.0/y.size))),14)

