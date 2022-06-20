'''
==================
Simple Regression
==================

MSE Regression

'''


import numpy as np
import statsmodels.api as sm

y = [1,2,3,4,3,4,5,4,5,5,4,5,4,5,4,5,6,5,4,5,4,3,4]

x = [
     [4,2,3,4,5,4,5,6,7,4,8,9,8,8,6,6,5,5,5,5,5,5,5],
     [4,1,2,3,4,5,6,7,5,8,7,8,7,8,7,8,7,7,7,7,7,6,5],
     [4,1,2,5,6,7,8,9,7,8,7,8,7,7,7,7,7,7,6,6,4,4,4]
     ]

x = [4,2,3,4,5,4,5,6,7,4,8,9,8,8,6,6,5,5,5,5,5,5,5]

def reg_m(y, x):
    ones = np.ones(len(x))
    X = sm.add_constant(np.column_stack((x, ones)))
    #for ele in x[1:]:
    #    X = sm.add_constant(np.column_stack((ele, X)))
    results = sm.OLS(y, X).fit()
    return results

val = reg_m(y, x)

print(val.summary())

import matplotlib.pyplot as plt

print(val.params)

plt.plot(x,y, 'rx')
plt.plot(x, np.asarray(x)*val.params[0]+val.params[1], 'b')
plt.show()

