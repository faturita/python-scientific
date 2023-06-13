"""
==================
Symbolic
==================

Usage of python sympy symbolics to perform symbolic derivation/integration

"""
print(__doc__)
from sympy import symbols, integrate, exp, oo

x = symbols('x')
gaussian = exp(-x ** 2 / 2)
a = integrate(gaussian, (x,-oo, oo))

print ( a) 