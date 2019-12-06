# coding: latin-1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Algo de programación científica y métodos numéricos

val = (0.1)**2 
print (val == 0.01)

# Numpy da una funcion para chequear valores en punto flotante

print (np.isclose (0.1**2, 0.01))

#  Ojo al sumar valores en punto flotantes
a,b,c = 1e14, 25.44, 0.74
print ((a+b)+c)

print (a + (b+c))
