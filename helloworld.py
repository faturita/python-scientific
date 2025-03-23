"""
==================
HelloWorld
==================

Python se puede correr desde 5 lugares

1- X Terminal, ejecutando los comandos con 'python programa.py'
2- X Terminal, de manera interactiva con 'python', 'import programa'
3- X Terminal, con 'ipython', 'run 'holamundo''
4- Jupyter Notebooks.
5- Google Colab (u otros cloud providers).



-------------------------------------------
Un Holamundo de Data Science tiene que tener,

¿ Cómo leer un archivo y cargar un dataset ?
¿ Cómo ploter la info para arrancar un análisis exploratorio ?

¿ Qué pueden hacer para tratar de practicar ? 

1- Verifiquen que el campo counter en el archivo es consecutivo.  Este campo está asociado
a la frecuencia de sampleo del dispositivo.  Si todos los número están presentes entonces está ok.

2- Traten de identificar los picos.

"""
print(__doc__)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO

print('Hello Python Scientific World')

print('Objetivo: leer tensores y poder plotear sus valores...')

online = False
if (online == True):
    url = requests.get('https://raw.githubusercontent.com/faturita/python-scientific/master/data/blinking.dat')  
    csv_raw = StringIO(url.text)
    signals = pd.read_csv(csv_raw, delimiter=' ', names = ['timestamp','counter','eeg','attention','meditation','blinking'])

signals = pd.read_csv('data/blinking.dat', delimiter=' ', names = ['timestamp','counter','eeg','attention','meditation','blinking'])

print('Estructura de la informacion:')
print(signals.head())

data = signals.values

print('Ahora tienen un tensor de numpy (data)')
print (data)

print('Forma %2d,%2d:' % (signals.shape))

print('Python slicing...[:,].  El \':\' sirve para indicar el rango desde hasta.  Los indices son posiciones segun la forma del tensor.')
eeg = data[:,2]

print(eeg)

plt.plot(eeg,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'EEG Signal')     # r'' representa un raw string que no tiene caracteres especiales
plt.ylim([-2000, 2000]);
plt.xlim([0,len(eeg)])
plt.savefig('grafico.eps')
plt.show()

import seaborn as sns
sns.set(style="darkgrid")
sns.lineplot(x="timestamp", y="eeg", hue="attention", data=signals)
import matplotlib.pyplot as plt
plt.show()

