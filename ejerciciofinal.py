"""
==================
Final Assignment
==================

Todos los archivos están subidos a campus.
El largo de los registros es entre 10 y 11 minutos
Fs = 512

FECHA DE ENTREGA: 24/12/2024
SEGUNDA FECHA DE ENTREGA: 10/01/2025


|---- BASELINE --------|
|---- PESTANEO ------|
|---- RISA ------- |
|---- TRUCO_DOS ----|
|---- TRUCO_SIETE --------|
|---- TRUCO_SECUENCIA ------|
|---- DEATHMETAL --------|
|---- BETHOVEN --------|

* Baseline: esta parte la pueden utilizar para tener ejemplos negativos de cualquier cosa que deseen detectar.  Por 
ejemplo si quieren detectar que algo cambia cuando hay "imaginación en colores violeta", extraen features de ese momento y de
este e intentan armar un clasificador.
* Pestaneos: los pestaneos son eventos temporales que pueden ser detectados directamente en la señal.
* Risa: pueden tratar de detectar eventos temporales (picos) en la señal que se correspondan con la risa.
* Truco_dos: pueden tratar de detectar cambios que representen el movimiento de la cara de los besitos.
* Truco_siete: pueden tratar de detectar cambios que representen el movimiento de la cara de las muecas.
* Truco_secuencia: entrenando con los otros dos tienen que tratar de detectar aca la secuencia que genero Agustina.
* Deathmetal: pueden tratar de detectar cambios ritmicos (espectrales) relacionados con la musica.
* Bethoven: pueden tratar de detectar cambios ritmicos (espectrales) relacionados con la musica.

Objetivo:
El objetivo es dado este registro implementar un análisis de estos datos, exploratorio, superviado 
o no supervisado, para intentar identificar que es lo que el sujeto está haciendo en cada bloque.  Pueden 
intentar separar dos bloques entre sí, un bloque particular frente al BASELINE (esto es el momento cuando el sujeto
no hace nada particular).  Pueden usar una parte de dos bloques para entrenar y luego intentar predecir las otras partes.
Tienen que producir un PDF informe con gráficos/tablas breve y resumido (no más de 4 páginas)

"""

print(__doc__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO

# El protocolo experimental que implementamos tiene 2 datasets:
# 1- Dataset de las señales de EEG
# 2- El video de las imágenes (de la grabación de la clase)
#
#
# La idea es tomar estos datasets y derivar de forma automática las diferentes secciones.  Esto se puede hacer en base self-supervised, es
# decir tomar los datos de algún dataset, derivar los labels para cada secciones y luego intentar implementar un clasificador multiclase.
#
# Tienen que entregar un PDF, tipo Markdown con código, gráficos y cualquier insight obtenido del dataset.

signals = pd.read_csv('data/blinking.dat', delimiter=' ', names = ['timestamp','counter','eeg','attention','meditation','blinking'])

print('Estructura de la informacion:')
print(signals.head())

data = signals.values
eeg = data[:,2]

print(eeg)

plt.plot(eeg,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'EEG Signal')     # r'' representa un raw string que no tiene caracteres especiales
plt.ylim([-2000, 2000]);
plt.xlim([0,len(eeg)])
plt.show()