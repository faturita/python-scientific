"""
==================
Final Assignment
==================

El largo de los registros es entre 10 y 11 minutos
Fs = 512

FECHA DE ENTREGA: 31/07/2024
SEGUNDA FECHA DE ENTREGA: 10/08/2024


|---- BASELINE --------|
|---- PESTANEO ------|
|---- INHALAR ------- |
|---- EXHALAR ----|
|---- GOLPES (ESTIMULOS) --------|
|---- MENTAL IMAGERY ------|

* Baseline: esta parte la pueden utilizar para tener ejemplos negativos de cualquier cosa que deseen detectar.  Por 
ejemplo si quieren detectar que algo cambia cuando hay "imaginación en colores violeta", extraen features de ese momento y de
este e intentan armar un clasificador.
* Inhalar: Cambios en frecuencias bajas, donde se enfatice el movimiento del diafragma para inhalar.
* Exhalar: Idem pero enfatizando el movimiento del diafragma para exhalar.
* Golpes: Deberían aparecer algún evento con cierto pico en la señal.  Pueden intentar detectar estos picos y ver si hay algo.
* Mental Imagery: Pueden aparecer frecuencias altas de 20-50 Hz.  Aumentos en la potencia de estas bandas entre este bloque y el baseline.


Los datos, el registro de EEG y el video, están disponibles en el siguiente link:
https://drive.google.com/file/d/1ByQDK4ZPxbqw7T17k--avTcgSCCzs3vi/view?usp=sharing

Objetivo:
El objetivo es dado este registro implementar un análisis de estos datos, exploratorio, superviado 
o no supervisado, para intentar identificar que es lo que el sujeto está haciendo en cada bloque.  Pueden 
intentar separar dos bloques entre sí, un bloque particular frente al BASELINE (esto es el momento cuando el sujeto
no hace nada particular).  Pueden usar una parte de dos bloques para entrenar y luego intentar predecir las otras partes.
Tienen que producir un PDF informe con gráficos/tablas breve y resumido.

"""

print(__doc__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO

# El protocolo experimental que implementamos tiene 2 datasets:
# 1- Dataset de las señales de EEG
# 2- El video de las imágenes.
#
#
# La idea es tomar estos datasets y derivar de forma automática las diferentes secciones.  Esto se puede hacer en base self-supervised, es
# decir tomar los datos de algún dataset, derivar los labels para cada secciones y luego intentar implementar un clasificador multiclase.
#
# Tienen que entregar un PDF, tipo Markdown con código, gráficos y cualquier insight obtenido del dataset.

signals = pd.read_csv('data/B001/eeg.dat', delimiter=' ', names = ['timestamp','counter','eeg','attention','meditation','blinking'])

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