"""
==================
Final Assignment
==================

Entre 10 y 11 minutos
Fs = 512


|---- BASELINE --------|
|---- TOSER ------|
|---- RESPIRAR FONDO ------- |
|---- RESPIRAR RAPIDO ----|
|---- CUENTA MENTAL --------|
|---- COLORES VIOLETA ------|
|---- COLORES ROJO --------|
|---- SONREIR -----|
|---- DESEGRADABLE -----| 
|---- AGRADABLE --------|
|---- PESTANEOS CODIGO ------ |

* Baseline: esta parte la pueden utilizar para tener ejemplos negativos de cualquier cosa que deseen detectar.  Por 
ejemplo si quieren detectar que algo cambia cuando hay "imaginación en colores violeta", extraen features de ese momento y de
este e intentan armar un clasificador.
* Toser: Probablemente queden registrados como picos, provocados por el propio movimiento de la cabeza.
* Respirar fondo vs respirar rápido: quizás puede haber un cambio en alguna frecuencia.
* Cuenta mental: Está reportado que esto genera cambios en las frecuencias altas gamma y beta, de entre 20-30 Hz.
* Colores violeta / rojo:  de acá primero pueden intentar ver si hay cambio en relación a baseline en la frecuencia
de 10 Hz porque para ambos casos cerré los ojos.  Luego pueden intentar ver si un clasificador les puede diferenciar las clases.
* Sonreir: esto quizás genere algunos artefactos, picos en la señal debido al movimiento de la cara.
* Agradable/Desagradable: aca no tengo idea, prueben contra baseline.  No hay nada reportado así.
* Pestañeos:  En esta parte hay pestañeos que pueden intentar extraer.


Los datos, el registro de EEG y el video, están disponibles en el siguiente link:
https://drive.google.com/file/d/1ByQDK4ZPxbqw7T17k--avTcgSCCzs3vi/view?usp=sharing

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