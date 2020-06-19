"""
==================
Ejercicio Final
==================


"""
print(__doc__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print('ECD2020')
print('Este ejercicio tiene dos maneras de resolverse.')
print('Lo tiene que tener listo para el fin de la Cuarentena')

print('Opción B: elijan una (al menos) pregunta e intentan implementar una solución, codificando en R, Java o python.')

print('0 - Construyan una alternativa para detectar pestañeos y trabajen sobre el dataset de pestañeos para simular y testear el abordaje propuesto.')
print('1 - De las señales del EPOC Emotiv que obtuvimos de SUJETO, intenten estudiar las señales detectando: los pestañeos sobre F8 y F7, el momento donde el sujeto cierra los ojos, donde abre y cierra la boca, donde mueve la cabeza haciendo Roll, y donde mueve la cabeza haciendo YAW.')
print('2 - Sobre los datos de MNIST, intenten luego de clusterizar armar un clasificador.')
print('3 - Busquen un dataset de internet público de señales de sensores.  ¿Cómo lo abordarían exploratoriamente, qué procesamiento y qué análisis harían?')
print('4 - Prueben alternativas para mejorar la clasificación de las ondas alfa.')
print('5 - ¿Que feature utilizarian para mejorar la clasificacion que ofrece Keras con MLP para las series de tiempo?')
print('6 - Suban un snippet que aborde alguna problemática de solución, implementé algún otro método de clasificación o de análisis, sobre los datos registrados en este repositorio.')


'''
El experimento para el Ejercicio (1) está en el directorio data/experimentosujeto.dat

Sujeto #1 se colocó el dispositivo de captura de señales de EEG EPOC Emotiv.  Cuatro canales se habilitaron F7, F8 frontales y O1,O2 occipitales.
El dispositvo además tiene información de dos IMUs, en Gyro_x y Gyro_y.
La persona estuvo sentada durante 5 minutos aproximadamente.  Durante diferentes períodos de tiempo realizó las siguientes acciones

* Movimiento de la cabeza hacia los laterales (Yaw)
* Movimiento de la cabeza hacia adelante y atrás (pitch)
* Movimiento de la cabeza hacia los lados (llevando las orejas a los hombros) (roll)
* Pestañeo voluntario intermitente
* Apertura y cierre de la boca.
* Cerró los ojos.
* Permaneció inmovil mirando un punto fijo (y pestañando naturalmente).

El formato de los datos es

        "COUNTER",
        "AF3",
        "F7",
        "F3",
        "FC5",
        "T7",
        "P7",
        "O1",
        "O2",
        "P8",
        "T8",
        "FC6",
        "F4",
        "F8",
        "AF4",
        "GYRO_X",
        "GYRO_Y",
        "RESERVED",
        "RESERVED",
        "RESERVED",
        "RESERVED",
        "RESERVED"

Los datos buenos que tomamos deberían ser F7 y F8, GYRO_X y GYRO_Y.

'''



# In[1]:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO



# %%

signals = pd.read_csv('data/experimentosujeto.dat', delimiter=' ', names = [
            "COUNTER",
        "AF3",
        "F7",
        "F3",
        "FC5",
        "T7",
        "P7",
        "O1",
        "O2",
        "P8",
        "T8",
        "FC6",
        "F4",
        "F8",
        "AF4",
        "GYRO_X",
        "GYRO_Y",
        "RESERVED1",
        "RESERVED2",
        "RESERVED3",
        "RESERVED4",
        "RESERVED5"])



# %%
signals.shape

# In[1]:
data = signals.values

eeg = data[:,8]

plt.plot(eeg,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'EEG Signal')     # r'' representa un raw string que no tiene caracteres especiales
plt.ylim([-2000, 2000]);
plt.xlim([0,len(eeg)])
plt.show()


# %%
data.shape

# %%
import mne
ch_names = [
            "COUNTER",
        "AF3",
        "F7",
        "F3",
        "FC5",
        "T7",
        "P7",
        "O1",
        "O2",
        "P8",
        "T8",
        "FC6",
        "F4",
        "F8",
        "AF4",
        "GYRO_X",
        "GYRO_Y",
        "RESERVED1",
        "RESERVED2",
        "RESERVED3",
        "RESERVED4",
        "RESERVED5"]

sfreq = 128
data =data[:,list([2,7,8,13,15,16])]

ch_renames = [ch_names[2],ch_names[7],ch_names[8],ch_names[13],ch_names[15],ch_names[16]]
ch_types = ['eeg' for _ in ch_renames]



info = mne.create_info(ch_renames, sfreq, ch_types=ch_types)

raw = mne.io.RawArray(data.T, info)
#raw.add_events(events)

raw.plot_psd()

raw.filter(1,20)

raw.plot_psd()


# %%
raw.plot(scalings='auto', block=True)

# %%
