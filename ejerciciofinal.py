# coding: latin-1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print('Este ejercicio tiene dos maneras de resolverse.')
print('Lo tiene que tener listo para el Miercoles 18 de Diciembre.')

print('Opción A: contestan preguntas directamente en relación a cómo resolverían el problema.  Es decir explicando por qué lo que plantean podría funcionar.')
print('Opción B: elijan una pregunta e intentan implementar una solución, codificando en R, Java o python.')

print('0 - Construyan una alternativa para detectar pestañeos y trabajen sobre el dataset de pestañeos para simular y testear el abordaje propuesto.')
print('1 - De las señales del EPOC Emotiv que obtuvimos de Adrian, intenten clasificarlos y cómo abordarían el problema con estas series de tiempo.')

print('2 - Sobre los datos de MNIST, intenten luego de clusterizar armar un clasificador.')

print('3 - Busquen un dataset de internet público de señales de sensores.  ¿Cómo lo abordarían exploratoriamente, qué procesamiento y qué análisis harían?')

print('4 - Prueben alternativas para mejorar la clasificación de las ondas alfa.')

print('5 - ¿Que feature utilizarian para mejorar la clasificacion que ofrece Keras con MLP para las series de tiempo?')

# El experimento que hicimos con Adrian está en el directorio data/experimentoadrian.dat

'''
El formato de los datos es

        "COUNTER",
        "BATTERY",
        "FC6",
        "F8",
        "T8",
        "PO4",
        "F4",
        "AF4",
        "FP2",
        "OZ",
        "P8",
        "FP1",
        "AF3",
        "F3",
        "P7",
        "T7",
        "F7",
        "FC5",
        "GYRO_X",
        "GYRO_Y",
        "RESERVED",
        "ETE1",
        "ETE2",
        "ETE3",

Los datos buenos que tomamos son F6 y F7, GYRO_X y GYRO_Y.

'''