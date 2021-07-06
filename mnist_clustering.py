"""
En este programa generaremos una clusterización de las diferentes imágenes del programa MNIST a partir del algoritmo
KMEANS. Luego vamos a asignar cada cluster_label a cada dígito real. Por último generaremos algunos modelos de ML
para clasificación de los diferentes dígitos.
"""


#Importamos las librerías a utilizar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
import random
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.cluster import spectral_clustering
from sklearn.cluster import AgglomerativeClustering


#Carga y análisis exploratorio del dataset a utilizar
X,y= load_digits(return_X_y=True)
print("\n Dimensiones del dataset:\n")
print(X.shape)
print("\n Dimensiones del vector solución:\n")
print(y.shape)
print("\n Cantidad de imágenes de cada dígito:\n")
print(np.unique(y,return_counts=True)) #Se ve que las clases están muy balanceadas (entre 174 y 183 ncounts)
imagen=X[14,:].reshape(8,8) #Ahora vamos a imprimir una de las imágenes
print(imagen)
print(y[14])
plt.figure(figsize=(10,10))
plt.imshow(imagen,cmap=plt.cm.gray)
plt.title("Imagen 1")
plt.gca().invert_yaxis()
plt.show()
X=255-X #Ahora invertiremos los colores para ver las imágenes com mayor claridad...
imagen2=X[14,:].reshape(8,8) #...y volvemos a imprimir la misma imagen
print(imagen2)
plt.figure(figsize=(10,10))
plt.imshow(imagen2,cmap=plt.cm.gray)
plt.title("Imagen 2")
plt.gca().invert_yaxis()
plt.show()
imagen3=X[1,:].reshape(8,8) #Realizamos una 3er impresión de otra imagen
print(imagen3)
plt.figure(figsize=(1,1))
plt.imshow(imagen3,cmap=plt.cm.gray)
plt.title("Imagen 3")
plt.gca().invert_yaxis()
plt.show()


#Realizamos ahora una clusterización de nuestro dataset
random.seed(0)
np.random.seed(0) #Colocamos una semilla para poder replicar el experimento
modelo=KMeans(n_clusters=10,init='random').fit(X) #Armado del modelo
resultado=modelo.predict(X) #Predicción
print("\n Imprimimos algunos parámetros del modelo:\n")
print("\n Cantidad de clusters: {}\n".format(modelo.n_clusters))
print("Coeficioente de Silhouette: %0.3f" % metrics.silhouette_score(X,resultado))
print("\n Centros de cada uno de los clusters: {}\n".format(modelo.cluster_centers_))
print("Grupo al cual pertenece cada una de las imágenes:\n {}".format(resultado))


#Seguimos con algunos gráficos de interés
#Tomemos el centro de algún cluster y observemos si se asemeja a algún dígito
imagen_prueba=modelo.cluster_centers_[3,:].reshape(8,8)
plt.imshow(imagen_prueba,cmap=plt.cm.gray)
plt.title("Imagen de prueba del centro del cluster n°3")
plt.gca().invert_yaxis()
plt.show() #Parece ser un 6. Lo verificaremos en breve...
#Ahora vamos a imprimir las imágenes de cada cluster
for i in range(0,10):
    indicador = np.where(resultado==i)[0] 
    Cantidad = indicador.shape[0]      
    altura = np.floor(Cantidad/10.)     
    plt.figure(figsize=(10,10))
    for j in range(0, Cantidad):
        plt.subplot(altura+1, 10, j+1)
        imagen = X[indicador[j], ]
        imagen = imagen.reshape(8, 8)
        plt.imshow(imagen, cmap=plt.cm.gray)
        plt.axis('off')
    plt.suptitle("Cluster N°{}: {} imágenes".format(i,Cantidad),size=15)
    plt.show()
 

"""
Hay algunos clusters que con el método de Kmeans quedaron medio feos y no separan bien las clases. Hay
algunos grupos en los que visiblemente aparecen 2, 3 y en algú caso hasta 4 dígitos diferenetes. Esto no
me serviría para hacer un reconocimiento por imágenes de diferentes dígitos. Entonces va a haber que probar
otro/s modelo/s de clusterización y/o otro/s parametro/s dentor del mismo modelo Kmeans. Entonces antes que nada,
necesitaremos una función que replique las pruebas del último bloque de código escrito, pero adaptado a otros
modelos.
"""


#Definimos la función de impresión de clusters
def cluster_imagen(result,XX):
    for i in range(0,10):
        indicador = np.where(result==i)[0] 
        Cantidad = indicador.shape[0]      
        altura = np.floor(Cantidad/10.)     
        plt.figure(figsize=(10,10))
        for j in range(0, Cantidad):
            plt.subplot(altura+1, 10, j+1)
            imagen = XX[indicador[j], ]
            imagen = imagen.reshape(8, 8)
            plt.imshow(imagen, cmap=plt.cm.gray)
            plt.axis('off')
        plt.suptitle("Cluster N°{}: {} imágenes".format(i,Cantidad),size=15)
        plt.show()

#Probemos la función con el único modelo que tenemos hasta ahora
cluster_imagen(resultado,X)

#Generamos ahora otro modelo de Kmeans con un método de inicialización distinta
modelo2=KMeans(n_clusters=10,init='k-means++').fit(X) #Armado del modelo
resultado2=modelo2.predict(X) #Predicción
cluster_imagen(resultado2,X) #Bastante mejor!!!

#Generamos un nuevo modelo, pero ahora con una técnica diferente
modelo3= AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
resultado3=modelo3.fit_predict(X)
cluster_imagen(resultado3,X) #Horrible. Casi que no separa en grupos. Logro encontrar sólo 2 clusters.


"""
A simple vista, de los 3 modelos de clusterización generados, el que más nítidamente separa los grupos en dígitos
únicos, es el K-means con método de inicialización k-means++.
"""