"""
========================
SaveImage
========================

Store an image.

"""
print(__doc__)
 

# OpenCV es una librería que se usa para realizar procesamiento de imágenes.
import cv2
import numpy as np

# Pickle son unas rutinas para serializar datos de descriptores de imágenes (para guardarlos en un archivo)
import pickle

# Sys es la librería para acceder a utilidades del sistema.
import sys

# Con esto chequeo si hay parámetros en la línea de comando.  El primero parametro es siempre el nombre del programa.
if (len(sys.argv)<2):
	print ("Descriptor filename should be provided.")
	quit()

# Instruye a OpenCV a que acceda a la primer cámara disponible.
cap = cv2.VideoCapture(0)


print ("Connecting..")

temp_array = []
gray = None

file = sys.argv[1]


# Se repite el código anterior, pero ahora una sola vez, como si fuera una foto, se sacan los descriptores
# y se guardan.
for i in range(1,20):
   # Capture frame-by-frame
   ret, frame = cap.read()

   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

   cv2.imshow("Computer Eye", frame)

   cv2.imwrite(file + f"{i-1}.png", gray)

   if cv2.waitKey(1) & 0xFF == ord('q'):
      break

# Guardo los descriptores (entre 500 y 2000 puede haber). Cada descriptor es un feature que voy a utilizar
# posteriormente para clasificarlos.
print ('Done.')


#When everything done, release the capture
cap.release()
cv2.destroyAllWindows()