"""
========================
OnePassFeatureExtraxtion
========================

OpenCV basic code.
Saving/Storing keypoints https://isotope11.com/blog/storing-surf-sift-orb-keypoints-using-opencv-in-python


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

def pickle_keypoints(keypoints, descriptors):
    i = 0
    temp_array = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
        point.class_id, descriptors[i])     
        i+=1
        temp_array.append(temp)
    return temp_array

def unpickle_keypoints(array):
    keypoints = []
    descriptors = []
    for point in array:
        temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors)

# Instruye a OpenCV a que acceda a la primer cámara disponible.
cap = cv2.VideoCapture(0)


print ("Connecting..")

for i in range(1,10):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Estas instrucciones sirven para invertir la imagen.
    #frame = cv2.flip(frame,0)
    ##frame = cv2.flip(frame,1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite('01.png', gray)

    #gray = frame;

    # Se calculan los descriptores de AKAZE.  Puntos de la imágen destacados que sirven para discriminar.
    detector = cv2.AKAZE_create()
    (kps, descs) = detector.detectAndCompute(gray, None)
    print("keypoints: {}, descriptors: {}".format(len(kps), descs.shape))


    # Se dibujan los puntos en la imagen.
    cv2.drawKeypoints(frame, kps, frame, (0, 255, 0))

    cv2.imshow("Computer Eye", frame)

    # Si se pone el foco en la pantalla, y se apreta la letra q se cierra.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


temp_array = []

# Se repite el código anterior, pero ahora una sola vez, como si fuera una foto, se sacan los descriptores
# y se guardan.
for i in range(1,2):
   # Capture frame-by-frame
   ret, frame = cap.read()

   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

   detector = cv2.AKAZE_create()
   (kps, descs) = detector.detectAndCompute(gray, None)
   print("keypoints: {}, descriptors: {}".format(len(kps), descs.shape))

   #Store and Retrieve keypoint features
   temp = pickle_keypoints(kps, descs)
   temp_array.append(temp)

   # draw the keypoints and show the output image
   cv2.drawKeypoints(frame, kps, frame, (0, 255, 0))

   cv2.imshow("Computer Eye", frame)


   if cv2.waitKey(1) & 0xFF == ord('q'):
      break

# Guardo los descriptores (entre 500 y 2000 puede haber). Cada descriptor es un feature que voy a utilizar
# posteriormente para clasificarlos.
print ('Done.')
file = sys.argv[1]
pickle.dump(temp_array, open(file, "wb"))

#When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


