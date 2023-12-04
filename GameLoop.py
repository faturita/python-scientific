"""
========================
Basic OpenCV Game Loop
========================

"""
print(__doc__)
 

# OpenCV es una librería que se usa para realizar procesamiento de imágenes.
import cv2
import numpy as np

import os

# Pickle son unas rutinas para serializar datos de descriptores de imágenes (para guardarlos en un archivo)
import pickle

# Sys es la librería para acceder a utilidades del sistema.
import sys

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

timer = 0

# Loop game.
while True:
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
    #print("keypoints: {}, descriptors: {}".format(len(kps), descs.shape))

    #kps = kps[0:10]


    # Se dibujan los puntos en la imagen.
    cv2.drawKeypoints(frame, kps, frame, (0, 255, 0))

    print(kps[0].pt[0])


    # Calculo rapido el centro de masa de todos los puntos.
    X = 0
    Y = 0
    for kp in kps:
        X += kp.pt[0]
        Y += kp.pt[1]

    X /= len(kps)
    Y /= len(kps)

    X = int(X)
    Y = int(Y)

    height = 2
    width = 2


    # Dibujo la X
    cv2.line(frame, (X-width, Y-height), (X+width, Y+height), (0, 0, 255), 5)
    cv2.line(frame, (X-width, Y+height), (X+width, Y-height), (0, 0, 255), 5)

    # Esto se calcula en base a la identificacion de la carta x el modelo
    X_target, Y_target = 500, 500


    # Este es el punto donde se arma una finite-state-machine super simple 
    azimuth = 40
    X, Y

    timer += 1
    if azimuth > 10 and azimuth < 250:
        if timer > 50: 
            os.popen("say rotate")          # This works for Mac, you need to use any shell program to create text-to-speech
            timer = 0

    # ....... you get the point



    cv2.namedWindow("Computer Eye", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Computer Eye", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow("Computer Eye", frame)

    # Si se pone el foco en la pantalla, y se apreta la letra q se cierra.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Guardo los descriptores (entre 500 y 2000 puede haber). Cada descriptor es un feature que voy a utilizar
# posteriormente para clasificarlos.
print ('Done.')
#file = sys.argv[1]
#pickle.dump(temp_array, open(file, "wb"))

#When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


