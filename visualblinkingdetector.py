'''
Based on code by    ROGGER ARICHUNA HERRERA CANONICO and 
                    JUAN MANUEL DONADIO and ideas provided by 
                    FERNANDO MARTÍN RABALLO AIME (https://github.com/Sabonrab/infovis).


@NOTE: Use me with the 'bot' environment.

'''
import urllib.request
import shutil
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2
import os
import urllib.request
import shutil
import sys

import bz2



# Parte 1 del codigo:
def eye_aspect_ratio(eye):
	# Computa la distancia euclidiana
	# Calcula la distancia entre los dos conjuntos de  puntos de referencia del ojo vertical
	# Punto de referencia del ojo vertical (x, y)-coordenadas
	DistA = dist.euclidean(eye[1], eye[5])
	DistB = dist.euclidean(eye[2], eye[4])

	# Computa la distancia euclidiana
	# Calcula la distancia entre  los puntos de referencia del ojo horizontal
	# Punto de referencia del ojo horizontal (x, y)-coordenadas
	DistC = dist.euclidean(eye[0], eye[3])

	# computa el eye aspect ratio (EAR)
	ear = (DistA + DistB) / (1.5 * DistC)

	# Devuelve el eye aspect ratio
	return ear


# Parsea los argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default="shape_predictor_68_face_landmarks.dat",
                help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="camera",
                help="path to input video file")
ap.add_argument("-t", "--threshold", type=float, default=0.27,
                help="threshold to determine closed eyes")
ap.add_argument("-f", "--frames", type=int, default=2,
                help="the number of consecutive frames the eye must be below the threshold")

# Se definen dos constantes, una para la relación de aspecto del ojo para indicar
# el parpadeo y luego una segunda constante para el número de
# Frames en que el ojo debe estar por debajo del umbral


def main():
	args = vars(ap.parse_args())
	EYE_AR_THRESH = args['threshold']
	EYE_AR_CONSEC_FRAMES = args['frames']

	# Initializa en cero los parpadeos
	COUNTER = 0
	TOTAL = 0

	# Download the file if it does not exist
	filename = 'shape_predictor_68_face_landmarks.dat'
    
	if not os.path.isfile(filename):
	
		url = 'https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2?raw=true'

		# Download the file from `url` and save it locally under `file_name`:
		with urllib.request.urlopen(url) as response, open(filename+'.bz2', 'wb') as out_file:
			shutil.copyfileobj(response, out_file)

		zipfile = bz2.BZ2File(filename+'.bz2') # open the file
		data = zipfile.read() # get the decompressed data
		open(filename, 'wb').write(data) # write a uncompressed file

	# Inicializa el predictor de rostros con la libreria dlib's face detector (HOG-based) y luego usa ==>
	# el predictor de puntos de referencia del rostro
	print("[INFO] cargando el predictor de puntos de referencia del rostro...")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(args["shape_predictor"])

	# Se seleccionan los indexes de los puntos de referencia del ojo derecho e izquierdo
	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

	# Inicializa el video
	print("[INFO] inicia video stream...")
	print("[INFO] presione s para salir...")
	if args['video'] == "camera":
		vs = VideoStream(src=0).start()
		fileStream = False
	else:
		vs = FileVideoStream(args["video"]).start()
		fileStream = True

	time.sleep(1.0)

	# ciclo de procesado ppal
	# Si se trata de un archivo de flujo de vídeo, entonces tenemos que comprobar si
	# Hay más cuadros dejados en el búfer para procesar

	# Parte 2 del codigo:
	while True:
		if fileStream and not vs.more():
			break

		# Agarrar el marco de la secuencia de archivo de vídeo  cambiar el tamaño
		# y convertirlo a escala de grises
		frame = vs.read()
		frame = imutils.resize(frame, width=550)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# detecta caras en la imagen en escala de grises
		rects = detector(gray, 0)
		# Parte 3 del codigo:
		# ciclo sobre las detecciones de la cara
		for rect in rects:
			# Determina las marcas faciales para la región de la cara, luego
			# Convierte el punto de referencia facial (x, y) a coordenada y se genera un array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			# Extrae las coordenadas de los ojos izquierdo y derecho y calcula
			# la relación de aspecto (AR) del ojo para ambos ojos
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)

			# Calcula la media AR (Average Ratio) Para ambos ojos
			ear = (leftEAR + rightEAR) / 2.0

			# hace la convex hull para los dos ojos
			# Se dibujan ambos ojos
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

			# Compruebe si la relación de aspecto del ojo está por debajo del parpadeo
			# Y si es así se incrementa el contador del marco intermitente
			if ear < EYE_AR_THRESH:
				COUNTER += 1

			# De lo contrario, la relación de aspecto del ojo no está por debajo del limite de parpadeo
			else:
				# Si los ojos estaban cerrados por un número suficiente de
				# Luego incrementar el número total de parpadeos
				if COUNTER >= EYE_AR_CONSEC_FRAMES:
					TOTAL += 1

				# Resetea el contador
				COUNTER = 0
			# Parte 4 del codigo:
			# Dibuja el número total de destellos en el marco junto con
			# la relación de aspecto calculada del ojo para el marco
			cv2.putText(frame, "Parpadeos: {}".format(TOTAL), (10, 30),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# Se muestra el frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# Parte 5 del codigo:
		#  si pulsa s se sale de la app y se rompe el ciclo
		if key == ord("s"):
			break

	# Se limpia el entorno
	cv2.destroyAllWindows()
	vs.stop()
if __name__ == '__main__':
    main()
