
"""
========================
OnePassImage Classifier
========================

Load a previous stored model and classify a camera image in real time using the AKAZE descriptors.
The classification is by voting.  The model is a Random Forest Classifier.

"""
print(__doc__)
 

import cv2
import numpy as np

import pickle

import sys

if (len(sys.argv)<2):
	print ("Descriptor filename should be provided.")
	quit()

# Access the first camera....
cap = cv2.VideoCapture(0)


print ("Connecting..")

for i in range(1,2000):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Estas instrucciones sirven para invertir la imagen.
    #frame = cv2.flip(frame,0)
    ##frame = cv2.flip(frame,1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite('01.png', gray)

    #gray = frame;

    # Pick AKAZE keypoints and descriptors.
    detector = cv2.AKAZE_create()
    (kps, descs) = detector.detectAndCompute(gray, None)
    #print("keypoints: {}, descriptors: {}".format(len(kps), descs.shape))


    # Draw them on the screen.
    cv2.drawKeypoints(frame, kps, frame, (0, 255, 0))

    cv2.imshow("Computer Eye", frame)

    # Si se pone el foco en la pantalla, y se apreta la letra q se cierra.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


temp_array = []
gray = None
descs = None

# Do just one shot to store the descriptors.
for i in range(1,2):
   # Capture frame-by-frame
   ret, frame = cap.read()

   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

   detector = cv2.AKAZE_create()
   (kps, descs) = detector.detectAndCompute(gray, None)
   print("keypoints: {}, descriptors: {}".format(len(kps), descs.shape))

   # draw the keypoints and show the output image
   cv2.drawKeypoints(frame, kps, frame, (0, 255, 0))

   cv2.imshow("Computer Eye", frame)


   if cv2.waitKey(1) & 0xFF == ord('q'):
      break

# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix

import joblib
# Load the model from the stored file
frs = joblib.load(sys.args[1])

testdata = descs

from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler(feature_range=(-1,1)).fit(testdata)
testdata = scaling.transform(testdata)

# predict probabilities
rf_probs = frs.predict_proba(testdata)
# keep probabilities for the positive outcome only
rf_probs = rf_probs[:, 1]

predlabels = frs.predict(testdata)

print(predlabels)


print(f'Class 1: {predlabels[predlabels == 0].shape[0]}')
print(f'Class 2: {predlabels[predlabels == 1].shape[0]}')

