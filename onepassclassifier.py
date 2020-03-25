"""
==================
OnePassClassifier
==================

This toy sample get two files of image descriptors (Akaze or whatever you want), build two classes with them and try to classify them with different methods.


"""
print(__doc__)

import cv2
import numpy as np

import pickle

from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

import sys

if (len(sys.argv)<3):
	print ("Descriptors filename should be provided (2 files).  These files should have been generated with onpassfeatureimages.py.")
	quit()


# These two functions help me to serialize and save to disk the features.
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

#Retrieve Keypoint Features for class 1 (the first picture) 1 and 3 are the same.
file1 = sys.argv[1]
keypoints_database = pickle.load( open( file1, "rb" ) )
kp1, desc1 = unpickle_keypoints(keypoints_database[0])
print("Found keypoints: {}, descriptors: {}".format(len(kp1), desc1.shape))
print ( desc1[1,:])  # There are many 61 lentgh vectors.

#Retrieve Keypoint Features for Class 2
file2 = sys.argv[2]
keypoints_database = pickle.load( open( file2, "rb" ) )
kp2, desc2 = unpickle_keypoints(keypoints_database[0])
print("Found keypoints: {}, descriptors: {}".format(len(kp2), desc2.shape))
print ( desc2[1,:])  # There are many 61 lentgh vectors.

#desc1=desc1[0:100,:]
#desc2=desc2[0:100,:]


# Los descriptores son matrices.  La primera dimension 0 representa la cantidad de descriptores, en 
# tanto que la segunda 1 es la dimension de cada uno de los features.
featuresize = desc1.shape[1]


# This is X, and y.  X being the data samples, and y their lables (0 or 1)
# Los descriptores de las dos imágenes se concatenan en uno solo tensor, y se componen los 
# labels a sabiendas de donde viene cada uno.  
featuredata = np.concatenate ((desc1,desc2))
featurelabels = np.concatenate( (np.zeros(desc1.shape[0]),(np.zeros(desc2.shape[0])+1) )  )

# Preprocessing: esto es requerido para que funcionen mejor los clasificadores porque los descriptores
# tienen algunas dimensiones con valores muy altos y otras con valores muy pequeños.
from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler(feature_range=(-1,1)).fit(featuredata)
featuredata = scaling.transform(featuredata)

# This is the boundary where the data will be divided into training and testing.
boundary = int(featuredata.shape[0]/2.0)

print ('Boundary %d:' % boundary)

# Reshape and shuffle the features
reorder = np.random.permutation(featuredata.shape[0])
#reorder = np.asarray([x for x in range(0,featuredata.shape[0])])

# You can uncomment this code to shuffle all the labels and see what happens with the classification.
# Spolier, it should be like random guessing.
#featurelabels = featurelabels[np.random.permutation(featuredata.shape[0])]

# Parto el dataset en 2.  Uso boundary para marcar la frontera de división (la mitad)
# Pueden intentar cambiar este valor para asignar más info al entrenamiento o al testing.
trainingdata = featuredata[reorder[0:boundary]]
traininglabels = featurelabels[reorder[0:boundary]]

# Reorder is a list of indices, from 0 to the length of the features
testdata = featuredata[reorder[boundary+1:featuredata.shape[0]]]
testlabels = featurelabels[reorder[boundary+1:featuredata.shape[0]]]

print ('Training Dataset Size %d,%d' % (trainingdata.shape))
print ('Test Dataset Size %d,%d' % (testdata.shape))

# Primero usamos SVM con un kernel lineal.  Hago aca una clasificación directa
# Fit construye el modelo.
clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(trainingdata,traininglabels)

print("Done with fitting...")

# Se hace la predicción de los labels en base a los features de test.
predlabels = clf.predict(testdata)
C = confusion_matrix(testlabels, predlabels)
acc = (float(C[0,0])+float(C[1,1])) / ( testdata.shape[0])
print ('SVM Feature Dim: %d Accuracy: %f' % (featuresize,acc))
print(C)

target_names = ['Class1', 'Class2']
report = classification_report(testlabels, predlabels, target_names=target_names)
print(report)

# ==== A partir de aca se construyen otros modelos y todos se comparan entre sí mediante las curvas ROC
# Para realizar la comparación con las curvas ROC necesito que la predicción retorne un valor de probabilidad
# en vez de retornar directamente el label estimado de la clase para cada uno de los vectores features.

# Use SVM but instead of guessing the class (0 or 1), configure it to output the
# probability value to belong to each class (this is needed to calculate ROC curves)
clf = svm.SVC(kernel='linear', C = 1.0, probability=True)
clf.fit(trainingdata,traininglabels)

ns_probs = [0 for _ in range(len(testlabels))]

# kNN
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(trainingdata,traininglabels)

# LogReg
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(trainingdata,traininglabels)

# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
ldaf = LinearDiscriminantAnalysis()
ldaf.fit(trainingdata,traininglabels)

# predict probabilities
sr_probs = clf.predict_proba(testdata)
# keep probabilities for the positive outcome only
sr_probs = sr_probs[:, 1]


# predict probabilities
kr_probs = model.predict_proba(testdata)
# keep probabilities for the positive outcome only
kr_probs = kr_probs[:, 1]


# predict probabilities
lr_probs = logisticRegr.predict_proba(testdata)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]

# predict probabilities
dr_probs = ldaf.predict_proba(testdata)
# keep probabilities for the positive outcome only
dr_probs = dr_probs[:, 1]


# calculate scores
ns_auc = roc_auc_score(testlabels, ns_probs)
sr_auc = roc_auc_score(testlabels, sr_probs)
kr_auc = roc_auc_score(testlabels, kr_probs)
lr_auc = roc_auc_score(testlabels, lr_probs)
dr_auc = roc_auc_score(testlabels, dr_probs)


# summarize scores
print('Trivial: ROC AUC=%.3f' % (ns_auc))
print('SVM: ROC AUC=%.3f' % (sr_auc))
print('kNN: ROC AUC=%.3f' % (kr_auc))
print('LogReg: ROC AUC=%.3f' % (lr_auc))
print('LDA: ROC AUC=%.3f' % (dr_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(testlabels, ns_probs)
sr_fpr, sr_tpr, _ = roc_curve(testlabels, sr_probs)
kr_fpr, kr_tpr, _ = roc_curve(testlabels, kr_probs)
lr_fpr, lr_tpr, _ = roc_curve(testlabels, lr_probs)
dr_fpr, dr_tpr, _ = roc_curve(testlabels, dr_probs)


pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Trivial')
# plot the roc curve for the model
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Trivial')
pyplot.plot(sr_fpr, sr_tpr, marker='.', label='SVM')
# plot the roc curve for the model
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Trivial')
pyplot.plot(sr_fpr, sr_tpr, marker='.', label='SVM')
pyplot.plot(kr_fpr, kr_tpr, marker='.', label='kNN')
# plot the roc curve for the model
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Trivial')
pyplot.plot(sr_fpr, sr_tpr, marker='.', label='SVM')
pyplot.plot(kr_fpr, kr_tpr, marker='.', label='kNN')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='LogReg')
# plot the roc curve for the model
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Trivial')
pyplot.plot(sr_fpr, sr_tpr, marker='.', label='SVM')
pyplot.plot(kr_fpr, kr_tpr, marker='.', label='kNN')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='LogReg')
pyplot.plot(dr_fpr, dr_tpr, marker='.', label='LDA')
# plot the roc curve for the model
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

# Cross Validation ==========================
from sklearn.model_selection import KFold

# La idea de la convalidación cruzada, es partir el conjunto de datos muchas veces, de maneras distintas, y 
# probar la clasificación sobre el conjunto de test.  Luego se promedia todo, para ver cuán bien con pocos datos
# de entrenamiento, se puede predecir los valores de testeo. Con esto se obtiene un valor de accuracy menos sesgado
# y que muestra mejor el poder de generalización de la clasificación.  En este caso se está haciendo con SVM.
avgaccuracy = []
kf = KFold(n_splits=10)
for train, test in kf.split(featuredata):
    trainingdata = featuredata[train]
    traininglabels = featurelabels[train]

    # Reorder is a list of indices, from 0 to the length of the features
    testdata = featuredata[test]
    testlabels = featurelabels[test]

    clf = svm.SVC(kernel='linear', C = 1.0)
    clf.fit(trainingdata,traininglabels)

    predlabels = clf.predict(testdata)
    C = confusion_matrix(testlabels, predlabels)

    if (C.shape == (2,2)):
        acc = (float(C[0,0])+float(C[1,1])) / ( testdata.shape[0])
    else:
        acc = (float(C[0,0])) / ( testdata.shape[0])
    
    avgaccuracy.append(acc*1.0)

acc = sum(avgaccuracy)/len(avgaccuracy)
print ('SVM K-Fold Cross Validated Feature Dim: %d Accuracy: %f' % (featuresize,acc))