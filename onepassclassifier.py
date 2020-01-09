#coding: latin-1

# Saving/Storing keypoints https://isotope11.com/blog/storing-surf-sift-orb-keypoints-using-opencv-in-python
#
#

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

# These two functions help me to serialize and save to disk the features.
def pickle_keypoints(keypoints, descriptors):
    i = 0
    temp_array = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
        point.class_id, descriptors[i])     
        ++i
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

#Retrieve Keypoint Features for class 1 (the first picture)
keypoints_database = pickle.load( open( "data/kd1.p", "rb" ) )
kp1, desc1 = unpickle_keypoints(keypoints_database[0])
print("Found keypoints: {}, descriptors: {}".format(len(kp1), desc1.shape))
print ( desc1[1,:])  # There are many 61 lentgh vectors.

#Retrieve Keypoint Features for Class 2
keypoints_database = pickle.load( open( "data/kd2.p", "rb" ) )
kp1, desc2 = unpickle_keypoints(keypoints_database[0])
print("Found keypoints: {}, descriptors: {}".format(len(kp1), desc2.shape))
print ( desc1[1,:])  # There are many 61 lentgh vectors.

# Featuresize is the dimension of the feature vectors.
featuresize = desc1.shape[1]

# This is X, and y.  X being the data samples, and y their lables (0 or 1)
featuredata = np.concatenate ((desc1,desc2))
featurelabels = np.concatenate( (np.zeros(desc1.shape[0]),(np.zeros(desc2.shape[0])+1) )  )

# This is the boundary where the data will be divided into training and testing.
boundary = int(featuredata.shape[0]/2.0)

print ('Boundary %d:' % boundary)

# Reshape and shuffle the features
reorder = np.random.permutation(featuredata.shape[0])

# You can uncomment this code to shuffle all the labels and see what happens with the classification.
# Spolier, it should be like random guessing.
#featurelabels = featurelabels[np.random.permutation(featuredata.shape[0])]

trainingdata = featuredata[reorder[0:boundary]]
traininglabels = featurelabels[reorder[0:boundary]]

# Reorder is a list of indices, from 0 to the length of the features
testdata = featuredata[reorder[boundary+1:featuredata.shape[0]]]
testlabels = featurelabels[reorder[boundary+1:featuredata.shape[0]]]

print ('Training Dataset Size %d,%d' % (trainingdata.shape))
print ('Test Dataset Size %d,%d' % (testdata.shape))

# Classify using SVM lineal.
clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(trainingdata,traininglabels)

predlabels = clf.predict(testdata)
C = confusion_matrix(testlabels, predlabels)
acc = (float(C[0,0])+float(C[1,1])) / ( testdata.shape[0])
print ('SVM Feature Dim: %d Accuracy: %f' % (featuresize,acc))
print(C)

target_names = ['Class1', 'Class2']
report = classification_report(testlabels, predlabels, target_names=target_names)
print(report)

# Use SVM but instead of guessing the class (0 or 1), configure it to output the
# probability value to belong to each class (this is needed to calculate ROC curves)
clf = svm.SVC(kernel='linear', C = 1.0, probability=True)
clf.fit(trainingdata,traininglabels)

ns_probs = [0 for _ in range(len(testlabels))]

# predict probabilities
lr_probs = clf.predict_proba(testdata)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(testlabels, ns_probs)
lr_auc = roc_auc_score(testlabels, lr_probs)
# summarize scores
print('Trivial: ROC AUC=%.3f' % (ns_auc))
print('SVM: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(testlabels, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(testlabels, lr_probs)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Trivial')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='SVM')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

