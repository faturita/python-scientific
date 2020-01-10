#coding: latin-1
#
#
# One pass classifier...

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

#Retrieve Keypoint Features for class 1 (the first picture)
keypoints_database = pickle.load( open( "data/kd2.p", "rb" ) )
kp1, desc1 = unpickle_keypoints(keypoints_database[0])
print("Found keypoints: {}, descriptors: {}".format(len(kp1), desc1.shape))
print ( desc1[1,:])  # There are many 61 lentgh vectors.

#Retrieve Keypoint Features for Class 2
keypoints_database = pickle.load( open( "data/kd1.p", "rb" ) )
kp2, desc2 = unpickle_keypoints(keypoints_database[0])
print("Found keypoints: {}, descriptors: {}".format(len(kp2), desc2.shape))
print ( desc2[1,:])  # There are many 61 lentgh vectors.

#desc1=desc1[0:600,:]
#desc2=desc2[0:600,:]


# Featuresize is the dimension of the feature vectors.
featuresize = desc1.shape[1]


# This is X, and y.  X being the data samples, and y their lables (0 or 1)
featuredata = np.concatenate ((desc1,desc2))
featurelabels = np.concatenate( (np.zeros(desc1.shape[0]),(np.zeros(desc2.shape[0])+1) )  )

# Preprocessing
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

print("Done with fitting...")

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

# kNN
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(trainingdata,traininglabels)

# LogReg
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(trainingdata,traininglabels)

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


# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Trivial')
pyplot.plot(sr_fpr, sr_tpr, marker='.', label='SVM')
pyplot.plot(kr_fpr, kr_tpr, marker='.', label='kNN')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='LogReg')
pyplot.plot(dr_fpr, dr_tpr, marker='.', label='LDA')


# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

