#coding: latin-1

# Run with ann virtual environment
# EPOC Emotiv file format https://www.researchgate.net/publication/332514530_EPOC_Emotiv_EEG_Basics

# OpemMP sometimes raises coredumps, try export KMP_DUPLICATE_LIB_OK=TRUE

import numpy as np

from struct import *

import sys, select

import platform
import socket
import gevent

import time
import datetime
import os

from scipy.fftpack import fft

import math

from scipy.signal import firwin, remez, kaiser_atten, kaiser_beta
from scipy.signal import butter, filtfilt, buttord

from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix

from scipy.signal import butter, lfilter

def isartifact(window, threshold=80):
    # Window is EEG Matrix

    awindow = np.asarray(window)
    ameans = np.asarray(  window   ).mean(0)
    signalaverage = ameans.tolist()
    athresholds = np.asarray([threshold]*len(signalaverage))

    #print awindow
    #print ameans
    #print athresholds

    # FIXME
    for t in range(0,len(window)):
        asample = (ameans+athresholds)-awindow[t]
        #print asample
        for c in range(0,asample.shape[0]):
            # while (ameans+athresholds)>(awindow)
            if asample[c]<0:
                return True


    return False

import matplotlib.pyplot as plt

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def psd(y):
    # Number of samplepoints
    N = 128
    # sample spacing
    T = 1.0 / 128.0
    # From 0 to N, N*T, 2 points.
    #x = np.linspace(0.0, 1.0, N)
    #y = 1*np.sin(10.0 * 2.0*np.pi*x) + 9*np.sin(20.0 * 2.0*np.pi*x)


    # Original Bandpass
    fs = 128.0
    fso2 = fs/2
    #Nd,wn = buttord(wp=[9/fso2,11/fso2], ws=[8/fso2,12/fso2],
    #   gpass=3.0, gstop=40.0)
    #b,a = butter(Nd,wn,'band')
    #y = filtfilt(b,a,y)

    y = butter_bandpass_filter(y, 8.0, 15.0, fs, order=6)


    yf = fft(y)
    #xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
    #import matplotlib.pyplot as plt
    #plt.plot(xf, 2.0/N * np.abs(yf[0:N/2]))
    #plt.axis((0,60,0,1))
    #plt.grid()
    #plt.show()

    return np.sum(np.abs(yf[0:int(N/2)]))


import time
import datetime
import os

class Packet():
    def init(self):
        self.O1 = 0
        self.O2 = 0
        self.gyro_x = 0
        self.gyro_y = 0


class OfflineHeadset:
    def __init__(self, subject,label,paradigm='Alfa'):
        # @TODO Need to parametrize this.
        # @NOTE Search for datasets on current "Data" directory
        self.basefilename = 'Data/%s/%s/e.%d.l.%d.dat'
        self.paradigm = paradigm
        self.readcounter = 0
        self.running = True
        self.label = label
        self.subject = subject
        self.fileindex = 0
        self.f = None

    def setup(self):
        pass

    def setupfile(self):
        self.datasetfile = self.basefilename % (self.subject,self.paradigm,self.fileindex,self.label)
        print (self.datasetfile)
        if os.path.isfile(self.datasetfile):
            if self.f:
                self.f.close()
            self.f = open(self.datasetfile,'r')
            return True
        else:
            return False

    def nextline(self):
        line = None
        if self.f:
            line = self.f.readline()
        if (not line):
            self.fileindex = self.fileindex + 1

            if self.setupfile():
                return self.nextline()
            else:
                return None
        else:
            return line

    def dequeue(self):
        line = self.nextline()
        if (line):
            data = line.split('\r\n')[0].split(' ')
            packet = Packet()
            packet.O1 = [float(data[7]),0]
            packet.O2 = [float(data[8]),0]
            packet.gyro_x = 0
            packet.gyro_y = 0

            self.readcounter = self.readcounter + 1
            return packet
        else:
            self.running = False
            return None


    def close(self):
        if (self.f):
            self.f.close()

# Segmentación de la serie de tiempo.
def process(headset):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
    log = open('data/biosensor-%s.dat' % st, 'w')
    #plotter = Plotter(500,4000,5000)
    print ("Starting BioProcessing Thread...")
    readcounter=0
    iterations=0

    N = 128

    window = []
    fullsignal = []
    awindow = None
    afullsignal = None
    features = []


    while headset.running:
        packet = headset.dequeue()
        interations=iterations+1
        if (packet != None):
            datapoint = [packet.O1[0], packet.O2[0]]

            #plotter.plotdata( [packet.gyro_x, packet.O2[0], packet.O1[0]])
            log.write( str(packet.gyro_x) + "\t" + str(packet.gyro_y) + "\n" )

            window.append( datapoint )


            # Este punto establece cuando se hace el corte, 
            # como se genera el feature y 
            # como se hace el desplazamiento de la ventana.
            # Este es el metodo de Welsh para EEG.
            if len(window)>=N:
                if not isartifact(window):
                    awindow = np.asarray( window )
                    fullsignal = fullsignal + window
                    afullsignal = np.asarray( fullsignal )

                    if (len(fullsignal) > 0):
                        awindow = awindow - afullsignal.mean(0)

                    o1 = psd(awindow[:,0])
                    o2 = psd(awindow[:,1])

                    print (o1, o2)

                    features.append( [o1, o2] )

                # Slide window
                window = window[int(N/2):N]
                #window = window[1:N]

            readcounter=readcounter+1

        if (readcounter==0 and iterations>50):
            headset.running = False
        gevent.sleep(0.001)

    log.close()

    return features

def reshapefeature(feature, featuresize):
    feature=feature[0:feature.shape[0]-(feature.shape[0]%featuresize)]
    feature = np.reshape( feature, (int(feature.shape[0]/int(featuresize/feature.shape[1])),featuresize) )

    return feature

def classify(afeatures1, afeatures2, featuresize):

    print ('Feature 1 Size %d,%d' % (afeatures1.shape))
    print ('Feature 2 Size %d,%d' % (afeatures2.shape))

    afeatures1 = reshapefeature(afeatures1, featuresize)
    afeatures2 = reshapefeature(afeatures2, featuresize)

    featuredata = np.concatenate ((afeatures1,afeatures2))
    featurelabels = np.concatenate( (np.zeros(afeatures1.shape[0]),(np.zeros(afeatures2.shape[0])+1) )  )

    boundary = int(featuredata.shape[0]/2.0)

    print ('Boundary %d:' % boundary)

    # Reshape and shuffle the features
    reorder = np.random.permutation(featuredata.shape[0])

    trainingdata = featuredata[reorder[0:boundary]]
    traininglabels = featurelabels[reorder[0:boundary]]

    testdata = featuredata[reorder[boundary+1:featuredata.shape[0]]]
    testlabels = featurelabels[reorder[boundary+1:featuredata.shape[0]]]

    print ('Training Dataset Size %d,%d' % (trainingdata.shape))
    print ('Test Dataset Size %d,%d' % (testdata.shape))


    clf = svm.SVC(kernel='linear', C = 1.0)
    clf.fit(trainingdata,traininglabels)


    predlabels = clf.predict(testdata)
    C = confusion_matrix(testlabels, predlabels)
    acc = (float(C[0,0])+float(C[1,1])) / ( testdata.shape[0])
    print ('SVM Feature Dim: %d Accuracy: %f' % (featuresize,acc))
    print(C)

    target_names = ['Open', 'Closed']
    report = classification_report(testlabels, predlabels, target_names=target_names)
    print(report)

    from sklearn.linear_model import LogisticRegression

    # all parameters not specified are set to their defaults
    logisticRegr = LogisticRegression()
    logisticRegr.fit(trainingdata,traininglabels)

    # Returns a NumPy Array
    # Predict for One Observation (image)
    predlabels = logisticRegr.predict(testdata)
   
    C = confusion_matrix(testlabels, predlabels)
    acc = (float(C[0,0])+float(C[1,1])) / ( testdata.shape[0])
    print ('LogReg Feature Dim: %d Accuracy: %f' % (featuresize,acc))
    print(C)

    target_names = ['Open', 'Closed']
    report = classification_report(testlabels, predlabels, target_names=target_names)
    print(report)

    from keras.models import Sequential
    from keras.layers import Dense

    model = Sequential([
        Dense(64, activation='tanh', input_shape=(trainingdata.shape[1],)),
        Dense(32, activation='tanh'),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    hist = model.fit(trainingdata, traininglabels,
          batch_size=10, epochs=1000*trainingdata.shape[1],verbose=0,
          validation_split=0.4)

    predlabels = model.predict(testdata)
    #print(predlabels)
    predlabels = predlabels.round()
    #print(predlabels)
    C = confusion_matrix(testlabels, predlabels)
    acc = (float(C[0,0])+float(C[1,1])) / ( testdata.shape[0])
    print ('Keras Feature Dim: %d Accuracy: %f' % (featuresize,acc))
    print(C)

    print(model.evaluate(testdata,testlabels))
    print ('Keras Model Accuracy: %f' % (model.evaluate(testdata,testlabels)[1]))

    target_names = ['Open', 'Closed']
    report = classification_report(testlabels, predlabels, target_names=target_names)
    print(report)

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()

    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    plt.show()


# Esto es lo que hace el método principal.

# Primero toma muestras de señales de tiempo de una personas.  Estas señales corresponden a dos experiemntos, 
# donde la persona durante un tiempo estaba con los ojos cerrados, y luego con los ojos abiertos.
# Eso dispara un cambio en las señales occipitales, en O1 y O2 que son dos canales.  Ese cambio se manifiesta
# como un aumento de la potencia de 10 Hz cuando la persona tiene los ojos cerrados.
def featureextractor():
    # Get features from label 1.
    headset = OfflineHeadset('Subject',1,'Alfa')
    features1 = process(headset)
    headset.close()
    # Get features from label 2
    headset = OfflineHeadset('Subject',2,'Alfa')
    features2 = process(headset)
    headset.close()

    # En este punto se tienen una secuencia de features bidimensionales.  El PSD de O1 y O2 durante una ventana de tiempo.
    afeatures1 = np.asarray(features1)
    afeatures2 = np.asarray(features2)

    print (afeatures1.mean(0))
    print (afeatures2.mean(0))

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(afeatures1[:,0], afeatures1[:,1], s=10, c='b', marker="x", label='Open')
    ax1.scatter(afeatures2[:,0], afeatures2[:,1], s=10, c='r', marker="o", label='Closed')
    plt.xlabel('PSD O2')
    plt.ylabel('PSD O1')
    plt.legend(loc='upper left')
    plt.show()

    # Group time features in tuples, 4-tuples and 8-tuples and classify them
    classify(afeatures1, afeatures2,2)
    classify(afeatures1, afeatures2,4)
    classify(afeatures1, afeatures2,8)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(afeatures1[:,0], afeatures1[:,1], s=10, c='b', marker="x", label='Open')
    ax1.scatter(afeatures2[:,0], afeatures2[:,1], s=10, c='r', marker="o", label='Closed')
    plt.xlabel('PSD O2')
    plt.ylabel('PSD O1')
    plt.legend(loc='upper left')
    plt.show()


# Este If de python, sirve cuando un programa funciona como una libreria, por lo que no tiene código que se ejecute
# que no esté en el bloque global (sin indentación).  En esos casos este if sirve para indicar que se tiene 
# que ejecutar cuando a este .py se lo ejecuta de manera directa.
if __name__ == "__main__":

    featureextractor()
