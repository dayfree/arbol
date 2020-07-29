from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn import tree

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from collections import defaultdict
from random import random, shuffle
import numpy as np
import mahotas
import cv2
import os



##nuevas importaciones



fixed_size = tuple((76, 76)) # las muestras son 76 por 76 pixeles
count = 75 # how many to take per class
bins = 8

def fd_hu_moments(image): # feature-descriptor-1: Hu Moments
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_haralick(image): # feature-descriptor-2: Haralick Texture
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

def fd_histogram(image, mask=None): # feature-descriptor-3: Color Histogram
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

labels = []
features = []
targets = ['green', 'yellow', 'red', 'leafless']
counts = defaultdict(int)

for label in targets:
    listing = list(os.scandir(f'samples//{label}'))
    shuffle(listing)
    for entry in listing:
        if entry.path.endswith('.png') and entry.is_file():
            if counts[label] < count:
                filename = entry.path
                image = cv2.imread(filename)
                image = cv2.resize(image, fixed_size)
                fv_hu_moments = fd_hu_moments(image)
                fv_haralick = fd_haralick(image)
                fv_histogram  = fd_histogram(image)
                global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
                labels.append(label)
                features.append(global_feature)
                counts[label] += 1

scaler = MinMaxScaler(feature_range = (0, 1)) # normalizador
rescaled_features = scaler.fit_transform(features) # normalizar caracteristicas

(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(rescaled_features),
                                                                  np.array(labels),
                                                                  test_size = 0.3) # 30 % para pruebas
clf = tree.DecisionTreeClassifier()
clf = clf.fit(trainData, trainLabels)

correct = 0
predict = []
for (data, label) in zip(testData, testLabels):
    assigned = clf.predict(data.reshape(1,-1))[0]
    print(label, assigned)
    predict.append(assigned)
    if label == assigned:
        correct += 1
print('#', 100 * correct / len(testLabels), '% correct')
print(confusion_matrix(testLabels, predict))
print(classification_report(testLabels, predict))