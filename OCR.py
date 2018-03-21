import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn import svm

data = datasets.load_digits()
#clf = DecisionTreeClassifier()
clf = svm.SVC(gamma=0.001,C=100.)
clf.fit(data.data[0:1000],data.target[0:1000])
print(clf.predict([data.data[5]]))
'''
xtrain = data[0:1000,1:]
train_label=data[0:1000,0]'''
print(type(data))
plt.imshow(data.images[],cmap="Blues",interpolation="nearest")
#clf.fit(xtrain,train_label)
test = cv2.imread("data/0.jpg")
test = cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)
#print(clf.predict(test))

#p = clf.predict()
