import numpy as np
import imutils
from sklearn import svm
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os

data = []
target = []
def load_images(folder):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            img = cv2.resize(img,(50,50),interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            _,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
            data.append(thresh)
            target.append(int(folder.split('\\')[1]))

for i in range(1,10):
    load_images("data\\"+str(i))
data = np.array(data)
data2 = []
target = np.array(target)
for i in data:
    data2.append(*i.reshape(1,2500))
data2 = np.array(data2)
print(data2.shape)
clf = svm.SVC(gamma = 0.001)
clf.fit(data2,target)
print(type(int(*clf.predict([data2[185]]))))

cv2.imshow('ans',data[185])
cv2.waitKey(0)
