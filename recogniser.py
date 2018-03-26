import numpy as np
import cv2
from sklearn import datasets
from sklearn import svm
from scipy import misc

digits = datasets.load_digits()
x,y = digits.data, digits.target

clf = svm.SVC(gamma= 0.0001)
#clf.fit(x,y)

img = cv2.imread("data2/67.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = cv2.resize(img,(8,8))

img = img.astype(digits.data.dtype)
cv2.imshow("px8",digits.images[1].reshape(8,8))
cv2.waitKey(0)

img = img.reshape(1,64)
clf.fit(x,y)
print(clf.predict(img))
