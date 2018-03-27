import numpy as np
import cv2
from sklearn import datasets
from sklearn import svm
from scipy import misc
import matplotlib.pyplot as plt

digits = datasets.load_digits()
tar = digits.target
truth_table = []
for i in tar:
    if i == 0:
        truth_table.append(False)
    else:
        truth_table.append(True)
mask = np.array(truth_table)
x = digits.data[mask]
y = digits.target[mask]
#x,y = digits.data, digits.target
print(len(y))
clf = svm.SVC(gamma= 0.0001)
#clf.fit(x,y)

img = misc.imread("data2/198.jpg")
img = misc.imresize(img,(8,8))
img = img.astype(digits.images.dtype)
img = misc.bytescale(img,high=16, low=0)
print(img)
img_test = []
for i in img:
    for j in i:
        img_test.append(j/1.0)
#print(np.array(img_test).reshape(8,8))

#img = img.astype(digits.data.dtype)
plt.imshow(img,cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()

#cv2.imshow("px8",digits.images[0])
#cv2.waitKey(0)
#print(digits.images[0])
#img = img.reshape(1,64)
clf.fit(x,y)
print(clf.predict([img_test]))
