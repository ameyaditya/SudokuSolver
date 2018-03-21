from sklearn import datasets
digits=datasets.load_digits()
x=digits.data
y=digits.target
import matplotlib.pyplot as plt
import cv2
from sklearn.neural_network import MLPClassifier

xtrain=x[:100000]
#xtest=x[-500:]
ytrain=y[:100000]
#ytest=y[-500:]
imag = cv2.imread("data/264.jpg")
imag = cv2.cvtColor(imag,cv2.COLOR_BGR2GRAY)
cv2.imshow("img",imag)
cv2.waitKey(0)
imag = imag.reshape(1,64)
#print(xtest.shape)
#print(xtest[[1]].shape)
#plt.imshow(xtest[[1]],cmap="Blues",interpolation="nearest")
#print(len(xtrain))
#print(len(xtest))
#print(len(ytrain))
#print(len(ytest))
nn=MLPClassifier()
nn.fit(xtrain,ytrain)
print("predicted value")
print(nn.predict(imag))
#img=digits.images
#img=img[-500:]
#plt.imshow(img[1],cmap="Blues",interpolation="nearest")
#plt.show()
