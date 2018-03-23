from sklearn import datasets
import matplotlib.pyplot as plt
import cv2
from sklearn.neural_network import MLPClassifier
import pandas as pd

digits=datasets.load_digits()
x=digits.data
y=digits.target
df = pd.read_csv("train.csv")
#print(df.loc[1:10000])
xtrain = df.iloc[0:10000,1:]
ytrain = df.iloc[0:10000,0]
#print(ytrain.head())
#xtrain=x[:100000]
#xtest=x[-500:]
#ytrain=y[:100000]
#ytest=y[-500:]
#print(digits.images[0])
#plt.imshow(digits.images[1],cmap='gray')
print(xtrain.shape)
print(ytrain.shape)
imag = cv2.imread("data/1.jpg")
imag = cv2.cvtColor(imag,cv2.COLOR_BGR2GRAY)


#_,imag = cv2.threshold(imag,150,255,cv2.THRESH_BINARY_INV)

#print(imag)
#plt.show()
#cv2.imshow("img",imag)
#cv2.waitKey(0)
#imag = imag.reshape(8,8)

#print(xtest.shape)
#print(xtest[[1]].shape)
#plt.imshow(xtest[[1]],cmap="Blues",interpolation="nearest")
#print(len(xtrain))
#print(len(xtest))
#print(len(ytrain))
#print(len(ytest))
#nn=MLPClassifier()
#nn.fit(xtrain,ytrain)
#print("predicted value")
#print(nn.predict(df.iloc[12345,1:]))
#img=digits.images
#img=img[-500:]
#plt.imshow(img[1],cmap="Blues",interpolation="nearest")
#plt.show()
