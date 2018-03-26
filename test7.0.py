from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import numpy as np
import cv2
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
'''
digits=datasets.load_digits()
x=digits.data
y=digits.target
xtrain=x[:-500]
xtest=x[-500:]
ytrain=y[:-500]
ytest=y[-500:]
nn=MLPClassifier(hidden_layer_sizes=(30,30,30),activation='logistic',tol=0.001)
nn.fit(xtrain,ytrain)
'''
i = 0
image = cv2.imread("sudoku.jpg")
'''
ratio = image[0]/300.0
print(ratio)
orig = image.copy()
image = imutils.resize(image,height=300)
'''
#blurred = cv2.pyrMeanShiftFiltering(image,31,91)
blurred = cv2.medianBlur(image,9)
gray = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)
#blurred = cv2.medianBlur(gray,5)
th3 = cv2.adaptiveThreshold(gray,200,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,25,3)
edged = cv2.Canny(th3,100,200)
_,contours,_ = cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key=cv2.contourArea,reverse=True)[0:10]
screenCnt = None
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c,0.02*peri, True)
    '''
    x,y,h,w = cv2.boundingRect(approx)
    ar = w/float(h)
    print(ar)
    '''
    if len(approx) == 4:
        #if ar>=0.95 and ar<=1.05:
        screenCnt = approx
        break
pts = screenCnt.reshape(4,2)
rect = np.zeros((4,2), dtype='float32')
s=pts.sum(axis=1)
rect[0] = pts[np.argmin(s)]
rect[2] = pts[np.argmax(s)]
diff = np.diff(pts, axis=1)
rect[1] = pts[np.argmin(diff)]
rect[3] = pts[np.argmax(diff)]
#rect *= ratio


(tl, tr, br, bl) = rect
widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

# ...and now for the height of our new image
heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

# take the maximum of the width and height values to reach
# our final dimensions
maxWidth = max(int(widthA), int(widthB))
maxHeight = max(int(heightA), int(heightB))

# construct our destination points which will be used to
# map the screen to a top-down, "birds eye" view
dst = np.array([
	[0, 0],
	[maxWidth - 1, 0],
	[maxWidth - 1, maxHeight - 1],
	[0, maxHeight - 1]], dtype = "float32")

# calculate the perspective transform matrix and warp
# the perspective to grab the screen
M = cv2.getPerspectiveTransform(rect, dst)
warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

res = cv2.resize(warp,(600,600), interpolation= cv2.INTER_AREA)

for i in range(0,600-(600//9),600//9):
    for j in range(0,600-(600//9),600//9):
        roi = res[i:i+600//9,j:j+600//9]
        grey = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(grey,150,255,cv2.THRESH_BINARY)
        _,cnts,_=cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if(len(cnts) > 1):
            cnts = sorted(cnts, key = cv2.contourArea, reverse=True)[1]
            area = cv2.contourArea(cnts,True)
            if area>100 and area<1000:
                M = cv2.moments(cnts)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                cropped = thresh[cy-22:cy+22,cx-20:cx+20]
                #px8 = cv2.resize(cropped,(8,8),interpolation=cv2.INTER_AREA)
                _,inverted_img = cv2.threshold(cropped,180,255,cv2.THRESH_BINARY_INV)
                #print(i/(600//9),j/(60//9))
                #input()
                #print(nn.predict(px8))
                #cv2.imshow("px2",inverted_img)
                #cv2.imshow("px",px8)
                cv2.imwrite("data2/"+str(i)+".jpg",inverted_img)
                i+=1
                cv2.waitKey(0)

cv2.drawContours(image,[screenCnt],-1,(0,0,255),3)
cv2.imshow("image",image)
cv2.imshow("warped",res)
cv2.waitKey(0)
cv2.destroyAllWindows()
