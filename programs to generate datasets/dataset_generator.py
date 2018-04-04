import numpy as np
import cv2
import imutils

#first 35 lines are same as from the previous program
#read the image of sudoku
x = 300
a1 = 0
b1 = 0
image = cv2.imread("sudo4.jpg")

#process the image for contour recognition
#run the canny edge to detect edges
blurred = cv2.medianBlur(image,7)
gray = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)
th3 = cv2.adaptiveThreshold(gray,150,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,31,3)
edged = cv2.Canny(th3,50,100)

#after applying the canny edge function
#find the contours(closed loops) in the image(edged)
_,contours,_ = cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

#sort the contours based on their area
#assuming the area of the sudoku will the largest
#label1: updating contour value
contours = sorted(contours,key=cv2.contourArea,reverse=True)[0:10]

#storing the particular contour values in a this variable
puzzle = None
#extracting the coordinates of the actual sudoku puzzle
for c  in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c,0.02*peri, True)

    #if the approximated polygon has 4 sides, it is our sudoku puzzle
    if len(approx) == 4:
        puzzle = approx
        break

#we have till now found our sudoku puzzle in the image
#to isolate only our puzzle, and removing extra stuff, we write the following code

#puzzle is a 3D array of size (4,1,2) we need to reshape it into (4,2)
pts = puzzle.reshape(4,2)

#creating a temporary array of zeros to store coordinates in order
rect = np.zeros((4,2), dtype='float32')

#storing the sum along all the rows
s = pts.sum(axis =1)
#the minimum sum value is the top left point and max sum value is the bottom right point
rect[0] = pts[np.argmin(s)]
rect[2] = pts[np.argmax(s)]

#storing the difference across each row
diff = np.diff(pts, axis=1)

#the minimum difference will be of the top right corner and maximum difference is of the bottom left
rect[1] = pts[np.argmin(diff)]
rect[3] = pts[np.argmax(diff)]

#unpacking the array
(tl,tr,br,bl) = rect

#finding the width of two horizontal lines of the image
#using the formula, length = sqrt((X2 - X1)^2 + (Y2 - Y1)^2)
widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

#find the two heights and storing them
#using the formula, length = sqrt((X2 - X1)^2 + (Y2 - Y1)^2)
heightA = np.sqrt(((bl[0] - tl[0]) ** 2) + ((bl[1] - tl[1]) ** 2))
heightB = np.sqrt(((br[0] - tr[0]) ** 2) + ((br[1] - tr[1]) ** 2))

#getting the maximum values out of these to get our final dimensions
maxWidth = max(int(widthA), int(widthB))
maxHeight = max(int(heightA), int(heightB))

#construct our destination points which will be used to
#map the screen to top-down, "bird eye" view
dst = np.array([
[0,0],
[maxWidth - 1, 0],
[maxWidth - 1, maxHeight - 1],
[0, maxHeight - 1]], dtype="float32")

#calculate the perspective transform matrix and warp
#the perspective to grab the screen
M = cv2.getPerspectiveTransform(rect,dst)
warp = cv2.warpPerspective(image, M, (maxWidth,maxHeight))

#resizing our sudoku to a 630x630 image
res = cv2.resize(warp,(630,630), interpolation=cv2.INTER_AREA)
res2 = res.copy()
#cv2.imshow("Sudoku", res)
#cv2.waitKey(0)

#dividing our region of interest into squares of sizes 600//9,600//9
#we are able to look into each cell of the sudoku
for i in range(0,561,70):
    for j in range(0,561,70):
        #creating our ROI and moving through each cell
        roi = res[i:i+70,j:j+70]
        #process the image to check if the cell is empty or has numbers in them
        grey = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(grey,150,255,cv2.THRESH_BINARY)

        crop_width = 10
        cropped = thresh[crop_width:70-crop_width,crop_width:70-crop_width]
        print(cv2.countNonZero(cropped))
        #finding the contours
        #_,cnts,_ = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #cnts = sorted(cnts, key=cv2.contourArea, reverse= True)[1]
        #M = cv2.moments(thresh)
        #cx = int(M['m10']/M['m00'])
        #cy = int(M['m01']/M['m00'])
        #cropped = thresh[cy-20:cy+20,cx-20:cx+20]
        #print(cv2.countNonZero(cropped))
        #cv2.imshow("Thresh",cropped)
        #cv2.waitKey(0)
        if int(cv2.countNonZero(cropped)) < 2400:
            cv2.imwrite("data/{}.jpg".format(str(x)),cropped)
        x +=1
        #b1 +=1
    #a1+=1
        #if a number exists, it will have its contour
'''
        if(len(cnts) > 1):
            #cnts = sorted(cnts, key=cv2.contourArea, reverse= True)[1]

            #finding thr area of the contour deected
            area = cv2.contourArea(cnts, True)
            #print(area)
            #if the area is big enough to fit the number but not
            #too big to be read by junk values
            if area>100 and area<1000:

                #crop the roi only to our number
                M = cv2.moments(cnts)

                #cx and cy hold the center coordinates of the cell
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])

                #creating the cropped image
                #the height and width of number can be approximated
                cropped = thresh[cy-22:cy+22,cx-20:cx+20]

                #inverting the colour of foreground and background
                _,inverted_img = cv2.threshold(cropped,230,255,cv2.THRESH_BINARY_INV)

                #show the image
                #cv2.imshow("cells",inverted_img)
                #cv2.waitKey(0)
        '''

'''
grey = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
_,thr = cv2.threshold(grey,120,255,cv2.THRESH_BINARY)
edge = cv2.Canny(thr,100,200)
_,cnts,_ = cv2.findContours(edge.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#cnts = sorted(cnts, key=cv2.contourArea, reverse= True)
print(len(cnts))
cv2.imshow("img",res)
cv2.waitKey(0)
#for c in cnts:
    #cv2.drawContours(res,[c],-1,(0,0,255),3)
cv2.drawContours(res,[cnts[]],0,(0,0,255),3)
cv2.imshow("image",res)
cv2.waitKey(0)
'''
cv2.imshow("image",res)
cv2.waitKey(0)
cv2.destroyAllWindows()
