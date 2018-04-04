import numpy as np
import cv2
import imutils

#first 35 lines are same as from the previous program
#read the image of sudoku
image = cv2.imread("sample3.jpg")

#process the image for contour recognition
#run the canny edge to detect edges
blurred = cv2.medianBlur(image,9)
gray = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)
th3 = cv2.adaptiveThreshold(gray,200,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,25,3)
edged = cv2.Canny(th3,100,200)

#after applying the canny edge function
#find the contours(closed loops) in the image(edged)
_,contours,_ = cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

#sort the contours based on their area
#assuming the area of the sudoku will the largest
#label1: updating contour value
contours = sorted(contours,key=cv2.contourArea,reverse=True)[4:10]

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

#resizing our sudoku to a 600x600 image
res = cv2.resize(warp,(600,600), interpolation=cv2.INTER_AREA)
res2 = res.copy()

cv2.imshow("sudoku",res)
cv2.waitKey(0)


cv2.destroyAllWindows()
