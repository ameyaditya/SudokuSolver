import numpy as np
import imutils
import cv2

#read the image of sudoku
image = cv2.imread("sudo4.jpg")

#process the image for contour recognition
#run the canny edge to detect edges
blurred = cv2.medianBlur(image,7)
gray = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)
th3 = cv2.adaptiveThreshold(gray,150,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,31,3)
edged = cv2.Canny(th3,50,100)
cv2.imshow("edged",edged)
cv2.waitKey(0)
#cv2.imshow("Edged_image",edged)
#cv2.waitKey(0)

#after applying the canny edge function
#find the contours(closed loops) in the image(edged)
_,contours,_ = cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

#sort the contours based on their area
#assuming the area of the sudoku will the largest
#label1: updating contour value
contours = sorted(contours,key=cv2.contourArea,reverse=True)[0:10]
print(len(contours))
#extracting the coordinates of the actual sudoku puzzle
for c  in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c,0.02*peri, True)

    #if the approximated polygon has 4 sides, it is our sudoku puzzle
    if len(approx) == 4:
        #draw a rectangle around the puzzle
        cv2.drawContours(image,[c],-1,(0,255,0),3)
        break

#show the image on which the rectangle was drawn on
cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#if the above code doesnt find the sudoku properly
#goto label1: increment the index value and try again
