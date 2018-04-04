import numpy as np
import matplotlib.pyplot as plt
import imutils
from sklearn import svm
import cv2
import pandas as pd
import os

#used to check if the puzzle is identified properly
#takes the image or photo as the input
def image_checker(image2):
    image = image2.copy()
    """
    processing the image for contour recognition
    run the canny edge to detect the edges
    """
    #reduce or increase the value of blur based on the amount of lines detected in canny edge
    blurred = cv2.medianBlur(image,7)

    gray = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)

    #change the threshold value according to the amount of light falling on the image
    #the current setting are for a well lit image
    thresh = cv2.adaptiveThreshold(gray,150,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,31,3)
    #cv2.imshow("thresh",thresh)
    #cv2.waitKey(0)

    #the canny edge funtion is used to detect edges in the image
    #adjust the values accordingly if the output is not obtained
    edged = cv2.Canny(thresh.copy(),50,100)
    #cv2.imshow("Edged",edged)
    #cv2.waitKey(0)

    """
    after applying the canny edge function
    find the contours(closed loops) in the image edged
    and sort them according to their size
    we have assumed that the puzzle is the largest contour present
    """
    _,contours,_ = cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

    contours = sorted(contours,key=cv2.contourArea,reverse=True)[0:10]
    #sliced the list to first 10 biggest contours, in case the puzzle is not found
    #change the index values and try again

    """
    find the coordinates of the actual puzzles and draw a rectangle around it
    """
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c,0.02*peri,True)

        #if the approximated polygon has 4 sides, our puzzle is found
        if len(approx) == 4:
            cv2.drawContours(image,[c],-1,(0,255,0),3)
            coordinates = approx
            break
    #show the image with rectangle drawn on it
    cv2.imshow("Puzzle",image)
    cv2.waitKey(0)
    return coordinates


"""
this function is to read the dataset images from a particular folder and append
it to a list with samples,features and targets which will later be used to
train the machine learning classifier
"""
def load_images(folder,xtrain,ytrain):
    #iterate through all the images in the folder
    for filename in os.listdir(folder):

        #store the numpy array of the image in the variable dataset
        #we will process this image and append it to xtrain to
        #train the classifier
        dataset = cv2.imread(os.path.join(folder,filename))

        #if the image of the dataset exists
        #basically eroor handling
        if dataset is not None:

            """
            the dataset/image that we have read in has 3 channels of RGB
            and might be of varied size, so we resize our dataset to 50x50 pixels
            which will make sure that we do not loose much of the data, and generalise
            between all the datasets,

            To convert the datasets to a single channel, we apply threshold to the image
            and convert the 2D matrix/ numpy array into a 1D array as the classifier
            can one read one arrays.

            the names of the files are actually the targets to the data values in the
            images, so we append that into ytrain
            """
            dataset = cv2.resize(dataset,(50,50),interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(dataset,cv2.COLOR_BGR2GRAY)
            _,thresh = cv2threshold(gray,150,255,cv2.THRESH_BINARY)
            xtrain.append(*thresh.reshape(1,2500))
            ytrain.append(int(folder.split('\\')[1]))

def training_classifier(clf,folder):
    xtrain = []
    ytrain = []

    #this calls the dataset over all the folders in the master Data folder
    for i in range(1,10):
        load_images(folder+str(i),xtrain,ytrain)
    #convert the training dataset into numpy array to have more flexibility
    #in performing operations on them
    xtrain = np.array(xtrain)
    ytrain = np.array(ytrain)

    #fits the training dataset to the SVC classifier
    clf.fit(xtrain,ytrain)

"""
the sudoku puzzle is isolated from the whole image
itertating through each box, and determining the number present in that particular
cell and storing it into an array
"""
def create_sudoku_matrix(image2):
    sudoku = [[0 for i in range(9)]for j in range(9)]
    iterX = 0
    iterY = 0
    image = image2.copy()
    pts = image_checker(image)

    #pts is a 3D array of size(4,1,2) we reshape it into (4,2)
    pts = pts.reshape(4,2)

    #creating a temporary array of zeros to store the coordinates in order
    rect = np.zeros((4,2), dtype='float32')

    #storing the sum along the rows and difference across each row
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    #the minimum sum value is the top left point and maximum sum value is the
    #bottom right point, similarl the minimum difference will the the top right
    #point and maximum difference is the bottom left point
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    #unpacking the array
    (tl,tr,br,bl) = rect

    #finding the width and height of the sudoku puzzle in the image
    #using the coordinates obtained using the formula
    #length = sqrt((x2 - x1)^2 + (y2 - y1)^2)
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    heightA = np.sqrt(((bl[0] - tl[0]) ** 2) + ((bl[1] - tl[1]) ** 2))
    heightB = np.sqrt(((br[0] - tr[0]) ** 2) + ((br[1] - tr[1]) ** 2))

    #getting the maximum values of the final dimensions of the puzzle
    maxwidth = max(int(widthA),int(widthB))
    maxheight = max(int(heightA),int(heightB))

    #construct our destinatio points which wil be used to map the screen to
    #to-down, "brid eye" view
    dst = np.array([
    [0,0],
    [maxwidth - 1, 0],
    [maxwidth - 1, maxheight - 1],
    [0, maxheight - 1]], dtype='float32')

    #calculate the perspective transform matrix and warp the perspective to grab
    #the screen
    M = cv2.getPerspectiveTransform(rect,dst)
    warp = cv2.warpPerspective(image,M,(maxwidth,maxheight))

    #resizing the image into 630x630 image
    res = cv2.resize(warp,(630,630),interpolation=cv2.INTER_AREA)
    #cv2.imshow("sudoku",res)
    #cv2.waitKey(0)

    """
    according to the image generated and stored in res, the size of the puzzle
    is 630x630, which imples that each cell of the sudoku is 70x70 as it is as
    square image
    we interate through rows and columns of the entire sudoku and if we find a
    number in that cell we predict the number using the svm classifier and
    store it in a 2D list
    """
    for i in range(0,561,70):
        for j in range(0,561,70):

            #creating our ROI
            roi = res[i:i+70,j:j+70]

            #process the image and using countnonzero() we check if the cell
            #has a number or not

            grey = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
            _,thresh = cv2.threshold(grey,150,255,cv2.THRESH_BINARY)
            crop_width = 10
            cropped = thresh[crop_width:70-crop_width,crop_width:70-crop_width]

            if int(cv2.countNonZero(cropped)) < 2400:
                
