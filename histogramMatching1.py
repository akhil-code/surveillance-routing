import cv2,urllib,sys,math, sys
import numpy as np
import inspect
from matplotlib import pyplot as plt

#FUNCTIONS
#executes first part of the program. i.e to find the difference between two frames
def getDifferenceHulls(imgFrame1,imgFrame2):
    #making duplicates of the above frames
    imgFrame1Copy = imgFrame1.copy()
    imgFrame2Copy = imgFrame2.copy()

    #changing the colorspace to grayscale
    imgFrame1Copy = cv2.cvtColor(imgFrame1Copy,cv2.COLOR_BGR2GRAY)
    imgFrame2Copy = cv2.cvtColor(imgFrame2Copy,cv2.COLOR_BGR2GRAY)

    #applying gaussianblur
    imgFrame1Copy = cv2.GaussianBlur(imgFrame1Copy,(5,5),0)
    imgFrame2Copy = cv2.GaussianBlur(imgFrame2Copy,(5,5),0)

    #finding the difference of the two frames and thresholding the diff
    imgDifference = cv2.absdiff(imgFrame1Copy,imgFrame2Copy)
    _,imgThresh = cv2.threshold(imgDifference,30,255,cv2.THRESH_BINARY)

    # cv2.imshow("imgThresh",imgThresh)

    # morphological operations: dilation and erosion
    kernel = np.ones((5,5),np.uint8)
    imgThresh = cv2.dilate(imgThresh,kernel,iterations = 1)
    imgThresh = cv2.dilate(imgThresh,kernel,iterations = 1)
    imgThresh = cv2.erode(imgThresh,kernel,iterations = 1)


    #finding contours of the thresholded image
    contours, hierarchy = cv2.findContours(imgThresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #finding and drawing convex hulls
    hulls = []  #used to store hulls
    for cnt in contours:
        hulls.append(cv2.convexHull(cnt))

    return hulls

#draws the rectangles on the motion detected object
def drawBlobInfoOnImage(blobs,imgFrame2Copy):
    for i in range(len(blobs)):
        if (blobs[i].blnStillBeingTracked == True):
            rect_corner1 = (blobs[i].currentBoundingRect[0],blobs[i].currentBoundingRect[1])
            rect_corner2 = (blobs[i].currentBoundingRect[0]+blobs[i].width, blobs[i].currentBoundingRect[1]+blobs[i].height)

            # font settings
            intFontFace = cv2.FONT_HERSHEY_SIMPLEX;
            dblFontScale = blobs[i].dblCurrentDiagonalSize / 60.0
            intFontThickness = int(round(dblFontScale * 1.0))
            point = ((rect_corner1[0]+rect_corner2[0])/2,(rect_corner1[1]+rect_corner2[1])/2)

            # labels blob numbers
            cv2.putText(imgFrame2Copy, str(i), blobs[i].centerPositions[-1], intFontFace, dblFontScale, (0,255,0), intFontThickness);
            # draws box around the blob
            cv2.rectangle(imgFrame2Copy, rect_corner1,rect_corner2, (0,0,255))

#draws the contours on the image
def drawAndShowContours(imageSize,contours,strImageName):
    image = np.zeros(imageSize, dtype=np.uint8)
    cv2.drawContours(image, contours, -1,(255,255,255), -1)
    cv2.imshow(strImageName, image);

#draws the contours similar to the drawAndShowContours function
#but here the input provided is not the contours but object of class Blob
def drawAndShowBlobs(imageSize,blobs,strWindowsName):
    image = np.zeros(imageSize, dtype=np.uint8)
    contours = []
    for blob in blobs:
        if blob.blnStillBeingTracked == True:
            contours.append(blob.currentContour)

    cv2.drawContours(image, contours, -1,(255,255,255), -1);
    cv2.imshow(strWindowsName, image);


#CLASS
#class Blob consisting of variables and functions related to it
class Blob:
    #functions
    def printInfo(self):
        print 'area: '+str(self.area)+' Pos: '+str(self.centerPositions)

    def __init__(self, _contour,srcImage):
        self.centerPositions = []
        self.predictedNextPosition = [-1,-1]


        self.currentContour = _contour

        # mask = np.zeros(imgFrame2.shape, np.uint8)
        # cv2.drawContours(mask, self.currentContour, -1, (255,255,255),1)
        # roi = cv2.bitwise_and(imgFrame2,imgFrame2,mask=self.currentContour)
        # cv2.imshow("roii",roi)

        self.currentBoundingRect = cv2.boundingRect(self.currentContour)  #x,y,w,h
        x = (self.currentBoundingRect[0] + self.currentBoundingRect[0] + self.currentBoundingRect[2])/2
        y = (self.currentBoundingRect[1] + self.currentBoundingRect[1] + self.currentBoundingRect[3]) / 2
        self.currentCenter = (x,y)
        self.width = self.currentBoundingRect[2]
        self.height =  self.currentBoundingRect[3]
        self.area = self.currentBoundingRect[2] * self.currentBoundingRect[3]

        self.centerPositions.append(self.currentCenter)

        self.dblCurrentDiagonalSize = math.sqrt(math.pow(self.currentBoundingRect[2], 2) + math.pow(self.currentBoundingRect[3], 2));
        self.dblCurrentAspectRatio = float(self.currentBoundingRect[2])/float(self.currentBoundingRect[3])

        x,y,w,h = self.currentBoundingRect #x,y,w,h
        self.currentROI = srcImage[y:y+h, x:x+w]
        self.rois = []
        self.noOfTimesAppeared = 1
        self.featureMatches = 0

        # flags
        self.blnStillBeingTracked = True;
        self.blnCurrentMatchFoundOrNewBlob = True;

        self.intNumOfConsecutiveFramesWithoutAMatch = 0;


def detect_point(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print (x,y)

def findMatch(target,images):
    # methods = (("Correlation", cv2.cv.CV_COMP_CORREL),("Chi-Squared", cv2.cv.CV_COMP_CHISQR),("Intersection", cv2.cv.CV_COMP_INTERSECT),("Hellinger", cv2.cv.CV_COMP_BHATTACHARYYA))
    results = []

    target = cv2.cvtColor(target,cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist(target, [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    target_histogram = cv2.normalize(hist).flatten()

    for i in range(len(images)):
        img = cv2.cvtColor(images[i],cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist(img, [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
        hist = cv2.normalize(hist).flatten()

        result = cv2.compareHist(target_histogram,hist,cv2.cv.CV_COMP_BHATTACHARYYA)
        results.append((i,result))

    results = sorted(results, key=lambda val: val[1])
    return results[0]

#MAIN CODE
src = cv2.imread("database/img0.jpg")
cap = cv2.VideoCapture('video.avi')     #video file object
target = cv2.imread("database/img204.jpg")
images = []


cv2.namedWindow("target",cv2.WINDOW_NORMAL)
cv2.imshow("target",target)

#checks if the video file is valid
if cap.isOpened():
    _,imgFrame1 = cap.read()   #capturing the first reference frame
else:
    sys.exit()

#variables used within the infinite loop
blnFirstFrame = True        #is true if the frame captured is first frame

while cap.isOpened():

    #capturing second reference frame
    _,imgFrame2 = cap.read()

    if imgFrame2 is None:
        break

    #obtaining convex hulls and newly captured image
    hulls = getDifferenceHulls(imgFrame1,imgFrame2)

    #Blob validation
    currentFrameBlobs = []
    for hull in hulls:
        possibleBlob = Blob(hull,imgFrame2.copy())

        #conditions to approximate the blobs
        if (possibleBlob.area > 100 and \
        possibleBlob.dblCurrentAspectRatio >= 0.2 and \
        possibleBlob.dblCurrentAspectRatio <= 1.75 and \
        possibleBlob.width > 20 and \
        possibleBlob.height > 20 and \
        possibleBlob.dblCurrentDiagonalSize > 30.0 and \
        (cv2.contourArea(possibleBlob.currentContour) / float(possibleBlob.area)) > 0.40):
            currentFrameBlobs.append(possibleBlob)
        del possibleBlob

    # appending images
    for i in range(len(currentFrameBlobs)):
        images.append(currentFrameBlobs[i].currentROI)


    #replacing the frame1 with frame2, so that newly captured frame can be stored in frame2
    imgFrame1 = imgFrame2.copy()

    #displaying any movement in the output screen
    img_current_blobs = imgFrame2.copy()

    # drawing current frame blobs
    drawBlobInfoOnImage(currentFrameBlobs,img_current_blobs)

    cv2.imshow("current blobs",img_current_blobs)

    #flagging subsequent frames
    blnFirstFrame = False
    del currentFrameBlobs[:]    #clearing the currentFrameBlobs to capture newly formed blobs

    key_in = cv2.waitKey(10) & 0xFF
    if(key_in == ord('q')):
        break



match_idx, match_val = findMatch(target,images)
print 'index {}, val: {}'.format(match_idx, match_val)

if match_val < 0.8:
    cv2.namedWindow("match",cv2.WINDOW_NORMAL)
    cv2.imshow("match",images[match_idx])
else:
    cv2.destroyWindow("match")

cv2.waitKey(0)
#deletes all the opened windows
cap.release()
cv2.destroyAllWindows()
