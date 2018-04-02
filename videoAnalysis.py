import cv2,urllib,sys,math
import numpy as np

#FUNCTIONS

#executes first part of the program. i.e to find the difference between two frames
def getDifferenceHulls(cap):
    #capturing two reference frames
    _,imgFrame2 = cap.read()

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
    _, contours, hierarchy = cv2.findContours(imgThresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #finding and drawing convex hulls
    hulls = []  #used to store hulls
    for cnt in contours:
        hulls.append(cv2.convexHull(cnt))

    return hulls,imgFrame2

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
            # cv2.putText(imgFrame2Copy, str(i), blobs[i].centerPositions[-1], intFontFace, dblFontScale, (0,255,0), intFontThickness);
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

def getBlobROIs(blobs, srcImage):
    global file_counter
    rois = []
    for blob in blobs:
        if blob.blnStillBeingTracked == True:
            x,y,w,h = blob.currentBoundingRect #x,y,w,h
            roi = srcImage[y:y+h, x:x+w]
            rois.append(roi)
            cv2.imwrite("database/img{}.jpg".format(file_counter),roi)
            file_counter += 1

#find the distance between two points p1 and p2
def distanceBetweenPoints(point1,point2):
    intX = abs(point1[0] - point2[0])
    intY = abs(point1[1] - point2[1])
    return math.sqrt(math.pow(intX, 2) + math.pow(intY, 2))

#matching algorithm to corelate two blob objects by matching it with the expected one
def matchCurrentFrameBlobsToExistingBlobs(existingBlobs,currentFrameBlobs):
    for existingBlob in existingBlobs:
        existingBlob.blnCurrentMatchFoundOrNewBlob = False
        existingBlob.predictNextPosition()

    for currentFrameBlob in currentFrameBlobs:
        intIndexOfLeastDistance = 0
        dblLeastDistance = 100000.0

        for i in range(len(existingBlobs)):
            if (existingBlobs[i].blnStillBeingTracked == True):
                dblDistance = distanceBetweenPoints(currentFrameBlob.centerPositions[-1], existingBlobs[i].predictedNextPosition)
                # print dblDistance
                if (dblDistance < dblLeastDistance):
                    dblLeastDistance = dblDistance
                    intIndexOfLeastDistance = i

        if (dblLeastDistance < currentFrameBlob.dblCurrentDiagonalSize * 1.15):
            addBlobToExistingBlobs(currentFrameBlob, existingBlobs, intIndexOfLeastDistance)
        else:
            addNewBlob(currentFrameBlob, existingBlobs)


    for existingBlob in existingBlobs:
        if (existingBlob.blnCurrentMatchFoundOrNewBlob == False):
            existingBlob.intNumOfConsecutiveFramesWithoutAMatch +=1;

        if (existingBlob.intNumOfConsecutiveFramesWithoutAMatch >= 5):
            existingBlob.blnStillBeingTracked = False;

#adds the details of the matching blob to the existingBlob
def addBlobToExistingBlobs(currentFrameBlob,existingBlobs,i):
    # print 'found continuos blob'
    existingBlobs[i].currentContour = currentFrameBlob.currentContour;
    existingBlobs[i].currentBoundingRect = currentFrameBlob.currentBoundingRect;

    existingBlobs[i].centerPositions.append(currentFrameBlob.centerPositions[-1])

    existingBlobs[i].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize;
    existingBlobs[i].dblCurrentAspectRatio = currentFrameBlob.dblCurrentAspectRatio;

    existingBlobs[i].blnStillBeingTracked = True;
    existingBlobs[i].blnCurrentMatchFoundOrNewBlob = True;

#adds new blob to the list
def addNewBlob(currentFrameBlob,existingBlobs):
    currentFrameBlob.blnCurrentMatchFoundOrNewBlob = True
    existingBlobs.append(currentFrameBlob)


#CLASS
#class Blob consisting of variables and functions related to it
class Blob:
    #functions
    def printInfo(self):
        print 'area: '+str(self.area)+' Pos: '+str(self.centerPositions)

    def __init__(self, _contour):

        self.centerPositions = []
        self.predictedNextPosition = [-1,-1]

        self.currentContour = _contour
        self.currentBoundingRect = cv2.boundingRect(self.currentContour)  #x,y,w,h
        x = (self.currentBoundingRect[0] + self.currentBoundingRect[0] + self.currentBoundingRect[2])/2
        y = (self.currentBoundingRect[1] + self.currentBoundingRect[1] + self.currentBoundingRect[3]) / 2
        self.currentCenter = (x,y)
        self.width = self.currentBoundingRect[2]
        self.height =  self.currentBoundingRect[3]
        self.area = self.currentBoundingRect[2] * self.currentBoundingRect[3]

        self.inside = isWithinEllipse(x,y+(self.height/2))


        self.centerPositions.append(self.currentCenter)

        self.dblCurrentDiagonalSize = math.sqrt(math.pow(self.currentBoundingRect[2], 2) + math.pow(self.currentBoundingRect[3], 2));
        self.dblCurrentAspectRatio = float(self.currentBoundingRect[2])/float(self.currentBoundingRect[3])

        self.blnStillBeingTracked = True;
        self.blnCurrentMatchFoundOrNewBlob = True;

        self.intNumOfConsecutiveFramesWithoutAMatch = 0;

    def predictNextPosition(self):
        numPositions = len(self.centerPositions)
        if (numPositions == 1):
            self.predictedNextPosition[0] = self.centerPositions[-1][0]
            self.predictedNextPosition[1] = self.centerPositions[-1][1]

        elif (numPositions == 2):
            deltaX = self.centerPositions[1][0] - self.centerPositions[0][0]
            deltaY = self.centerPositions[1][1] - self.centerPositions[0][1]

            self.predictedNextPosition[0] = self.centerPositions[-1][0] + deltaX
            self.predictedNextPosition[1] = self.centerPositions[-1][1] + deltaY

        elif (numPositions == 3):
            sumOfXChanges = ((self.centerPositions[2][0] - self.centerPositions[1][1]) * 2) + \
            ((self.centerPositions[1][0] - self.centerPositions[0][0]) * 1)

            deltaX = int(round(float(sumOfXChanges)/3.0))

            sumOfYChanges = ((self.centerPositions[2][1] - self.centerPositions[1][1]) * 2) + \
            ((self.centerPositions[1][1] - self.centerPositions[0][1]) * 1)

            deltaY = int(round(float(sumOfYChanges) / 3.0))

            self.predictedNextPosition[0] = self.centerPositions[-1][0] + deltaX
            self.predictedNextPosition[1] = self.centerPositions[-1][1] + deltaY

        elif (numPositions == 4) :
            sumOfXChanges = ((self.centerPositions[3][0] - self.centerPositions[2][0]) * 3) + \
            ((self.centerPositions[2][0] - self.centerPositions[1][0]) * 2) + \
            ((self.centerPositions[1][0] - self.centerPositions[0][0]) * 1)

            deltaX = int(round(float(sumOfXChanges) / 6.0))

            sumOfYChanges = ((self.centerPositions[3][1] - self.centerPositions[2][1]) * 3) + \
            ((self.centerPositions[2][1] - self.centerPositions[1][1]) * 2) + \
            ((self.centerPositions[1][1] - self.centerPositions[0][1]) * 1)

            deltaY = int(round(float(sumOfYChanges) / 6.0))

            self.predictedNextPosition[0] = self.centerPositions[-1][0] + deltaX;
            self.predictedNextPosition[1] = self.centerPositions[-1][1] + deltaY;

        elif (numPositions >= 5):
            sumOfXChanges = ((self.centerPositions[numPositions - 1][0] - self.centerPositions[numPositions - 2][0]) * 4) + \
            ((self.centerPositions[numPositions - 2][0] - self.centerPositions[numPositions - 3][0]) * 3) + \
            ((self.centerPositions[numPositions - 3][0] - self.centerPositions[numPositions - 4][0]) * 2) + \
            ((self.centerPositions[numPositions - 4][0] - self.centerPositions[numPositions - 5][0]) * 1)

            deltaX = int(round(float(sumOfXChanges) / 10.0));

            sumOfYChanges = ((self.centerPositions[numPositions - 1][1] - self.centerPositions[numPositions - 2][1]) * 4) + \
            ((self.centerPositions[numPositions - 2][1] - self.centerPositions[numPositions - 3][1]) * 3) + \
            ((self.centerPositions[numPositions - 3][1] - self.centerPositions[numPositions - 4][1]) * 2) + \
            ((self.centerPositions[numPositions - 4][1] - self.centerPositions[numPositions - 5][1]) * 1)

            deltaY = int(round(float(sumOfYChanges) / 10.0))

            self.predictedNextPosition[0] = self.centerPositions[-1][0] + deltaX;
            self.predictedNextPosition[1] = self.centerPositions[-1][1] + deltaY;

        else:
            #should never get here
            pass

def detect_point(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print (x,y)


def isWithinEllipse(x,y):
    # print ((math.pow((x-436),2))/22500) +((math.pow((y-381),2))/5625)
    if ((math.pow((x-436),2))/40000) +((math.pow((y-381),2))/10000)  <= 1:
        return True
    else:
        return False

#MAIN CODE

cap = cv2.VideoCapture('video.avi')     #video file object
#checks if the video file is valid
if cap.isOpened():
    _,imgFrame1 = cap.read()   #capturing the first reference frame
else:
    sys.exit()

#variables used within the infinite loop
blnFirstFrame = True        #is true if the frame captured is first frame
blobs = []                  #captures all the new blobs found
file_counter = 0

while cap.isOpened():
    #obtaining convex hulls and newly captured image
    hulls,imgFrame2 = getDifferenceHulls(cap)

    #Blob validation
    currentFrameBlobs = []
    for hull in hulls:
        possibleBlob = Blob(hull)

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

    #replacing the frame1 with frame2, so that newly captured frame can be stored in frame2
    imgFrame1 = imgFrame2.copy()

    #displaying any movement in the output screen
    img_rectangles = imgFrame2.copy()
    if(len(currentFrameBlobs) > 0):
        drawAndShowBlobs(imgFrame2.shape,currentFrameBlobs,"imgCurrentFrameBlobs")
        drawBlobInfoOnImage(currentFrameBlobs,img_rectangles)

    #checks if the frame is the first frame of the video
    if blnFirstFrame == True:
        for currentFrameBlob in currentFrameBlobs:
            blobs.append(currentFrameBlob)
    else:
        matchCurrentFrameBlobsToExistingBlobs(blobs,currentFrameBlobs)

    #displays the blobs on the screen that are consistent or matched
    drawAndShowBlobs(imgFrame2.shape,blobs,"imgBlobs")

    # blobs is the output of processing till here which has to be further used for our needs
    getBlobROIs(blobs,imgFrame2.copy())

    cv2.imshow("output",img_rectangles)
    #flagging the further frames
    blnFirstFrame = False
    del currentFrameBlobs[:]    #clearing the currentFrameBlobs to capture newly formed blobs

    key_in = cv2.waitKey(100) & 0xFF
    if(key_in == ord('q')):
        break

#deletes all the opened windows
cap.release()
cv2.destroyAllWindows()
