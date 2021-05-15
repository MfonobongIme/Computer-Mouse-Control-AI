#import libraries
import cv2
import numpy as np
import time
import os
import handTrackingModule as htm
import autopy

cap = cv2.VideoCapture(0)

#resize webcam
wCam, hCam = 640, 480
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

wScr, hScr = autopy.screen.size() #this gives the size of the screen

frameR = 100 #frame reduction

smoothening = 7
plocX, plocY = 0, 0 #prev location of x and x that will be used for smoothening
clocX, clocY = 0, 0 #current location of x and x that will be used for smoothening

clickText = 'Clicking mode'
MovingText = 'Moving'
ClickingText = 'Click!'


detector = htm.handDetector(maxHands=1)#create detector object

while True:
    #find hand landmarks

    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # get the tip of the index and middle finger
    if len(lmList) != 0:
        x1, y1 = lmList[4][1:] #gives the coordinate of middle finger
        x2, y2 = lmList[8][1:] #gives the coordinate of index finger

        #print(x1, y1, x2, y2)

        #check which fingers are up
        fingers = detector.fingersUp()
        print(fingers)

        #cv2.rectangle(img, (frameR, frameR),
                      #(wCam - frameR, hCam - frameR), (255, 0, 255), 2)  # frame reduction to make movement smoother

        #only index finger in moving mode
        if fingers[1] == 1 and fingers[0] == 0:

            #convert coordinates

            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            #smoothen values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening


            #move mouse
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX,clocY  #update values

            cv2.putText(img, 'Jarvis: {}'.format(MovingText), (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

        #both index and middle fingers are up:clicking mode
        if fingers[1] == 1 and fingers[0] == 1:
            # find distance btw fingers
            length, img, lineInfo = detector.findDistance(4, 8, img) #find distance btw landmark 8 and 12
            #print(length)

            cv2.putText(img, 'Jarvis: {}'.format(clickText), (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

            # click mouse if distance is short
            if length < 50: #check if length is less than 40
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 255), cv2.FILLED)

                autopy.mouse.click()

                cv2.putText(img, 'Jarvis: {}'.format(ClickingText), (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255),
                            3)






    #frame rate
    cTime = time.time()  # this will give us the current time
    fps = 1 / (cTime - pTime)
    pTime = cTime
    #cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)

    #display

    cv2.imshow('image', img)
    cv2.waitKey(1)