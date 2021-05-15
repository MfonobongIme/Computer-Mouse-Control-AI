#import libraries

import cv2
import mediapipe as mp
import time
import math
class handDetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode #creates an object, and the object will have its own variable
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        #create object from our class hands
        self.mpHands = mp.solutions.hands #formailty before using mediapipe model
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon) #create hands object

        self.mpDraw = mp.solutions.drawing_utils #helps us draw the points on the hands
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw = True): #detect hands



        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert to rgb because the object only uses rgb images
        self.results = self.hands.process(imgRGB) #processes the frame for us and gives us the result
        #print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks: #if true,we go in
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS) #draws the landmarks and connects them together

        return img #return image if we have drawn on it

    def findPosition(self, img, handNo = 0, draw = True): #find position of landmark

        xList = []
        yList = []
        bbox = []
        self.lmList = [] #this list will have all the landmarks list
        if self.results.multi_hand_landmarks:  # if true,we go in
            myHand = self.results.multi_hand_landmarks[handNo] #gets the hand

            # find id and landmark inside myhand.landmark
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape  # get the height, width and center of image
                cx, cy = int(lm.x * w), int(lm.y * h)  # multiply landmark position with img
                xList.append(cx) #append the landmarks positions to xList
                yList.append(cy) #append the landmarks positions to yList
                #print(id, cx, cy)  # prints the id number of the landmark and its positions
                self.lmList.append([id, cx, cy]) #append the values of the landmark position into lmList
                #if id == 4:  # the first landmark
                if draw: #if draw is True....
                    cv2.circle(img, (cx, cy), 8, (102, 0, 102), cv2.FILLED)  # draw circle on the specified landmark

            #for bounding box
            xmin, xmax = min(xList), max(xList) #find the minmum and maximum values of the landmarks in the x coordinate
            ymin, ymax = min(yList), max(yList) #find the minmum and maximum values of the landmarks in the y coordinate
            bbox = xmin, ymin, xmax, ymax #use the minimum and maximum values of the landmark coordinates for our bounding box


            #draw bounding box rectangle
            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax+20),
                              (255, 0, 255), 2)
        return self.lmList, bbox


        return self.lmList

    #functions that checks which fingers are up
    def fingersUp(self):
        fingers = [] #create list called fingers

        #thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]: #check if the tip of our thumb is on the right or left
            fingers.append(1)
        else:
            fingers.append(0)

        #4 fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img, draw = True, r= 15, t = 3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (255, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    pTime = 0  # previous time
    cTime = 0  # current time

    cap = cv2.VideoCapture('for.mp4')
    detector = handDetector() #create detector object

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        #findPosition(img)
        lmList = detector.findPosition(img)


        #if len(lmList) != 0:
            #print(lmList[4])


        cTime = time.time()  # this will give us the current time
        fps = 1 / (cTime - pTime)
        pTime = cTime

        #cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)

        cv2.imshow("image", img)
        cv2.waitKey(1)





if __name__ == "__main__":
    main()
