import cv2
import os
import numpy as np
import HandTrackingModule1 as htm

folderPath = 'Header'
myList = os.listdir(folderPath)
print(myList)
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))
header = overlayList[0]

brushThickness = 15
eraserThickness = 50
drawColor = (206, 131, 222)

cap = cv2.VideoCapture(0)
cap.set(4, 720)
cap.set(3, 1280)

detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0
canvas = np.zeros((720, 1280, 3), np.uint8)

while True:

    success, img = cap.read()
    img = cv2.flip(img,1)


    header = cv2.resize(header, (img.shape[1], header.shape[0]))


    img[0:header.shape[0], 0:header.shape[1]] = header


    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw = False)

    if len(lmList) != 0:


        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # check which fingers are up
        fingers = detector.fingersUp()
        #print(fingers)

        # selection if two fingers up
        if fingers[1] and fingers[2] and fingers[0]==False and fingers[3]==False and fingers[4]==False:
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
            xp, yp = 0, 0
            print('Selection Mode')

            #Checking for the click
            if y1 < 125:
                if 140 < x1 < 160:
                    header = overlayList[0]
                    drawColor = (206, 131, 222)
                if 250 < x1 < 310:
                    header = overlayList[1]
                    drawColor = (242,108,185)
                if 420 < x1 < 440:
                    header = overlayList[2]
                    drawColor = (141, 205, 116)
                if 500 < x1 < 640:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # Drawing mode - index finger up
        if fingers[1] and fingers[2]==False and fingers[0]==False and fingers[3]==False and fingers[4]==False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print('Drawing Mode')

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (255, 255, 255):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(canvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img,(xp, yp),(x1, y1), drawColor, brushThickness)
                cv2.line(canvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    imgGrey = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGrey, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)



    cv2.imshow("Image",img)
    cv2.imshow("Canvas", canvas)
    cv2.waitKey(1)
