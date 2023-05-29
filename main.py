import cv2
from  cvzone.HandTrackingModule import HandDetector
import math

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4, 720)

#hand detecore

detecore = HandDetector(detectionCon=0.8, maxHands=1)

#find function
# x is the value of distance
# y is the value in CM
x = [300, 245, 200, 170, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]


while True:
    success, img = cap.read()
    hands, img = detecore.findHands(img)

    if hands:
        lmList = hands[0]['lmList']
        x1, y1 = lmList[5][0], lmList[5][1]
        x2, y2 = lmList[17][0], lmList[17][1]

        distance = int(math.sqrt((y2-y1)**2 + (x2-x1)**2))





    cv2.imshow("image", img)

    cv2.waitKey(1)