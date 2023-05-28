from ultralytics import YOLO
import cv2
import cvzone
import math


cap = cv2.VideoCapture(0) #for webcam and live stream
#cap = cv2.VideoCapture("../video folder/video filename") #for videos
cap.set(3,1280) #set the width
cap.set(4, 720) #set the hight

model = YOLO("best.pt")
classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery', 'mini-van', 'sedan', 'semi', 'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']

mycolor = (0, 0, 255)

while True:
    success, img = cap.read()
    result = model(img, stream=True)
    for r in result:
        boxes = r.boxes
        for box in boxes:
            #bounding box
            x1, y1, x2, y2 =  box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1 , y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h),t=4)


            #confidence
            conf = math.ceil(box.conf[0]*100)/100
            cls = int(box.cls[0])

            if conf>0.6:
                currentclass = classNames[cls]
                if currentclass == 'Hardhat':
                    mycolor = (0, 255, 0)
                elif currentclass == 'Person':
                    mycolor = (0, 255, 0)
                else:
                    mycolor = (0, 0, 255)
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                   (max(0,x1),max(35,y1-20)), scale=1, thickness=2, colorR=mycolor, colorT=(255, 255, 255), offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), color=mycolor, thickness=3)

    cv2.imshow("image", img)
    cv2.waitKey(1)

