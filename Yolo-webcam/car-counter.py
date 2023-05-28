import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import  *
cap = cv2.VideoCapture("../videos/car.mp4") #for videos



cap.set(3,1280) #set the width
cap.set(4, 720) #set the hight

model = YOLO("../Yolo-weights/yolov8n.pt")

classNames = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign ',
              'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
              'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sport ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
              'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork' , 'knife', 'spoon' , 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
              'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet','tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard',
              'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
              'toothbrush']

#to mask an area to not be detected we need to create a mask file using canva first.
#mask = cv2.imread("mask.png")

#tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
lineLimits = [423,500,1200,500]

totalcounts = []

while True:
    success, img = cap.read()
    #using mask file to create new image
    #imgregen = cv2.bitwise_and(img, mask)

    result = model(img, stream=True)

    detections = np.empty((0,5))
    for r in result:
        boxes = r.boxes
        for box in boxes:
            #bounding box
            x1, y1, x2, y2 =  box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1 , y2-y1


            #confidence
            conf = math.ceil(box.conf[0]*100)/100
            cls = int(box.cls[0])
            currentclass = classNames[cls]

            if currentclass == "car" or currentclass == "truck" and conf > 0.5:
                #cvzone.putTextRect(img, f'{currentclass} {conf}', (max(0,x1),max(35,y1-20)),
                               #scale=0.6, thickness=1, offset=5)
                #cvzone.cornerRect(img, (x1, y1, w, h), t=4, l=15, rt=5)
                currentarray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentarray))
    resulttracker = tracker.update(detections)
    cv2.line(img, (lineLimits[0], lineLimits[1]),(lineLimits[2], lineLimits[3]), (0,0,255), 4)
    for res in resulttracker:
       x1,y1,x2,y2,id = res
       x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
       print(res)

       w, h = x2 - x1, y2 - y1
       cvzone.cornerRect(img, (x1, y1, w, h), t=4, l=15, rt=2, colorR=(255,0,255))
       #cvzone.putTextRect(img, f'', (max(0, x1), max(35, y1 - 20)),
                          #scale=1, thickness=3, offset=10)


       cx,cy = x1+w//2, y1+h//2
       cv2.circle(img, (cx,cy), 5, (255,0,255), cv2.FILLED)

       if lineLimits[0]<cx< lineLimits[2] and lineLimits[1]-10 <cy<lineLimits[3]+10:
           if totalcounts.count(id) == 0:
               totalcounts.append(id)
    cvzone.putTextRect(img, f'Count: {len(totalcounts)}', (50, 50))

    cv2.imshow("image", img)
    #cv2.imshow("imageRegion", imgregen)
    cv2.waitKey(1)

