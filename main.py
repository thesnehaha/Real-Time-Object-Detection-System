import cv2
import numpy as np
import random

R = random.randint(0, 255)
G = random.randint(0, 255)
B = random.randint(0, 255)

video=cv2.VideoCapture("video2.mp4")

CLASSES = ["Background", "Aeroplane", "Bicycle", "Bird", "Boat",
	"Bottle", "Bus", "Car", "Cat", "Chair", "Cow", "Dining-table",
	"Dog", "Horse", "Motorbike", "Person", "Pottedplant", "Sheep",
	"Sofa", "Train", "TV-monitor"]

color = [(R,G,B) for i in CLASSES]


net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt','MobileNetSSD_deploy.caffemodel')

while True:
    ret,frame=video.read()
    frame=cv2.resize(frame,(640,480))
    (h,w)=frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),0.007843, (300,300),127.5)
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>0.5:
            id=detections[0,0,i,1]
            box=detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY)=box.astype("int")
            cv2.rectangle(frame, (startX-1, startY-40), (endX+1, startY-3), color[int(id)], -1)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color[int(id)], 4)
            cv2.putText(frame, CLASSES[int(id)], (startX+10, startY-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255))
    print(detections)
    cv2.imshow("Frame", frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()