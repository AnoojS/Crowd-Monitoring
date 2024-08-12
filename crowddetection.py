import cv2
from ultralytics import YOLO
import math

model=YOLO('yolov8x.pt')

cap=cv2.VideoCapture(0)
cv2.namedWindow('Crowd Detection', cv2.WINDOW_NORMAL)

while(True):
    ret,frame=cap.read()
    
    result=model.predict(frame,classes=0)
    boxes=result[0].boxes

    count=0

    for box in boxes:
        count+=1
        x1,y1,x2,y2=box.xyxy[0]
        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        confidence = math.ceil((box.conf[0]*100))/100
        label = f'person: {confidence:.2f}'
        cv2.putText(frame, label, (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    label = f'Crowd Count: {count}'
    cv2.putText(frame, label, (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    print(count)

    cv2.imshow('Crowd Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()