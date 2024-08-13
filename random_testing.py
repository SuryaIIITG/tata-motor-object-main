"""import cv2
import numpy as np
import torch

def POINTS(events,x,y,flag,param):
    if events==cv2.EVENT_MOUSEMOVE:
        colorBGR=[x,y]
        print(colorBGR)

cv2.namedWindow("Frame")
cv2.setMouseCallback('Frame',POINTS)

model=torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)



# capture voideo
cap=cv2.VideoCapture(0)



count=0
area=[(56,48),(65,683),(811,672),(818,57)]
while True:
    ret,frame =cap.read()
    if not ret:
        break
    frame=cv2.resize(frame,(600,500))

    results=model(frame)
    a=results.pandas().xyxy(0).iterrows()
    print(a)    


    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()"""

import cv2
import torch
import numpy as np

from PIL import Image

""""points=[]
def POINTS(event, x, y, flags, param): 
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)  
        

cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME',POINTS) """        


  # Load Custom model 
model=torch.hub.load('ultralytics/yolov5','custom', path='C:\\Users\\Alime\\OneDrive\\Desktop\\YOLOv5-Flask2\\yolov5\\runs\\train\\exp47\\weights\\best.pt')


#model=torch.hub.load('ultralytics/yolov5','yolov5s', pretrained=True,classes=8)

cap=cv2.VideoCapture(1)         # 0 for builtin camera,1 for external camera       # for webcam capture

count=0


area=[(56,48),(65,683),(811,672),(818,57)]

while True:
    ret,frame=cap.read()
    if not ret:
        break
    frame=cv2.resize(frame,(900,800))

    results=model(frame)
    name=["rod","frame","armature body","wheel","bowl","dustbin","kabz","angle"]
    list1=[]
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        d=(row['name'])
        cx=int(x1+x2)//2
        cy=int(y1+y2)//2
        #if 'armature body' in d: 

            # For detect object only inside polylines green boundary
        results=cv2.pointPolygonTest(np.array(area,np.int32),((cx,cy)),False)
        if results>=0:
            print(results)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
            cv2.putText(frame,str(d),(x1,y1),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
            list1.append([cx]) 
            cv2.polylines(frame,[np.array(area,np.int32)],True,(0,255,0),2)
            a= len(list1)
            print(a)
    cv2.putText(frame,str(a),(136,24),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
    cv2.imshow("Live web cam ",frame)
    #cv2.setMouseCallback("FRAME",POINTS)
   

    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()