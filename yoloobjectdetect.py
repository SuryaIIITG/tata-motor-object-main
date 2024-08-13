
import torch
import numpy as np
import cv2
from time import time

  #  This code from counting .py

import os
import torch
import numpy as np
import pandas as pd
import torchvision
import yolov5
from yolov5 import utils  


points=[]
def POINTS(event, x, y, flags, param): 
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)  
        

cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME',POINTS)  



class Detection:
    """
    Class implements Yolo5 model to make inferences on a  video using Opencv2.
    """

    def __init__(self, capture_index, model_name):
        """
        Initializes the class with  url and output file.
        :param url: Has to be as  URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.capture_index = capture_index
        self.model = self.load_model(model_name) 
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device) 

    def get_video_capture(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """
      
        return cv2.VideoCapture(self.capture_index)

    def load_model(self, model_name):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)           # 

        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)                             # 
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]   
        results = self.model(frame)  
        df=results.pandas().xyxy[0]                               # create a Dataframe  
        print(df[['name','confidence']])                            
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]  
        #print(cord)     
        
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        count=0                        # for counting purpose 
        n = len(labels) 
        x_shape, y_shape = frame.shape[1], frame.shape[0]  
        for i in range(n): 
            row = cord[i]                        # return cordinate of input images
            #print(row)
            if row[4] >= 0.3:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                count+=1
        
        print(f'iterate is : {count}') 
        return frame 




    # Code  from counting.py
    # Create a function for counting
    """   
    def counte():
        #display = utils.notebook_init() 

        model =torch.hub.load(os.getcwd(),'custom',source='local', path="C:\\Users\\belal\\OneDrive\\Desktop\\YOLOv5-Flask-master\\runs\\train\\exp47\\weights\\best.pt",force_reload=True)
        vid_frame=model(frame)
        vid_frame.pandas().xyxy[0]

        # create Dataframe via web cam
        
        index,counts=np.unique(video_frame["name"].values, return_counts=True) 

        data2={"Part Number" :index,"count":counts} 
        df1=pd.DataFrame(data2)
        print(df1) 

        return counte

           """

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        cap = self.get_video_capture()  
        assert cap.isOpened()
      
        while True:
            ret, frame = cap.read()
            assert ret
            frame = cv2.resize(frame, (800,600))

            # Counting code
            #model =torch.hub.load(os.getcwd(),'custom',source='local', path="C:\\Users\\belal\\OneDrive\\Desktop\\YOLOv5-Flask-master\\runs\\train\\exp47\\weights\\best.pt",force_reload=True)
            #vid_frame=model(frame)  
            #vid_frame=vid_frame.pandas().xyxy[0] 

            # code for counting
            #vid_frame=model(frame)
            #vid_frame.pandas().xyxy[0]




            start_time = time()
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            
            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)   
            #print(f"Frames Per Second : {fps}")
             
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
            
            cv2.imshow('YOLOv5 Detection', frame)
        
            #if cv2.waitKey(5) & 0xFF == 27:
            if cv2.waitKey(25) & 0xFF == ord('q'):    
                break
      
        cap.release() 
        




# Create a new object and execute.
detector = Detection(capture_index=1, model_name='C:\\Users\\Alime\\OneDrive\\Desktop\\YOLOv5-Flask2\\yolov5\\runs\\train\\exp47\\weights\\best.pt') 
detector()



