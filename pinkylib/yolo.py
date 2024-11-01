import cv2
from IPython.display import display, clear_output, Image
import time

import numpy as np
from glob import glob

from ultralytics import YOLO

class Yolo():
    def __init__(self): 
        super().__init__()
        self.model = None

    def set_model(self, train_model="best.pt"):
        self.model = YOLO(train_model)

    def detect_yolo(self, frame):
        results = self.model(frame, verbose=False)
        clss = results[0].boxes.cls.numpy()
        
        annotated_frame = results[0].plot()
        
        return clss, annotated_frame

        
