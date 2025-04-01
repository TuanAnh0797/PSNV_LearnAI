import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
min_YCrCb = np.array([0,133,77],np.uint8)
max_YCrCb = np.array([235,173,127],np.uint8)
min_YCrCb_black = np.array([0, 0, 0], np.uint8)  # Màu đen hoàn toàn
max_YCrCb_black = np.array([50, 255, 255], np.uint8)  # Phạm vi cho màu đen

cam =  cv2.VideoCapture(0)
while(True):
    ret,frame = cam.read()
    if ret:
        #frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        imageYCrCb = cv2.cvtColor(frame,cv2.COLOR_BGR2YCR_CB)
        skinRegionYCrCb = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
        contours, _ = cv2.findContours(skinRegionYCrCb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 2000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h  
                if aspect_ratio > 0.6 and aspect_ratio < 1.5:  
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow("Person",frame)
        key = cv2.waitKey(1)
    if key == ord('q'):
        cam.release()
        cv2.destroyAllWindows()
