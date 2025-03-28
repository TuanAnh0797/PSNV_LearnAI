# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# img = cv2.imread("Day4/person.jpg")

# imgrs = cv2.resize(img,dsize=None,fx=0.2,fy=0.2)



# cv2.imshow("Person",imgrs)

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
min_YCrCb = np.array([0,133,77],np.uint8)
max_YCrCb = np.array([235,173,127],np.uint8)

#datasetspath = r"C:\Users\Administrator\Desktop\StudyAI\Day4\train"

# for subpath in os.listdir(datasetspath):
#     sub_path = os.path.join(datasetspath, subpath)  # Tạo đường dẫn đầy đủ
#     index = 0
#     for img_file in os.listdir(sub_path):
#         path_img = os.path.join(sub_path, img_file)
#         sourceimg = cv2.imread(path_img)
#         imageYCrCb = cv2.cvtColor(sourceimg,cv2.COLOR_BGR2YCR_CB)
#         skinRegionYCrCb = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
#         #skinYCrCb = cv2.bitwise_and(img_file, img_file, mask = skinRegionYCrCb)
        
#         data = pd.DataFrame({'x': [img_file], 'y': [skinRegionYCrCb]})
#         # Ghi vào file CSV (append nếu muốn ghi vào cuối file mà không ghi đè)
#         data.to_csv('output.csv', mode='a', header=False, index=False)   

#         index +=1
#         if index == 10:
#             break






cam =  cv2.VideoCapture(0)

while(True):
    ret,frame = cam.read()
    if ret:
        imageYCrCb = cv2.cvtColor(frame,cv2.COLOR_BGR2YCR_CB)
        skinRegionYCrCb = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
        skinYCrCb = cv2.bitwise_and(frame, frame, mask = skinRegionYCrCb)
        cv2.imshow("Person",skinYCrCb)
        key = cv2.waitKey(1)
    if key == ord('q'):
        cam.release()
        cv2.destroyAllWindows()



