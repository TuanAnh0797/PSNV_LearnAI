#Cài thêm thư viện pip install imutils
import cv2
import imutils

import cv2

imgori = cv2.imread('bi_ngo_2.jpg')
cv2.imshow("myimg1",imgori)

imgrotation = imutils.rotate(imgori,90)

cv2.imshow("myimg2",imgrotation)

cv2.waitKey()
cv2.destroyAllWindows()