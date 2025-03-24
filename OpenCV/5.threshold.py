import imutils

import cv2

imgori = cv2.imread('bi_ngo_2.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow("myimg1",imgori)

result,img = cv2.threshold(imgori,127,255,cv2.THRESH_BINARY);

cv2.imshow("myimg2",img)

cv2.waitKey()
cv2.destroyAllWindows()