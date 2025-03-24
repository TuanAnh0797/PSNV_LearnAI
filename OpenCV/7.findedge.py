import imutils
import cv2
imgori = cv2.imread('bi_ngo_2.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow("myimg1",imgori)
#img = cv2.adaptiveThreshold(imgori,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,33,5)
img = cv2.Canny(imgori,150,700)
cv2.imshow("myimg2",img)
cv2.waitKey()
cv2.destroyAllWindows()