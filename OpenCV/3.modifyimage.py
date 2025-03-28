import cv2

imgori = cv2.imread('bi_ngo_2.jpg')
imgblack = cv2.cvtColor(imgori,cv2.COLOR_BGR2GRAY)
cv2.imshow("myimg",imgblack)
#Resize
#imgresiz = cv2.resize(imgori,(50,50))
#Resize vector
imgresiz = cv2.resize(imgori,dsize=None,fx=0.5, fy=0.5)

cv2.imshow("myimg2",imgresiz)



cv2.waitKey()
cv2.destroyAllWindows()