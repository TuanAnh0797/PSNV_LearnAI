import cv2
# a = cv2.imread('bi_ngo_2.jpg',cv2.IMREAD_REDUCED_GRAYSCALE_8)
# cv2.imshow("anh",a)
# cv2.waitKey()
# cv2.destroyAllWindows()
#mở
cam = cv2.VideoCapture(0)
#Đọc
while(True):
    #ret trả về true nếu đọc được, frame ảnh được được
    ret,frame = cam.read()
    if ret:
        cv2.imshow("webcam",frame)
    press_key = cv2.waitKey(1)
    if press_key == ord('q'):
        break
cam.release
cv2.destroyAllWindows()