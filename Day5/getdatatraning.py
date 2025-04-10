import cv2
import datetime

video = cv2.VideoCapture(1)
while(True):

    Ret,Frame = video.read()
    if(Ret):
        cv2.imshow("img",Frame)
    waitkey = cv2.waitKey(1)
    if(waitkey == ord('f')):
        filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        print(filename)
        rs = cv2.imwrite(r"C:\Users\Administrator\Desktop\StudyAI\Day5\faces\\"+filename+".jpg",Frame)
        print(rs) 
    if(waitkey == ord('n')):
        filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        print(filename)
        rs = cv2.imwrite(r"C:\Users\Administrator\Desktop\StudyAI\Day5\non_faces\\"+filename+".jpg",Frame)
        print(rs)

    if(waitkey == ord('q')):
        break

video.release()
cv2.destroyAllWindows()