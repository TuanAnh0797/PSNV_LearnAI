import cv2
import numpy as np

# Khởi tạo camera
cam = cv2.VideoCapture(0)  # 0 là camera mặc định, thay đổi nếu cần

# Ngưỡng màu da trong không gian YCrCb
min_YCrCb = np.array([0, 135, 85], dtype=np.uint8)
max_YCrCb = np.array([255, 180, 135], dtype=np.uint8)

# Ngưỡng màu mắt trong không gian YCrCb
min_YCrCb_eyes = np.array([0, 100, 130], dtype=np.uint8)  # Ngưỡng thấp
max_YCrCb_eyes = np.array([255, 150, 180], dtype=np.uint8)  # Ngưỡng cao

while True:
    ret, frame = cam.read()
    if ret:
        # Tạo bản sao của khung hình để xử lý
        original_frame = frame.copy()
        imageYCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        
        # Phát hiện vùng da
        skinRegionYCrCb = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)
        contours, _ = cv2.findContours(skinRegionYCrCb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Chuyển sang ảnh xám để phát hiện vùng đen (mắt)
        gray_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 2000:  # Lọc theo diện tích
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h

                # Kiểm tra tỷ lệ khung hình để xác định vùng mặt
                if aspect_ratio > 0.6 and aspect_ratio < 1.5:
                    # Cắt phần khuôn mặt
                    face_region = original_frame[y:y + h, x:x + w]
                    
                    # Chuyển vùng mặt sang không gian YCrCb
                    imageYCrCb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2YCR_CB)
                    
                    # Phát hiện vùng mắt trong khuôn mặt
                    skinRegionYCrCb_eyes = cv2.inRange(imageYCrCb_face, min_YCrCb_eyes, max_YCrCb_eyes)
                    contours_eyes, _ = cv2.findContours(skinRegionYCrCb_eyes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Kiểm tra nếu có contour mắt
                    if len(contours_eyes) > 0 :
                        # Vẽ hình chữ nhật quanh khuôn mặt và mắt
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        # Có thể vẽ thêm hình chữ nhật quanh mắt nếu cần
                        # for eye_contour in contours_eyes:
                        #     ex, ey, ew, eh = cv2.boundingRect(eye_contour)
                        #     cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

        # Hiển thị kết quả
        cv2.imshow("Person", frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
            
cam.release()
cv2.destroyAllWindows()
