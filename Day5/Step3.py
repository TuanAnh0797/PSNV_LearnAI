import cv2
import numpy as np
import joblib

model = joblib.load(r"C:\Users\Administrator\Desktop\StudyAI\Day5\\"+"face_classifier.pkl")

min_YCrCb = np.array([0,133,77],np.uint8)
max_YCrCb = np.array([235,173,127],np.uint8)

cam = cv2.VideoCapture(1)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    imageYCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    mask = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1500:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)

            prediction = model.predict([[area, aspect_ratio]])[0]

            if prediction == 1:
                label = "Face"
                color = (0, 255, 0)
            # else:
            #     label = "Not face"
            #     color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Skin-based Face Detection (ML)", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
