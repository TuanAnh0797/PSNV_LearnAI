import cv2
import numpy as np
import os
import pandas as pd

min_YCrCb = np.array([0, 133, 77], np.uint8)
max_YCrCb = np.array([235, 173, 127], np.uint8)

def extract_features_from_image(image_path, label):
    img = cv2.imread(image_path)
    if img is None:
        return None

    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    mask = cv2.inRange(img_YCrCb, min_YCrCb, max_YCrCb)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    x, y, w, h = cv2.boundingRect(largest)
    aspect_ratio = w / float(h)

    return [area, aspect_ratio, label]

data = []

for folder, label in [(r"C:\Users\Administrator\Desktop\StudyAI\Day5\faces", 1), (r"C:\Users\Administrator\Desktop\StudyAI\Day5\non_faces", 0)]:
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        features = extract_features_from_image(path, label)
        if features:
            data.append(features)

df = pd.DataFrame(data, columns=["area", "aspect_ratio", "label"])
df.to_csv("face_skin_dataset.csv", index=False)
print("✅ Dữ liệu đã được lưu vào face_skin_dataset.csv")
