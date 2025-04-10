import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv("face_skin_dataset.csv")

X = df[["area", "aspect_ratio"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Lưu mô hình
joblib.dump(model,r"C:\Users\Administrator\Desktop\StudyAI\Day5\\"+"face_classifier.pkl")
print("✅ Đã lưu mô hình vào face_classifier.pkl")
