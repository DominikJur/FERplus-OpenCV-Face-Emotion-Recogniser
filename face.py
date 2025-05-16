import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image
from dataset import EMOTIONS_TO_IDX, data_transforms
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

transform = data_transforms["test"]

emotion_labels = {v: k for k, v in EMOTIONS_TO_IDX.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
num_classes = len(EMOTIONS_TO_IDX.values())
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
model.load_state_dict(
    torch.load("ResNet_epoch_66.pth", map_location=torch.device("cpu"))
)
print("Model loaded successfully")


model.eval()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if face_cascade.empty():
    print("Error: Could not load face cascade classifier XML file")
    exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera opened successfully. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame.")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        face_img = frame[y : y + h, x : x + w]
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = Image.fromarray(face_img)
        face_img = transform(face_img)
        face_img = face_img.unsqueeze(0)
        face_img = face_img.to(device)

        with torch.no_grad():
            outputs = model(face_img)
            _, predicted = torch.max(outputs, 1)
            emotion = emotion_labels[predicted.item()]

            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence = probs[predicted].item() * 100

        text = f"{emotion}: {confidence:.1f}%"
        cv2.putText(
            frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )

    cv2.putText(
        frame,
        "Facial Emotion Recognition - Press 'q' to quit",
        (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.imshow("Facial Emotion Recognition", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
