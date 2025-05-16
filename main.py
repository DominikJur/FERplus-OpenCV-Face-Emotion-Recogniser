import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import torch

from src.dataset import EMOTIONS_TO_IDX, data_transforms
from src.utils import get_model
from src.face import recognize_expression

if __name__ == "__main__":
    
    print("Running face recognition script...")
    
    transform = data_transforms["test"]
    emotion_labels = {v: k for k, v in EMOTIONS_TO_IDX.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = get_model(
        num_classes=len(emotion_labels),
        device=device,
    )
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    recognize_expression(
        model,
        face_cascade,
        device,
        emotion_labels,
        transform,
    )
