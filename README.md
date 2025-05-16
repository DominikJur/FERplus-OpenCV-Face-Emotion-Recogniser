# 😀 Facial Emotion Recognition

A real-time facial emotion recognition system that uses computer vision and deep learning to detect and classify facial expressions into seven basic emotions.

## Overview

This project uses a pre-trained ResNet18 model and OpenCV to recognize facial emotions in real-time from a webcam feed. The system can detect faces and classify emotions with reasonable accuracy.

### ✨ Features

- Real-time facial emotion detection from webcam
- Classification of 7 basic emotions: angry, disgust, fear, happy, neutral, sad, surprise
- Pre-trained ResNet18 model (61.81% accuracy on test set)
- User-friendly interface with emotion labels and confidence scores

## 📂 Project Structure

```
├── data/                  # Dataset directory
│   ├── train/             # Training data organized by emotion folders
│   └── test/              # Test data organized by emotion folders
├── models/                # Saved model weights
│   └── ResNet_epoch_66.pth # Pre-trained model
├── notebooks/             # Jupyter notebooks
│   └── training.ipynb     # Model training notebook
├── src/                   # Source code
│   ├── dataset.py         # Dataset class and data loaders
│   ├── face.py            # Face detection and emotion recognition
│   └── utils.py           # Utility functions
├── main.py                # Main application script
├── .gitignore             # Git ignore file
├── LICENSE                # MIT License
└── README.md              # Project documentation
```

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- PIL (Pillow)
- tqdm

## 🔧 Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/facial-emotion-recognition.git
   cd facial-emotion-recognition
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```
   pip install torch torchvision opencv-python numpy matplotlib scikit-learn pillow tqdm
   ```

## 🚀 Usage

1. Run the main application to start facial emotion recognition with your webcam:
   ```
   python main.py
   ```

2. The application will open your webcam and start detecting faces and emotions in real-time.

3. Press 'q' to quit the application.

## Dataset

This project uses the FER2013 dataset or a similar facial emotion dataset organized in the following structure:

```
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

Each subfolder contains facial images expressing the corresponding emotion.

## 🧠 Model

The system uses a pre-trained ResNet18 model modified for emotion classification:

- Base model: ResNet18 pre-trained on ImageNet
- Final layer replaced to output 7 emotion classes
- Trained on facial emotion dataset
- Current test accuracy: 61.81%

## Training

If you want to train the model yourself, you can run the Jupyter notebook:

```
jupyter notebook notebooks/training.ipynb
```

The notebook contains code to:
1. Load and preprocess the dataset
2. Create and configure the model
3. Train the model
4. Evaluate the model's performance

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The ResNet18 implementation is based on the torchvision library
- Face detection uses OpenCV's Haar Cascade Classifier

## 🤔 How It Works

1. Capture frames from webcam
2. Detect faces using OpenCV's Haar Cascade
3. Extract face region from image
4. Preprocess the face (resize, normalize)
5. Feed the face image to the ResNet18 model
6. Get emotion prediction and confidence score
7. Display result with bounding box and emotion label
