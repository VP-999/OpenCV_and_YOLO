# OpenCV Computer Vision Project

## Project Description
This project demonstrates various computer vision applications using OpenCV and YOLO (You Only Look Once). It includes features for face detection, eye detection, smile detection, emotion recognition, and custom model training for object detection. The project is designed for real-time image and video processing, making it suitable for applications like surveillance, emotion-based interactions, and general object recognition.

## Features
- **Face Detection**: Real-time detection of faces in images or videos using Haar cascades.
- **Eye and Smile Detection**: Identifies eyes and smiles within detected faces.
- **Emotion Recognition**: Analyzes facial expressions to detect emotions.
- **Object Detection with YOLO**: Uses pre-trained YOLO models for accurate object detection and custom training.
- **Model Training**: Scripts for training custom YOLO models with provided datasets.
- **Visualization**: Outputs processed images, screenshots, and video clips for testing.

## Installation
1. Ensure you have Python installed (version 3.8 or higher).
2. Install required dependencies:
   ```bash
   pip install opencv-python
   pip install ultralytics  # For YOLO
   pip install numpy
   pip install torch  # If using GPU for YOLO
