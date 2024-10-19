# Real-Time Facial Reconstruction from CCTV Footage

## Overview

This project aims to develop a machine learning solution capable of accurately reconstructing human faces from low-quality CCTV footage. The solution enhances blurry or distorted images, enabling investigators to identify suspects effectively.

## Problem Statement

The challenge is to create a robust application that can perform real-time facial reconstruction across diverse CCTV footage scenarios. The application will be demonstrated at IIT Guwahati, showcasing advanced features and real-world applicability in security and surveillance.

## Motivation

The increasing prevalence of surveillance cameras in public spaces has created a need for advanced technologies that can accurately identify individuals from low-quality footage. This project addresses this need by utilizing state-of-the-art machine learning techniques to enhance facial recognition capabilities.

## Features

- **Face Detection**: Utilizes MTCNN to detect faces in real-time from CCTV footage.
- **Face Alignment**: Geometrically aligns detected faces for consistent reconstruction.
- **Image Enhancement**: Employs Single Image Super-Resolution (SISR) techniques to improve image quality.
- **Feature Refinement**: Utilizes Graph Convolutional Networks (GCNs) for refining facial features.
- **Facial Reconstruction**: Implements GANs or Autoencoders for high-quality face reconstruction.
- **Facial Recognition**: Integrates ArcFace or FaceNet for robust face matching.
- **Post-Processing**: Includes noise reduction and evaluation metrics (PSNR, SSIM).

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - OpenCV for image processing
  - MTCNN for face detection
  - TensorFlow or PyTorch for deep learning models
  - NumPy for numerical operations
  - Scikit-learn for machine learning utilities
  - FastAPI or Flask for web deployment
- **Frameworks**: 
  - GANs (e.g., StyleGAN or ProGAN)
  - Graph Convolutional Networks (GCNs)

## System Architecture

The system architecture follows a multi-step process:

1. **Face Detection**: MTCNN detects faces and provides bounding boxes.
2. **Face Alignment**: Detected faces are aligned using geometric transformations.
3. **Image Enhancement**: SISR techniques improve the quality of detected faces.
4. **Feature Refinement**: GCNs refine facial features based on landmark positions.
5. **Facial Reconstruction**: GANs or Autoencoders reconstruct the face images.
6. **Facial Recognition**: The system recognizes faces using ArcFace or FaceNet.
7. **Post-Processing**: Noise reduction techniques are applied, and evaluation metrics are calculated.

## Installation

### Prerequisites

Ensure you have Python 3.x installed on your machine. You will also need to install the following dependencies:

```bash
pip install numpy opencv-python mtcnn tensorflow torch torchvision fastapi uvicorn scikit-learn
