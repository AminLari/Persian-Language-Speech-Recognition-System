# Persian Language Speech Recognition System

## Overview
This project implements a speech recognition system for Persian language commands using a Convolutional Neural Network (CNN). The system is developed in Python, leveraging TensorFlow and Keras, and it classifies spoken commands based on an audio dataset.

## Features
- **CNN Model:** Trained using a dataset of Persian commands.
- **Dataset:** Custom Persian voice commands in WAV format.
- **Example Audio:** Includes a sample audio file (`roshan.wav`) for testing.

## File Structure
- `CI_Lab_Project.py`: Python script that builds and trains the CNN model.
- `classification_model35.h5`: Pretrained CNN model for classification.
- `Voices-wav.zip`: Compressed dataset containing voice recordings.
- `roshan.wav`: Sample audio file used for testing the model.

---

## Installation
To run this project, you'll need Python and several libraries. Follow these steps:

### Prerequisites
- Python 3.x
- TensorFlow
- Keras
- NumPy
- LibROSA

### Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/AminLari/Persian-Language-Speech-Recognition-System.git
   cd Persian-Language-Speech-Recognition-System
