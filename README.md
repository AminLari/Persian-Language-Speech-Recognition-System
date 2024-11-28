# Persian Language Speech Recognition System

## Overview
This project implements a speech recognition system for Persian language commands using a Convolutional Neural Network (CNN). The system is developed in Python, leveraging TensorFlow and Keras, and it classifies spoken commands based on an audio dataset.
<p> <img src="https://github.com/user-attachments/assets/4b2d8a18-cafd-42b1-9e3a-3f916fea5dee" width="1000"> </p> 

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
1. **Clone the repository**
   ```bash
   git clone https://github.com/AminLari/Persian-Language-Speech-Recognition-System.git
   cd Persian-Language-Speech-Recognition-System

2. **Install the dependencies**
   ```bash
   pip install tensorflow librosa keras numpy

## Usage
1. **Extract the dataset:**
   After downloading the repository, you need to extract the voice command dataset.
   ```bash
   unzip Voices-wav.zip

2. **Run the training and evaluation script:**
   To train the CNN model on the dataset, run the following Python script. The script will load the dataset, preprocess the audio data, and train the model.
   ```bash
   python CI_Lab_Project.py

4. **Test the model with a sample audio:**
   After training the model, you can test it with a sample audio file. The provided sample file is roshan.wav. Modify the script to classify your own audio files if needed.
   ```bash
   # Modify script to test a different file
   test_audio = 'path/to/your/audio.wav'
   prediction = model.predict(test_audio)
   print(f'Predicted Class: {prediction}')

## Results

The developed framework extracts time-frequency domain features from audio signals using MFCC.
<p> <img src="https://github.com/user-attachments/assets/0c2eb23f-16e9-4d3d-9d79-8918b0cb4816" width="1000"> </p> 

The model can be evaluated using the provided sample audio file (roshan.wav).
- Sample Input: roshan.wav
- Predicted Output: The model predicts the class label for the input audio file.
<p> <img src="https://github.com/user-attachments/assets/64fb83b7-e784-4258-80d0-9ee6547e4693" width="1000"> </p> 

## Contact
For questions or suggestions, please contact Amin Lari.
