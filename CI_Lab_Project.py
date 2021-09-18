import numpy as np
import pandas as pd
import sklearn
from gpiozero.pins.mock import MockFactory
from scipy.io import wavfile
import librosa
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import warnings
import os
import IPython.display as ipd
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from matplotlib import pyplot as plt
import random
import gpiozero
from gpiozero import LED

#import RPi.GPIO as GPIO


np.set_printoptions(threshold=np.inf)

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Relative address of main audio folder
train_audio_path = 'Voices-wav/'
labels = os.listdir(train_audio_path)
print(labels)

no_of_recordings = []
for label in labels:
    waves = [f for f in os.listdir(train_audio_path + '/' + label) if f.endswith('.wav')]
    no_of_recordings.append(len(waves))
print(no_of_recordings)

# draw number of audio files for each label
f3 = plt.figure(figsize=(12, 5))
ax3 = f3.add_subplot(1, 1, 1)
index = np.arange(len(labels))
ax3.bar(labels, no_of_recordings)
ax3.set_xlabel('Commands', fontsize=12)
ax3.set_ylabel('Number of recordings', fontsize=12)
ax3.set_title('Number of recordings for each command')

duration_of_recordings = []
for label in labels:
    waves = [f for f in os.listdir(train_audio_path + '/' + label) if f.endswith('.wav')]
    for wav in waves:
        sample_rate, samples = wavfile.read(train_audio_path + '/' + label + '/' + wav)
        duration_of_recordings.append(float(len(samples) / sample_rate))

# Drawing duration of audio files
f4 = plt.figure(figsize=(12, 5))
ax4 = f4.add_subplot(1, 1, 1)
ax4.hist(np.array(duration_of_recordings))
ax4.set_xlabel('Duration(s)', fontsize=12)
ax4.set_ylabel('Number of recordings', fontsize=12)
for label in ax4.get_xticklabels():
    label.set_rotation(60)
ax4.set_title('Duration of recordings')

all_wave = []
all_label = []
for label in labels:
    print(label)
    waves = [f for f in os.listdir(train_audio_path + '/' + label) if f.endswith('.wav')]
    for wav in waves:
        samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr=16000)
        samples = librosa.resample(samples, sample_rate, 8000)

        samples = librosa.util.fix_length(samples, 20000, axis=-1)

        # draw each audio wave in time domain
        '''f5 = plt.figure(figsize=(7, 3))
        ax5 = f5.add_subplot(1, 1, 1)
        librosa.display.waveplot(data, sr=samplerate)
        plt.show()'''

        # feature extraction using MFCC
        n_mfcc = 20
        mfcc = librosa.feature.mfcc(samples, sr=8000, n_mfcc=n_mfcc)
        # print(mfcc.shape)

        # normalization
        scaler = sklearn.preprocessing.StandardScaler()
        mfcc_scaled = scaler.fit_transform(mfcc)

        # Drawing MFCC coefficients and visualization
        '''f6 = plt.figure(figsize=(6, 9))
        ax6 = f6.add_subplot(1, 1, 1)
        ax6.set_ylabel("MFCC (log) coefficient")
        ax6.set_title("Audio visualization")
        ax6.imshow(np.swapaxes(mfcc, 0, 1))
        plt.show()'''

        row = np.matrix.flatten(mfcc_scaled)
        # print(row.shape)

        if len(row) == 800:
            all_wave.append(row)
            all_label.append(label)

le = LabelEncoder()
y = le.fit_transform(all_label)
classes = list(le.classes_)
y = np_utils.to_categorical(y, num_classes=len(labels))
all_wave = np.array(all_wave).reshape(-1, 800, 1)

# Dividing dataset into test and train subsets
x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_wave), np.array(y), stratify=y, test_size=0.2,
                                            random_state=777, shuffle=True)

K.clear_session()

# Input layer
inputs = Input(shape=(800, 1))

# First Conv1D layer
conv = Conv1D(8, 13, padding='valid', activation='relu', strides=1)(inputs)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

# Second Conv1D layer
conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

# Third Conv1D layer
conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

# Fourth Conv1D layer
conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.4)(conv)

# Flatten layer
conv = Flatten()(conv)

# Dense Layer 1
conv = Dense(512, activation='relu')(conv)
conv = Dropout(0.4)(conv)

# Dense Layer 2
conv = Dense(256, activation='relu')(conv)
conv = Dropout(0.4)(conv)

# Dense Layer 3
conv = Dense(128, activation='relu')(conv)
conv = Dropout(0.4)(conv)

# Output layer
outputs = Dense(len(labels), activation='softmax')(conv)

model = Model(inputs, outputs)
print(model.summary())

model.compile(loss='kullback_leibler_divergence', optimizer='adam', metrics=['accuracy'])

if os.path.exists('classification_model35.h5'):
    model = load_model('classification_model35.h5')
else:
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.00001)
    mc = ModelCheckpoint('best_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    history = model.fit(x_tr, y_tr, epochs=200, callbacks=[es, mc], batch_size=32, validation_data=(x_val, y_val))
    model.save('classification_model35.h5')

    # Drawing model accuracy
    f1 = plt.figure()
    ax1 = f1.add_subplot(1, 1, 1)
    ax1.plot(history.history['accuracy'], label='train')
    ax1.plot(history.history['val_accuracy'], label='test')
    ax1.legend()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel("Accuracy")
    ax1.set_title('Training accuracy')

    # Drawing model loss
    f2 = plt.figure()
    ax2 = f2.add_subplot(1, 1, 1)
    ax2.plot(history.history['loss'], label='train')
    ax2.plot(history.history['val_loss'], label='test')
    ax2.legend()
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Error')


def predict(audio):
    prob = model.predict(audio.reshape(1, 800, 1))
    ind = np.argmax(prob[0])
    return classes[ind]


# GPIO
#led = LED(17)
#gpiozero.Device.pin_factory = MockFactory()

led_pin = 17
#GPIO.setwarnings(False)
#GPIO.setmode(GPIO.BOARD)
#GPIO.setup(17, GPIO.OUT, initial=GPIO.LOW)


# testing on hardware
def test():
    samples, sample_rate = librosa.load('roshan.wav', sr=16000)
    samples = librosa.resample(samples, sample_rate, 8000)

    samples = librosa.util.fix_length(samples, 20000, axis=-1)

    # feature extraction using MFCC
    n_mfcc = 20
    mfcc = librosa.feature.mfcc(samples, sr=8000, n_mfcc=n_mfcc)
    # print(mfcc.shape)

    # normalization
    scaler = sklearn.preprocessing.StandardScaler()
    mfcc_scaled = scaler.fit_transform(mfcc)

    row = np.matrix.flatten(mfcc_scaled)
    print('-'*80, '\nNow let\'s test on hardware:')
    print("Input label:{0}\t|\tNetwork predicted label:{1}".format('Roshan', predict(row)))

    if predict(row) == 'Roshan':
        print('Turning on LED connected to GPIO 17 port...')
        #led.on()
        #GPIO.output(led_pin, GPIO.HIGH)
    elif predict(row) == 'Tarik':
        print('Turning off LED connected to GPIO 17 port...')
        #led.off()
        #GPIO.output(led_pin, GPIO.LOW)
    else:
        print('Invalid command!')


# test model on 10 random samples
for i in range(10):
    index = random.randint(0, len(x_val) - 1)
    samples = x_val[index].ravel()
    ipd.Audio(samples, rate=8000)
    print("Audio label:{0}\t|\tNetwork predicted label:{1}".format(classes[np.argmax(y_val[index])], predict(samples)))

# Creating .csv file of dataset
out = np.reshape(all_wave, (np.sum(no_of_recordings), -1))
df = pd.DataFrame(out)
df.to_csv('audioNoisy.csv')

test()

plt.show()
