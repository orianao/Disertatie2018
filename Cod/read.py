import os.path
import wave
import struct
import numpy as np
import pandas as pd
from pylab import *
import matplotlib.pyplot as plt
import audioop
import keras
from sklearn.preprocessing import normalize
from keras.models import Sequential
from keras.layers import Conv1D, MaxPool1D, GlobalAvgPool1D, Dropout, BatchNormalization, Dense,Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.regularizers import l2

topdir = "heartbeat-sounds"
exten = '.wav'
Cframerate = 44100

recordings = []
x_train = []
y_train = []


def plotit(t,samples,name):
	fig = plt.figure()
	fig.suptitle(name.split('_')[0])
	the_plot = fig.add_subplot(111)
	the_plot = plot(t, samples)
	plt.show(block=False)

def is_peak (x,left,right):
	if np.amax(left)<x and np.amax(right)<x:
		return True
	return False

def get_peaks(samples):
	sol = []
	std = np.std(samples)
	for i in range(len(samples)):
		if i>Cframerate and i<len(samples)-Cframerate and samples[i]>std*2.8:
			if is_peak(samples[i],samples[i-int(Cframerate/5):i],samples[i+1:i+int(Cframerate/5)]):
				sol.append(i)
	return sol;

def add_to_recs(samples,name,n,dirpath='a_a'):
	i=1
	peaks_list = get_peaks(samples)
	for i in peaks_list:
		recordings.append((samples[i-Cframerate:i+Cframerate],name.split('_')[0],dirpath.split('_')[1]))
		x_train.append(samples[i-Cframerate:i+Cframerate])
		
		if name.split('_')[0]=="normal":
			y_train.append(0)
		elif name.split('_')[0] =="artifact":
			y_train.append(1)
		elif name.split('_')[0]=="extrahls":
			y_train.append(2)
		else:
			y_train.append(3)
		
		i=i+1

def readData():
	no=0

	for dirpath, dirnames, files in os.walk(topdir):
		for name in files:
			if name.lower().endswith(exten):
				filepath = os.path.join(dirpath, name)
				print(filepath)
				f = wave.open(filepath)

				frames = f.readframes(-1)

				framerate = f.getframerate()
				#state = None

				#frames, state = audioop.ratecv(frames, 1, 1, framerate, Cframerate, state)

				samples = list(struct.unpack('h' * f.getnframes(), frames))
				t = [float(i)/framerate for i in range(len(samples))]
				
				fac_norm = np.linalg.norm(samples)
				samples = (samples / fac_norm) * 100
				print (no+1, np.std(samples))
				# samples = samples / np.amax(samples)

				# plotit(t,samples,name)

				add_to_recs(samples,name,f.getnframes())

				no = no + 1


if __name__ == '__main__':
	readData()
	recordingsDF = pd.DataFrame(data=recordings, columns = ["samples","label","type"])
	print(recordingsDF.head())
	recordingsDF.to_csv("recordings.csv")

	input()

	x_train = np.stack(recordingsDF['samples'].values, axis=0)
	y_train = keras.utils.to_categorical(y_train)


	x_train = x_train[:,:,np.newaxis]


	model = Sequential()
	model.add(Conv1D(filters=4, 
		kernel_size=9, 
		activation='relu',
		kernel_regularizer = l2(0.05),
		input_shape=x_train.shape[1:]))
	model.add(Conv1D(filters=4, kernel_size=9, activation='relu'))
	model.add(MaxPool1D(pool_size=5))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(200, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4, activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adam(),
	              metrics=['accuracy'])

	model.fit(x_train, y_train,
	          batch_size=50,
	          epochs=20,
	          verbose=1,
	          validation_data=(x_train, y_train))
	score = model.evaluate(x_train, y_train, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	input()

