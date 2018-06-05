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
from sklearn.model_selection import train_test_split
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


def plotit(t, samples, name):
	fig = plt.figure()
	fig.suptitle(name.split('_')[0])
	the_plot = fig.add_subplot(111)
	the_plot = plot(t, samples)
	plt.show(block=False)


def is_peak (x, left, right):
	return np.amax(left) < x and np.amax(right) < x


def get_peaks(samples):
	sol = []
	std = np.std(samples)
	for i in range(len(samples)):
		if i > Cframerate and i < len(samples) - Cframerate and samples[i] > std * 2.8:
			if is_peak(samples[i], samples[i - Cframerate // 5 : i], samples[i + 1 : i + Cframerate // 5]):
				sol.append(i)
	return sol


def add_to_recs(samples, name, n, dirpath='a_a'):
	global y_train
	global x_train
	peaks_list = get_peaks(samples)
	for i in peaks_list:
		recordings.append((samples[i - Cframerate : i + Cframerate], name.split('_')[0], dirpath.split('_')[1]))
		recordings.append((samples[i + 20 - Cframerate : i + 20 + Cframerate], name.split('_')[0], dirpath.split('_')[1]))
		recordings.append((samples[i - 20 - Cframerate : i - 20 + Cframerate], name.split('_')[0], dirpath.split('_')[1]))
		x_train.append(samples[i - Cframerate : i + Cframerate])
		x_train.append(samples[i + 20 - Cframerate : i + 20 + Cframerate])
		x_train.append(samples[i - 20 - Cframerate : i - 20 + Cframerate])
		
		if name.split('_')[0] == "normal":
			y_train += [0] * 3
		elif name.split('_')[0] == "artifact":
			y_train += [1] * 3
		elif name.split('_')[0] == "extrahls":
			y_train += [2] * 3
		else:
			y_train += [3] * 3

def split_data(x_train, y_train):
	aux = [*zip(x_train,y_train)]
	np.random.shuffle(aux)
	x, y = zip(*aux)
	p = int(len(x_train) * 0.8)
	return x[ : p], y[ : p], x[p : ], y[p : ]


def readData():
	no=0

	for dirpath, dirnames, files in os.walk(topdir):
		for name in files:
			if name.lower().endswith(exten):
				filepath = os.path.join(dirpath, name)
				#print(filepath)
				f = wave.open(filepath)

				frames = f.readframes(-1)

				framerate = f.getframerate()
				#state = None

				#frames, state = audioop.ratecv(frames, 1, 1, framerate, Cframerate, state)

				samples = list(struct.unpack('h' * f.getnframes(), frames))
				t = [float(i)/framerate for i in range(len(samples))]
				
				fac_norm = np.linalg.norm(samples)
				samples = (samples / fac_norm) * 100
				#print (no+1, np.std(samples))
				# samples = samples / np.amax(samples)

				#if no < 15:
				#	plotit(t,samples,name)

				add_to_recs(samples,name,f.getnframes())

				no = no + 1


if __name__ == '__main__':
	readData()
	recordingsDF = pd.DataFrame(data=recordings, columns = ["samples","label","type"])
	print(recordingsDF.head())
	recordingsDF.to_csv("recordings.csv")

	x_train = np.stack(recordingsDF['samples'].values, axis=0)
	y_train = keras.utils.to_categorical(y_train)


	x_train = x_train[ : , : , np.newaxis]
	x_train, y_train, x_test, y_test = map(np.array, split_data(x_train, y_train))
 
	model = Sequential()
	model.add(Conv1D(filters=4, 
		kernel_size=10, 
		activation='relu',
		kernel_regularizer = l2(0.05),
		input_shape=x_train.shape[1:]))
	model.add(Conv1D(filters=4, kernel_size=5, activation='relu'))
	model.add(MaxPool1D(pool_size=5))
	model.add(Flatten())
	model.add(Dense(500, activation='relu'))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(20, activation='relu'))
	model.add(Dense(4, activation='softmax'))

	model.compile(loss='mse',
	              optimizer='adam',
	              metrics=['accuracy'])


	model.fit(x_train, y_train,
	          batch_size=15,
	          epochs=20,
	          validation_data=(x_test, y_test))
	score = model.evaluate(x_test, y_test)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
