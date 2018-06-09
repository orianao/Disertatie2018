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


def add_to_recs(samples, namefrom, n, dirpath='a_a'):
	global y_train
	peaks_list = get_peaks(samples)
	mapping = {'n' : 0, 'a' : 1, 'e' : 2, 'm' : 3}
	name = namefrom
	for i in peaks_list:
		if name[0] == 'm':
			name = 'n' + namefrom
		else:
			name = 'a' + namefrom
		recordings.append(((np.array(samples[i - Cframerate // 2 : i + Cframerate // 2])).real, mapping[name[0]], dirpath.split('_')[1]))
		recordings.append(((np.array(samples[i + 20 - Cframerate // 2 : i + 20 + Cframerate // 2])).real, mapping[name[0]], dirpath.split('_')[1]))
		recordings.append(((np.array(samples[i - 20 - Cframerate // 2 : i - 20 + Cframerate // 2])).real, mapping[name[0]], dirpath.split('_')[1]))
		recordings.append(((np.array(samples[i + 30 - Cframerate // 2 : i + 30 + Cframerate // 2])).real, mapping[name[0]], dirpath.split('_')[1]))
		recordings.append(((np.array(samples[i - 30 - Cframerate // 2 : i - 30 + Cframerate // 2])).real, mapping[name[0]], dirpath.split('_')[1]))
		

def readData():
	no=0
	for dirpath, dirnames, files in os.walk(topdir):
		for name in files:
			if name.lower().endswith(exten):
				filepath = os.path.join(dirpath, name)
				f = wave.open(filepath)
				frames = f.readframes(-1)
				framerate = f.getframerate()

				# state = None
				# frames, state = audioop.ratecv(frames, 1, 1, framerate, Cframerate, state)

				samples = list(struct.unpack('h' * f.getnframes(), frames))
				t = [float(i)/framerate for i in range(len(samples))]
				fac_norm = np.linalg.norm(samples)
				samples = (samples / fac_norm) * 100
				# plotit(t,samples,name)

				add_to_recs(samples,name,f.getnframes())

				no = no + 1


def split_data(x_data, y_data):
	aux = [*zip(x_data,y_data)]
	np.random.shuffle(aux)
	x, y = zip(*aux)
	p = int(len(x_data) * 0.8)
	q = int(len(x_data) * 0.85)
	return x[ : p], y[ : p], x[p : q], y[p : q], x[q : ], y[q : ]


if __name__ == '__main__':
	readData()
	# print(np.array_str(np.array(recordings)[:4,0])
	recordingsDF = pd.DataFrame(data=recordings, columns = ["samples","label","type"])
	recordingsDF.to_csv("recordings.csv", sep=';')
	
	x = np.stack(recordingsDF['samples'].values, axis=0)
	y = keras.utils.to_categorical(recordingsDF['label'].values)

	print(recordingsDF['samples'].values[:10])

	x = x[ : , : , np.newaxis]
	x_train, y_train, x_valid, y_valid, x_test, y_test = map(np.array, split_data(x, y))
 
	model = Sequential()
	model.add(Conv1D(filters=4, 
		kernel_size=10, 
		activation='relu',
		kernel_regularizer = l2(0.05),
		input_shape=x_train.shape[1:]))
	model.add(MaxPool1D(pool_size=5))
	model.add(Flatten())
	model.add(Dense(500, activation='relu'))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(20, activation='relu'))
	model.add(Dense(2, activation='softmax'))

	model.compile(optimizer='adam',
	              loss='kullback_leibler_divergence',
	              metrics=['accuracy'])

	model.fit(x_train, y_train,
	          batch_size=30,
	          epochs=10,
	          validation_data=(x_valid, y_valid))
	score = model.evaluate(x_test, y_test)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1]*100)

	# 99,36