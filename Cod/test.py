import timeit
start_time = timeit.default_timer()
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
from resizeimage import resizeimage
import os.path
import wave
import struct
import numpy as np
import pandas as pd
from pylab import *
import keras
from sklearn.metrics import classification_report, confusion_matrix


topdir = "test"
exten = '.wav'
Cframerate = 8000

recordings = []


def is_peak (x, left, right):
	return np.amax(left) < x and np.amax(right) < x


def get_peaks(samples):
	sol = []
	std = np.std(samples)
	for i in range(len(samples)):
		if i > (Cframerate + 31) and i < len(samples) - (Cframerate + 31) and samples[i] > std * 2.8:
			if is_peak(samples[i], samples[i - Cframerate // 5 : i], samples[i + 1 : i + Cframerate // 5]):
				sol.append(i)
	return sol


def add_to_recs(samples, namefrom, n):
	global y_train, cnt
	peaks_list = get_peaks(samples)
	mapping = {'n' : 0, 'm' : 1, 'e' : 2}
	name = namefrom
	for i in peaks_list:
		if namefrom[0] == 'm':
			name = 'n' + namefrom
		else:
			name = 'm' + namefrom
		recordings.append(((np.array(samples[i - Cframerate : i + Cframerate])).real, mapping[name[0]]))
		#recordings.append(((np.array(samples[i + 5 - Cframerate : i + 5 + Cframerate ])).real, mapping[name[0]], dirpath.split('_')[1]))
		#recordings.append(((np.array(samples[i - 5 - Cframerate : i - 5 + Cframerate ])).real, mapping[name[0]], dirpath.split('_')[1]))
		

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
	return x, y


if __name__ == '__main__':
	readData()
	# print(np.array_str(np.array(recordings)[:4,0])
	recordingsDF = pd.DataFrame(data=recordings, columns = ["samples","label"])

	def nof(a):
		return [a.count(it) for it in range(4)]
	
	x = np.stack(recordingsDF['samples'].values, axis=0)
	y = keras.utils.to_categorical(recordingsDF['label'].values)

	x = x[ : , : , np.newaxis]
	x_test, y_test = map(np.array, split_data(x, y))

	#model = load_model('ModelCNNA0.9996231493943473.h5')
	#model = load_model('ModelCNNE0.8975265002924646.h5')
	model = load_model('ModelCNNM0.9774520856820744.h5')
	#model = load_model('ModelCNNN0.9774647887323944.h5')
	#model = load_model('ModelCNNAll0.9746478873239437.h5')
	
	score = model.evaluate(x_test, y_test)

	print (nof(list(recordingsDF["label"])))

	print('Test loss:', score[0])
	print('Test accuracy:', score[1] * 100)

	print(confusion_matrix(np.argmax(model.predict(x_test), axis = 1),np.argmax(y_test, axis = 1)))

	model.summary()

	elapsed = timeit.default_timer() - start_time
	print ("Time " + str(elapsed))