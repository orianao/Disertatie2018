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
from sklearn import tree, ensemble
from sklearn.metrics import accuracy_score,confusion_matrix
import graphviz
# from keras.models import Sequential
# from keras.layers import Conv1D, MaxPool1D, GlobalAvgPool1D, Dropout, BatchNormalization, Dense,Flatten
# from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
# from keras.regularizers import l2
from sklearn.externals import joblib

topdir = "heartbeat-sounds"
exten = '.wav'
Cframerate = 8000
abel = 1

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
		if i > (Cframerate + 31) and i < len(samples) - (Cframerate + 31) and samples[i] > std * 2.8:
			if is_peak(samples[i], samples[i - Cframerate // 5 : i], samples[i + 1 : i + Cframerate // 5]):
				sol.append(i)
	return sol


def get_all_peaks(samples):
	sol = []
	trsh = 8
	std = np.std(samples)
	for i in range(len(samples)):
		if i > Cframerate // trsh and i < len(samples) -Cframerate // trsh and samples[i] > std * 2:
			if is_peak(samples[i], samples[i - Cframerate // trsh : i], samples[i + 1 : i + Cframerate // trsh]):
				sol.append(i)
	return sol


def distances(arr):
	sol = []
	for i in range(len(arr)-1):
		sol.append(arr[i+1]-arr[i])
	return sol


def features(arr):
	sol = []
	peaks_list = get_all_peaks(arr)
	dists = []
	dists = distances (peaks_list)
	# sol.append(np.mean(dists))
	# sol.append(np.std(dists))
	# sol.append(np.mean(peaks_list))
	# sol.append(np.std(peaks_list))
	sol.append(np.min(dists))
	sol.append(np.max(dists))
	# sol.append(np.min(peaks_list))
	# sol.append(np.max(peaks_list))
	sol.append(np.mean(arr))
	sol.append(np.std(arr))
	# sol.append(len(peaks_list))
	return sol


def add_to_recs(samples, namefrom, n):
	global y_train
	peaks_list = get_peaks(samples)
	mapping = {'n' : 0, 'm' : 1, 'e' : 2}
	name = namefrom
	for i in peaks_list:
		if namefrom[0] == 'm':
			name = 'n' + namefrom
		else:
			name = 'm' + namefrom
		if len(get_all_peaks(np.array(samples[i - Cframerate : i + Cframerate ]).real))>1:
			recordings.append((features((np.array(samples[i - Cframerate : i + Cframerate ])).real), mapping[name[0]]))
			#recordings.append(features(np.array(samples[i + 10 - Cframerate : i + 10 + Cframerate ]).real), mapping[name[0]])
			#recordings.append(features(np.array(samples[i - 10 - Cframerate : i - 10 + Cframerate ]).real), mapping[name[0]])
				

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
	aux = [*zip(x_data, y_data)]
	np.random.shuffle(aux)

	x, y = zip(*aux)

	p = int(len(x_data) * 0.9)
	q = int(len(x_data) * 0.95)

	return x[ : p], y[ : p], x[p : q], y[p : q], x[q : ], y[q : ]


if __name__ == '__main__':
	readData()
	recordingsDF = pd.DataFrame(data=recordings, columns = ["samples","label"])
	
	x = np.stack(recordingsDF['samples'].values, axis=0)
	y = keras.utils.to_categorical(recordingsDF['label'].values)

	def nof(a):
		return [a.count(it) for it in range(2)]

	print (nof(list(recordingsDF["label"])))

	clf = tree.DecisionTreeClassifier(min_samples_split=5)
	clf = clf.fit(np.stack(recordingsDF['samples'].values, axis=0)[abel:], recordingsDF[ 'label' ].values[abel:])
	joblib.dump(clf, 'filename1.pkl')
	clf = joblib.load('filename1.pkl') 
	print(clf.feature_importances_, accuracy_score(clf.predict(np.stack( recordingsDF[ 'samples' ].values, axis=0 )[:abel]), recordingsDF['label'].values[:abel]))
	print(confusion_matrix(clf.predict(np.stack( recordingsDF[ 'samples' ].values, axis=0 )[:abel]), recordingsDF['label'].values[:abel]))

	dot_data = tree.export_graphviz(clf, out_file=None)
	graph = graphviz.Source(dot_data) 
	graph.render("recs")

	print("RF")

	clf = ensemble.RandomForestClassifier()
	clf = clf.fit(np.stack(recordingsDF['samples'].values, axis=0)[abel:], recordingsDF[ 'label' ].values[abel:])
	joblib.dump(clf, 'filename2.pkl')
	clf = joblib.load('filename2.pkl') 
	print(clf.feature_importances_, accuracy_score(clf.predict(np.stack( recordingsDF[ 'samples' ].values, axis=0 )[:abel]), recordingsDF['label'].values[:abel]))
	print(confusion_matrix(clf.predict(np.stack( recordingsDF[ 'samples' ].values, axis=0 )[:abel]), recordingsDF['label'].values[:abel]))

	print("AB")

	clf = ensemble.AdaBoostClassifier()
	clf = clf.fit(np.stack(recordingsDF['samples'].values, axis=0)[abel:], recordingsDF[ 'label' ].values[abel:])
	joblib.dump(clf, 'filename3.pkl')
	clf = joblib.load('filename3.pkl') 
	print(clf.feature_importances_, accuracy_score(clf.predict(np.stack( recordingsDF[ 'samples' ].values, axis=0 )[:abel]), recordingsDF['label'].values[:abel]))
	print(confusion_matrix(clf.predict(np.stack( recordingsDF[ 'samples' ].values, axis=0 )[:abel]), recordingsDF['label'].values[:abel]))
