import os.path
import wave
import struct
import numpy as np
import pandas as pd
from pylab import *
import matplotlib.pyplot as plt


topdir = "heartbeat-sounds"
exten = '.wav'
file='recordings.csv'

recordingsDF = {}

def readData():
	no=0

	for dirpath, dirnames, files in os.walk(topdir):
		if no==10:
			break
		for name in files:
			if no == 10:
				break
			if name.lower().endswith(exten):
				f = wave.open(os.path.join(dirpath, name))
				
				frames = f.readframes(-1)
				samples = struct.unpack('h'*f.getnframes(), frames)
				
				# time interval for ploting the waveform from t0 = 0 to tn = max seconds
				# framerate = f.getframerate()
				# t = [float(i)/framerate for i in range(len(samples))]
				
				recordings.append((samples,name.split('_')[0]))

				x = np.fft.fft(np.array(samples))
				freq = np.fft.fftfreq(np.array(samples).shape[-1])

				fig = plt.figure()
				fig.suptitle(name.split('_')[0])
				theplot = fig.add_subplot(111)
				theplot = plt.plot(freq, x.real, freq, x.imag)

				plt.show(block=False)

				no = no + 1


def fftData():	
	recordingsDF=pd.read_csv(file)
	# fftRecordings = np.fft.fft(np.array(recordingsDF["samples"]))
	print (len(recordingsDF["samples"]),np.average(recordingsDF["samples"]))

def main ():
	fftData()

main ()