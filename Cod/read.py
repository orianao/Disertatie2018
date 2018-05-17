import os.path
import wave
import struct
import numpy as np
import pandas as pd
from pylab import *
import matplotlib.pyplot as plt


topdir = "heartbeat-sounds"
exten = '.wav'

recordings = []

def readData():
	no=0

	for dirpath, dirnames, files in os.walk(topdir):
		for name in files:
			if name.lower().endswith(exten):
				f = wave.open(os.path.join(dirpath, name))
				
				frames = f.readframes(-1)
				samples = struct.unpack('h'*f.getnframes(), frames)
				
				recordings.append((samples,name.split('_')[0],dirpath.split('_')[1]))

				no=no+1

def main ():
	readData()
	recordingsDF = pd.DataFrame(data=recordings, columns = ["samples","label","type"])
	print(recordingsDF.head())
	recordingsDF.to_csv("recordings.csv")

main ()