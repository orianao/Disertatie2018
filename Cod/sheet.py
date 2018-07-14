import os
import json
import wave
import struct
import numpy as np


topdir = "heartbeat-sounds"
exten = '.wav'
Cframerate = 4000


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


mapping = {'n' : 0, 'a' : 1, 'e' : 2, 'm' : 3}
with open("shit.json", "w") as fout:
	for dirpath, dirnames, files in os.walk("heartbeat-sounds"):
			for name in files:
				if name.lower().endswith(".wav"):
					filepath = os.path.join(dirpath, name)
					with wave.open(filepath) as f:
						frames = f.readframes(-1)
						framerate = f.getframerate()
						nfr = f.getnframes()
						samples = list(struct.unpack('h' * nfr, frames))
						fac_norm = np.linalg.norm(samples)
						samples = list((samples / fac_norm) * 100)

						for i in get_peaks(samples):
							fout.write(json.dumps([samples[i - Cframerate // 2 : i + Cframerate // 2], mapping[name[0]], dirpath.split('_')[1]]))
							fout.write(json.dumps([samples[i + 15 - Cframerate // 2 : i + 15 + Cframerate // 2], mapping[name[0]], dirpath.split('_')[1]]))
							fout.write(json.dumps([samples[i - 15 - Cframerate // 2 : i - 15 + Cframerate // 2], mapping[name[0]], dirpath.split('_')[1]]))
							#fout.write(json.dumps([samples[i + 30 - Cframerate // 2 : i + 30 + Cframerate // 2], mapping[name[0]], dirpath.split('_')[1]]))
							#fout.write(json.dumps([samples[i - 30 - Cframerate // 2 : i - 30 + Cframerate // 2], mapping[name[0]], dirpath.split('_')[1]]))

						


