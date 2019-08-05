import os
import sys
import glob
import numpy as np
import librosa
from scipy.io import wavfile

def main(pathAudio):
	for dirs in os.listdir(pathAudio):
		auds = glob.glob(os.path.join(pathAudio, dirs + '/audio'))
		for aud in auds:
			wavs = glob.glob(os.path.join(pathAudio, dirs, aud + '/*.wav'))
			for wav in wavs:
				y, sr = librosa.load(wav, sr = 16000, mono=True)
				y = y * 32767 / max(0.01, np.max(np.abs(y)))
				wavfile.write(wav, sr, y.astype(np.int16))

if __name__ == '__main__':
	main('path/to/folder')
