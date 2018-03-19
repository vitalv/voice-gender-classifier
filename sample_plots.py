#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from scipy.io import wavfile

wav_file = '/home/vitalv/voice-gender-classifier/raw/Aaron-20080318-lbb/wav/a0043.wav'

rate, data = wavfile.read(wav_file)

time = np.arange(0, float(data.shape[0]), 1) / rate

#plot amplitude (or loudness) over time
plt.figure(1)
plt.subplot(111)
plt.plot(time, data, linewidth=0.1, alpha=0.9, color='teal') #
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()




#plot also frequency ##########################

def plot_frequency(data):
	fourier = np.fft.fft(data)
	'''
	plt.plot(fourier, color='#ff7f00')
	plt.xlabel('k')
	plt.ylabel('Amplitude')
	'''
	n = len(data)
	fourier = fourier[0:(n/2)]
	# scale by the number of points so that the magnitude does not depend on the length
	fourier = fourier / float(n)
	#calculate the frequency at each point in Hz
	freqArray = np.arange(0, (n/2), 1.0) * (rate*1.0/n);
	plt.plot(freqArray/1000, 10*np.log10(fourier), color='#ff7f00', linewidth=0.15)
	plt.xlabel('Frequency (kHz)')
	plt.ylabel('Amplitude (dB)')
	plt.show()



# And Heatmap ######################################
# See http://myinspirationinformation.com/uncategorized/audio-signals-in-python/

plt.figure(1, figsize=(8,5))
plt.subplot(111)
Pxx, freqs, bins, im = plt.specgram(data, Fs=rate, NFFT=1024, cmap=plt.get_cmap('autumn_r'))
cbar=plt.colorbar(im)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
cbar.set_label('Amplitude dB')
plt.show()