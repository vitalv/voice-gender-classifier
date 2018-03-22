#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from scipy.io import wavfile
import seaborn as sb 
sb.set_style("whitegrid", {'axes.grid' : False})

#wav_file = '/home/vitalv/voice-gender-classifier/raw/Aaron-20080318-lbb/wav/a0043.wav'
#wav_file = '/home/vitalv/voice-gender-classifier/raw/Aaron-20130527-giy/wav/b0350.wav'
#wav_file = '/home/vitalv/voice-gender-classifier/raw/chris-20090325-esw/wav/a0060.wav'#Noise at 50Hz #check plot_frequency
#wav_file = '/home/vitalv/voice-gender-classifier/raw/zeroschism-20160710/wav/cc-01.wav' #Noise at 60Hz
wav_file = '/home/vitalv/voice-gender-classifier/raw/anonymous-20100621-cdr/wav/a0166.wav'

rate, data = wavfile.read(wav_file)



#plot amplitude (or loudness) over time ############
time = np.arange(0, float(data.shape[0]), 1) / rate
plt.figure(1)
plt.subplot(111)
plt.plot(time, data, linewidth=0.1, alpha=0.9, color='teal') #
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()




#plot also frequency ##########################


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
x = freqArray[freqArray<300] #human voice range
y = 10*np.log10(fourier)[0:len(x)]
plt.plot(x, y, color='#ff7f00', linewidth=0.15)
plt.xlabel('Frequency (Hz)')
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