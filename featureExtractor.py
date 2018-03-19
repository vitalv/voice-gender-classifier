#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import re
import scipy.stats as stats
from scipy.io import wavfile
import numpy as np
import os


pattern_date = re.compile('[0-9]{8}')
raw_folder = './raw'


def get_metadata(readme_file):

	#define variables in case startswith does not work:
	gender, age_range, pronunciation = '', '', '' 
	for line in open(readme_file):
		if line.startswith("Gender:"): 
			gender = line.split(":")[1].strip()
		elif line.startswith("Age Range:"): 
			age_range = line.split(":")[1].strip()
		elif line.startswith("Pronunciation dialect:"): 
			pronunciation = line.split(":")[1].strip()
	return gender, age_range, pronunciation



def get_features(wav_data):

	print "\nExtracting features "
	nobs, minmax, mean, variance, skew, kurtosis =  stats.describe(wav_data)
	median   = np.median(wav_data)
	mode     = stats.mode(wav_data).mode[0]
	std      = np.std(wav_data)
	low,peak = minmax
	q75,q25  = np.percentile(wav_data, [75 ,25])
	iqr      = q75 - q25
	return nobs, mean, skew, kurtosis, median, mode, std, low, peak, q25, q75, iqr



def get_date(sample_name):

	try:
		date = pattern_date.search(sample_name).group()
	except AttributeError:
		date = '20000000'
	return date



def get_user_name(sample_name):

	return re.compile("[-_]").split(sample_name)[0]



def get_wav_data(sample_name, sample_wav_folder):

	sample_wav_data = []
	print "\nReading wav files for sample %s: "% sample_name
	for wav_file in os.listdir(sample_wav_folder):
		if wav_file.endswith('.wav'):
			rate, data = wavfile.read(os.path.join(sample_wav_folder, wav_file))
			sample_wav_data.append(data)
	data = np.concatenate(sample_wav_data)
	return data, rate



def get_frequencies(data, rate):

	N = len(data)
	duration = len(data) / float(rate) #duration of the sample in seconds
	df = 1/duration #fundamental frequency in Hz
	
	#freqs = np.fft.fftfreq(N)

	freqs = np.fft.fftfreq(N)*N*df
	#freqs = freqs[freqs>0]
	#freqs = freqs[freqs<280]

	#freqs = np.arange(0, (len(data)/2), 1.0) * float(rate)/len(data)
	#freqs = freqs[freqs>80]
	#freqs = freqs[freqs<280]


    # Find the peak in the coefficients
    #idx = np.argmax(np.abs(np.fft.fft(data)))
    #freq = freqs[idx]
    #peak_freq = abs(freq * rate)

	return freqs





def main():

	samples = [d for d in os.listdir(raw_folder) if os.path.isdir(os.path.join(raw_folder, d))]
	n_samples = len(samples)

	columns=['nobs', 'mean', 'skew', 'kurtosis', 
	'median', 'mode', 'std', 'low', 
	'peak', 'q25', 'q75', 'iqr', 
	'user_name', 'sample_date', 'age_range', 
	'pronunciation', 'gender' ]

	myData = pd.DataFrame(columns=columns, index=range(n_samples))

	for i in range(n_samples):

		sample = sorted(samples)[i]
		sample_folder = os.path.join(raw_folder, sample)
		sample_wav_folder = os.path.join(sample_folder, 'wav')
		readme_file = os.path.join(sample_folder, 'etc', 'README')

		date = get_date(sample)
		user_name = get_user_name(sample)
		if os.path.isfile(readme_file):
			gender, age_range, pronunciation = get_metadata(readme_file)


		if os.path.isdir(sample_wav_folder): #some of the samples don't contain a wav folder (Ex: 'LunaTick-20080329-vf1')

			data, rate = get_wav_data(sample, sample_wav_folder)

			freqs = get_frequencies(data, rate)

			nobs, mean, skew, kurtosis, median, mode, std, low, peak, q25, q75, iqr = get_features(freqs)
			sample_dict = {'nobs':nobs, 'mean':mean, 'skew':skew, 'kurtosis':kurtosis,
							'median':median, 'mode':mode, 'std':std, 'low': low,
							'peak':peak, 'q25':q25, 'q75':q75, 'iqr':iqr, 
							'user_name':user_name, 'sample_date':date, 
							'age_range':age_range, 'pronunciation':pronunciation,
							'gender':gender}
			print "\nappending sample %s : %s"%(sample, sample_dict)			
			myData.loc[i] = pd.Series(sample_dict)

	myData.to_csv('myData.csv')



if __name__ == '__main__':
	main()






'''
The next thing to look at is the frequency of the audio. 
In order to do this you need to decompose the single audio wave into audio waves at different frequencies. 
This can be done using a Fourier transform. 
The Fourier transform effectively iterates through a frequency for as many frequencies as there are records (N) in the dataset,
and determines the Amplitude of that frequency. The frequency for record (fk) can be calculated using the sampling rate (fs)
The following code performs the Fourier transformation sound data and plots it. 
The maths produces a symetrical result, with one real data solution, and an imaginary data solution
'''


fourier = np.fft.fft(data)

#We only need the real data solution, so we can grab the first half, 
#then calculate the frequency and plot the frequency against a scaled amplitude.

n = len(data)
fourier = fourier[0:(n/2)]

# scale by the number of points so that the magnitude does not depend on the length
fourier = fourier / float(n)

#calculate the frequency at each point in Hz
freqArray = np.arange(0, (n/2), 1.0) * (rate*1.0/n);



'''
plt.plot(fourier, color='#ff7f00')
plt.xlabel('k')
plt.ylabel('Amplitude')
'''
