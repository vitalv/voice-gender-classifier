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
	return data




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

		data = get_wav_data(sample, sample_wav_folder)

		nobs, mean, skew, kurtosis, median, mode, std, low, peak, q25, q75, iqr = get_features(data)

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

#'BlindPilot-20100610-hnf' 12.00
#Catbbells-20110928-mxl 12.20