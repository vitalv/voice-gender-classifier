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



def get_features(frequencies):

  print "\nExtracting features "
  nobs, minmax, mean, variance, skew, kurtosis =  stats.describe(frequencies)
  median   = np.median(frequencies)
  mode     = stats.mode(frequencies).mode[0]
  std      = np.std(frequencies)
  low,peak = minmax
  q75,q25  = np.percentile(frequencies, [75 ,25])
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




def get_frequencies(sample_wav_folder):

  #extract list of dominant frequencies in sliding windows of duration defined by 'step' for each of the 10 wav files and return an array

  frequencies_lol = [] #lol: list of lists
  for wav_file in os.listdir(sample_wav_folder):
    rate, data = wavfile.read(os.path.join(sample_wav_folder, wav_file))

    #get dominating frequencies in sliding windows of 20ms
    step = rate/2 #8000 sampling points every 0.5 sec 
    window_frequencies = []

    for i in range(0,len(data),step):
      ft = np.fft.fft(data[i:i+step])
      freqs = np.fft.fftfreq(len(ft))
      imax = np.argmax(np.abs(ft))
      freq = freqs[imax]
      freq_in_hz = abs(freq *rate)
      window_frequencies.append(freq_in_hz)
      filtered_frequencies = [f for f in window_frequencies if 10<f<300 ]#and not 45<f<55] #50 Hz is noise
      frequencies_lol.append(filtered_frequencies)

  frequencies = [item for sublist in frequencies_lol for item in sublist]

  return frequencies




def main():

  samples = [d for d in os.listdir(raw_folder) if os.path.isdir(os.path.join(raw_folder, d))]
  n_samples = len(samples)

  columns=['nobs', 'mean', 'skew', 'kurtosis', 
  'median', 'mode', 'std', 'low', 
  'peak', 'q25', 'q75', 'iqr', 
  'user_name', 'sample_date', 'age_range', 
  'pronunciation', 'gender' ]

  myData = pd.DataFrame(columns=columns)#, index=range(n_samples))

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

      frequencies = get_frequencies(sample_wav_folder)

      if len(frequencies) > 10: 
        #for some of the files (ex: Aaron-20130527-giy) 
        #I only recover frequencies of 0.0 (even if I don't subspit in chunks) which is not integrated into my lol and frequencies is empty

        nobs, mean, skew, kurtosis, median, mode, std, low, peak, q25, q75, iqr = get_features(frequencies)
        sample_dict = {'nobs':nobs, 'mean':mean, 'skew':skew, 'kurtosis':kurtosis,
                       'median':median, 'mode':mode, 'std':std, 'low': low,
                       'peak':peak, 'q25':q25, 'q75':q75, 'iqr':iqr, 
                       'user_name':user_name, 'sample_date':date, 
                       'age_range':age_range, 'pronunciation':pronunciation,
                       'gender':gender}
        print "\nappending sample %s : %s"%(sample, sample_dict)
        myData.loc[i] = pd.Series(sample_dict)

  myData.to_csv('myData_filtered.csv')



if __name__ == '__main__':
  main()





