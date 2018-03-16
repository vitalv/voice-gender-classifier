#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from scipy.io import wavfile

wav_file = '/home/vitalv/voice-gender-classifier/raw/Aaron-20080318-lbb/wav/a0043.wav'

rate, data = wavfile.read(wav_file)
times = np.arange(len(data))/float(rate)

plt.figure(figsize=(12,8))

plt.fill_between(times, data)

plt.xlim(times[0], times[-1])

plt.xlabel('time(s)')

plt.ylabel('amplitude')

plt.show()