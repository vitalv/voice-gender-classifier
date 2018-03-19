import wave
import struct
import numpy as np

def isSilence(windowPosition):
    sumVal = sum( [ x*x for x in sound[windowPosition:windowPosition+windowSize+1] ] )
    avg = sumVal/(windowSize)
    if avg <= 0.0001:
        return True
    else:
        return False

wav_file = '/home/vital/voice-gender-classifier/raw/Aaron-20130527-giy/wav/b0350.wav'

#read from wav file
sound_file = wave.open(wav_file, 'r')
file_length = sound_file.getnframes()
data = sound_file.readframes(file_length)
sound_file.close()
#data = struct.unpack("<h", data)
data = struct.unpack('{n}h'.format(n=file_length), data)
sound = np.array(data)
#sound is now a list of values

#detect silence and notes
i=0
windowSize = 2205
windowPosition = 0
#listOfLists = []
listOfLists.append([])
maxVal = len(sound) - windowSize
while True:
    if windowPosition >= maxVal:
        break
    if not isSilence(windowPosition):
        while not isSilence(windowPosition):
            listOfLists[i].append(sound[windowPosition:windowPosition+ windowSize+1])
            windowPosition += windowSize
        listOfLists.append([]) #empty list
        i += 1
    windowPosition += windowSize

frequencies = []
#Calculating the frequency of each detected note by using DFT
for signal in listOfLists:
    if not signal:
        break
    w = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(w))
    l = len(signal)

    #imax = index of first peak in w
    imax = np.argmax(np.abs(w))
    fs = freqs[imax]

    freq = imax*fs/l
    frequencies.append(freq)

print frequencies