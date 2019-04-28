import matplotlib.pyplot as plt
import wave
import numpy as np

from scipy.fftpack import fft
from scipy.io import wavfile



fs, data = wavfile.read('./arfa.wav') # load the data
a = data.T[0] # I get the first track
c = fft(a) # calculate fourier transform (complex numbers list)
d = len(c) // 2  # you only need half of the fft list (real signal symmetry)

plt.figure(1)
plt.xlabel('k')
plt.ylabel('Amplitude')
plt.plot(abs(c[:(d-1)]),'r')
plt.show()

spf = wave.open('./arfa.wav','r')

#Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal_short = np.frombuffer(signal, 'int')
signal_long = np.frombuffer(signal, 'short')


#If Stereo
if spf.getnchannels() == 2:
    print('Just mono files')

plt.figure(2)
plt.title('Signal Wave...')
plt.plot(signal_short)
plt.show()