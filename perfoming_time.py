from scipy.io import wavfile
import matplotlib.pyplot as plt
import ast
import wave
import math
from numpy import array
import random
import numpy as np
from scipy.fftpack import fft
from pydub import AudioSegment

#TODO: Набор частот передачи вместо одной частоты (передача целого массива данных)
#TODO: Автоматическое определение нулевого урвоня до передачи и введение калибровочных частот для любого типа звука

def convert_to_wav():
    sound = AudioSegment.from_file('/Users/evgeny/PycharmProjects/stego/output/file.mp3')
    sound.export("/Users/evgeny/PycharmProjects/stego/output/result_file.wav", format="wav")
    print("Converting to wav was finished")


def convert_to_mp3():
    sound = AudioSegment.from_file('/Users/evgeny/PycharmProjects/stego/with_mix_250.wav')
    sound.export("/Users/evgeny/PycharmProjects/stego/output/file.mp3", format="mp3", bitrate="128k")
    print("Converting to mp3 was finished")


random.seed(30) #random initializing

fs, data = wavfile.read('./arfa.wav') # load the data
a = data.T[0] # I get the first track
c = fft(a) # calculate fourier transform (complex numbers list)
d = len(c) // 2  # you only need half of the fft list (real signal symmetry)
data_size = len(a)
nu=500 #  Heirz mp3 average seq for human
om = 2* math.pi*nu
dt = 1. / fs
Ts = len(a) // fs
k = 1
bound = 3
dtau = 1 / (nu*k)
A = 400
alpha = 2 / (bound + 1 / bound)
tau = 0
length = Ts * fs
length1 = math.ceil(Ts*k*nu)
min_freq = 500
max_freq = 1000
threshold_ratio = 0.5

process_freq_count = 8
fit_freq_count = 2
delta_freq = (max_freq - min_freq) / (process_freq_count + fit_freq_count - 1)

all_work_freq = [math.ceil(min_freq + i * delta_freq) for i in range(process_freq_count + fit_freq_count)]

cipher_bits = [1, 0, 0, 1, 1, 0, 0, 1]
all_provided_bits = cipher_bits + [0, 1]

spf = wave.open('./arfa.wav','r')


def randomVelocity(bound):
    return (1 / bound) + (bound - 1 / bound) * random.random()



def processTau(l):
    result = []
    tau = 0
    for i in range(l):
        result.append(tau)
        tau += alpha * dtau * randomVelocity(bound)

    return result


transformed_time_array = [[i * dtau for i in range(2 * length1)], processTau(2 * length1)]

# with open('transformed_time.txt', 'w') as f:
#     for item in transformed_time_array:
#         f.write(str(item))
#
#
#
# with open('transformed_time.txt', 'r') as f:
#     transformed_time_array = ast.literal_eval(f.read())


# plt.figure(3)
# plt.title('Transformed time array')
# plt.plot(transformed_time_array[1][0:200])
# plt.show()
#
# x = np.array(transformed_time_array[0][0:200])
# y = np.array(transformed_time_array[1][0:200])
# xvals = np.linspace(0, Ts * 200 / length, 50)
# yinterp = np.interp(xvals, x, y)
# plt.title('Interpolaited y coordinates')
# plt.plot(xvals, yinterp, '-x')
# plt.show()
#
# y = np.array(transformed_time_array[0][0:200])
# x = np.array(transformed_time_array[1][0:200])
# xvals = np.linspace(0, Ts * 200 / length, 50)
# yinterp = np.interp(xvals, x, y)
# plt.title("Reversed interpolaited process of coordinates")
# plt.plot(xvals, yinterp, '-x')
# plt.show()

def interpolaite(args, x_values, y_values):
    m = 0
    l = len(x_values)
    interpolaite_array = [0] * len(args)
    is_exit = False
    for n in range(len(args)):
        if (is_exit):
            break
        while (x_values[m + 1] < args[n]):
            m += 1
            if ((m + 1 >= l - 1)):
                is_exit = True
                break
        else:
            value = y_values[m] + (y_values[m + 1] - y_values[m]) * (args[n] - x_values[m]) / (
                        x_values[m + 1] - x_values[m])
            interpolaite_array[n] = value
    return interpolaite_array


def getSignalWithFreq(freq_array, time, cipher_bits):
    process = 0
    for i in range(len(all_work_freq)):
        process += cipher_bits[i] * math.sin(2 * math.pi * freq_array[i] * time)

    return A * process



x = np.array(transformed_time_array[0][0:2000])
y = np.array(transformed_time_array[1][0:2000])

xvals = np.linspace(0, Ts * 1000 / length, 200)
yinterp0 = interpolaite(xvals, x.tolist(), y.tolist())

plt.title('Interpolaited transformed time array of coordinates')
plt.plot(xvals, yinterp0, '-x')
plt.show()

fs, data = wavfile.read('./arfa.wav') # load the data

sound = data.T[0] # I get the first track

x = transformed_time_array[0]
y = transformed_time_array[1]

xvals_ext = np.linspace(0, 2*Ts, 2*length)
yinterp = interpolaite(xvals_ext, x, y)

#hidden = [A * math.sin(om*yinterp[n]) for n in range(data_size)]
hidden = [getSignalWithFreq(all_work_freq, yinterp[n], all_provided_bits) for n in range(data_size)]
hidden_before_transform = [getSignalWithFreq(all_work_freq, xvals_ext[n], all_provided_bits) for n in range(data_size)]


plt.figure(1)
plt.title('Hidden before transform')
plt.xlabel('k')
plt.ylabel('Amplitude')
plt.plot(abs(array(fft(hidden_before_transform))[0: len(hidden_before_transform) // 44]),'r')
plt.show()

plt.figure(1)
plt.title('Sin after transform of time')
plt.xlabel('k')
plt.ylabel('Amplitude')
plt.plot((array(hidden)[0: 1000]),'r')
plt.show()

mix = (hidden + sound) // 2

output_signal = array(mix).astype("h")
wavfile.write('with_mix_250.wav', fs, output_signal)


c = fft(mix) # calculate fourier transform (complex numbers list)

d = len(c) // 2  # you only need half of the fft list (real signal symmetry)


plt.figure(2)
plt.title('Fourier transform on sound with sin')
plt.xlabel('k')
plt.ylabel('Amplitude')
plt.plot(abs(c[nu*Ts-100:nu*Ts+100]),'r')
plt.show()

convert_to_mp3()
convert_to_wav()

fs, sound_after_converting_wav_mp3 = wavfile.read('./output/result_file.wav') # load the data

input_sound_length = len(sound_after_converting_wav_mp3)
graph_length = input_sound_length // 22

y_rev = interpolaite(xvals_ext, y, x)

xvals = np.linspace(0, Ts, length)
mix_rev_after_convering = interpolaite(y_rev, xvals, sound_after_converting_wav_mp3)
mix_rev = interpolaite(y_rev, xvals, mix)
sound_rev = interpolaite(y_rev, xvals,  sound)
hidden_rev =  interpolaite(y_rev, xvals, hidden)

length_mix_rev = len(mix_rev_after_convering)

mix_rev_converted_fft = fft(mix_rev_after_convering)
mix_rev_fft = fft(mix_rev)
sound_rev_fft = fft(sound_rev)
hidden_rev_fft = fft(hidden_rev)

mix_fft = fft(mix)
sound_fft = fft(sound)
hidden_fft = fft(hidden)

true_level = abs(mix_rev_converted_fft[2*Ts*max_freq])

for i in range(len(all_provided_bits)):
    print(abs(mix_rev_converted_fft[2 * Ts * all_work_freq[i]]))

decoded_array = [abs(mix_rev_converted_fft[2 * Ts * all_work_freq[i]]) > true_level * threshold_ratio for i in range(len(all_provided_bits))]

print(true_level)
print(decoded_array)

plt.subplot(3,3,1)
plt.title('Mix')
plt.xlabel('k')
plt.ylabel('Amplitude')
#plt.plot(abs(mix_rev_fft[(length_mix_rev // length)*nu*Ts-100:(length_mix_rev // length)*nu*Ts+100]),'r')
plt.plot(abs(mix_fft[0:  graph_length]),'r')

plt.subplot(3,3,2)
plt.title('Sound')
plt.xlabel('k')
plt.ylabel('Amplitude')
#plt.plot(abs(mix_rev_fft[(length_mix_rev // length)*nu*Ts-100:(length_mix_rev // length)*nu*Ts+100]),'r')
plt.plot(abs(sound_fft[0:  graph_length]),'r')

plt.subplot(3,3,3)
plt.title('Hidden')
plt.xlabel('k')
plt.ylabel('Amplitude')
#plt.plot(abs(mix_rev_fft[(length_mix_rev // length)*nu*Ts-100:(length_mix_rev // length)*nu*Ts+100]),'r')
plt.plot(abs(hidden_fft[0:  graph_length]),'r')


plt.subplot(3,3,4)
plt.title('Fourier transform on reversed time sound with sin')
plt.xlabel('k')
plt.ylabel('Amplitude')
#plt.plot(abs(mix_rev_fft[(length_mix_rev // length)*nu*Ts-100:(length_mix_rev // length)*nu*Ts+100]),'r')
plt.plot(abs(mix_rev_converted_fft[2*Ts*nu - 100:  2*Ts*nu  + 100]),'r')
#plt.plot(abs(mix_rev_converted_fft[0:  graph_length]),'r')


plt.subplot(3,3,5)
plt.title("Mix rev")
plt.xlabel('k')
plt.ylabel('Amplitude')
#plt.plot(abs(mix_rev_fft[(length_mix_rev // length)*nu*Ts-100:(length_mix_rev // length)*nu*Ts+100]),'r')
plt.plot(abs(mix_rev_fft[2*Ts*nu - 100:  2*Ts*nu  + 100]),'r')
#plt.plot(abs(mix_rev_fft[0:  graph_length]),'r')

plt.subplot(3,3,6)
plt.title("Clear signal rev")
plt.xlabel('k')
plt.ylabel('Amplitude')
plt.plot(abs(sound_rev_fft[(length_mix_rev // length)*nu*Ts-100:(length_mix_rev // length)*nu*Ts+100]),'r')
#plt.plot(abs(sound_rev_fft[0:  graph_length]),'r')

plt.subplot(3,3,7)
plt.title("Hidden rev")
plt.xlabel('k')
plt.ylabel('Amplitude')
#plt.plot(abs(mix_rev_fft[(length_mix_rev // length)*nu*Ts-100:(length_mix_rev // length)*nu*Ts+100]),'r')
#plt.plot(abs(hidden_rev_fft[2*Ts*nu - 100:  2*Ts*nu  + 100]),'r')
plt.plot(abs(hidden_rev_fft[0:  graph_length]),'r')

plt.show()