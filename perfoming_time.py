from scipy.io import wavfile
import matplotlib.pyplot as plt
import ast
import wave
import math
from numpy import array
import random
import numpy as np
from scipy.fftpack import fft

random.seed(30)

fs, data = wavfile.read('./arfa.wav') # load the data

a = data.T[0] # I get the first track

print(len(a))

c = fft(a) # calculate fourier transform (complex numbers list)

d = len(c) // 2  # you only need half of the fft list (real signal symmetry)

data_size = len(a)

nu=500 #  герцы  mp3 средняя частота для человеского уха
om = 2* math.pi*nu
dt = 1. / fs

Ts = len(a) // fs

spf = wave.open('./arfa.wav','r')

fs, data = wavfile.read('./arfa.wav') # load the data

a = data.T[0] # I get the first track


k = 1
bound = 3
dtau = 1 / (nu*k)



def randomVelocity(bound):
    return (1 / bound) + (bound - 1 / bound) * random.random()


alpha = 2 / (bound + 1 / bound)
tau = 0
length = Ts * fs
length1 = Ts*k*nu

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
    l = len(y_values)
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


x = np.array(transformed_time_array[0][0:2000])
y = np.array(transformed_time_array[1][0:2000])

xvals = np.linspace(0, Ts * 1000 / length, 200)
yinterp = interpolaite(xvals, x.tolist(), y.tolist())

plt.title('Interpolaited transformed time array of coordinates')
plt.plot(xvals, yinterp, '-x')
plt.show()

data_size = len(a)

nu=500 #  герцы  mp3 средняя частота для человеского уха
om = 2* math.pi*nu
dt = 1. / fs

NU = 25
OM = 2* math.pi*NU
m=0 #глубина модуляции - 0
Ts = len(a) // fs

length = Ts*fs

A = 2000

fs, data = wavfile.read('./arfa.wav') # load the data

sound = data.T[0] # I get the first track

x = transformed_time_array[0]
y = transformed_time_array[1]

xvals_ext = np.linspace(0, 2*Ts, 2*length)
yinterp = interpolaite(xvals_ext, x, y)

hidden = [A * math.sin(om*yinterp[n]) for n in range(data_size)]

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


fs, sound_after_convering_wav_mp3 = wavfile.read('./test_conver1.wav') # load the data

print(len(sound_after_convering_wav_mp3))

y_rev = interpolaite(xvals_ext, y, x)


xvals = np.linspace(0, Ts, length)
mix_rev_after_convering = interpolaite(y_rev, xvals, sound_after_convering_wav_mp3)
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

plt.figure(3)
plt.title('Mix')
plt.xlabel('k')
plt.ylabel('Amplitude')
#plt.plot(abs(mix_rev_fft[(length_mix_rev // length)*nu*Ts-100:(length_mix_rev // length)*nu*Ts+100]),'r')
plt.plot(abs(mix_fft[0:  20000]),'r')

plt.figure(4)
plt.title('Sound')
plt.xlabel('k')
plt.ylabel('Amplitude')
#plt.plot(abs(mix_rev_fft[(length_mix_rev // length)*nu*Ts-100:(length_mix_rev // length)*nu*Ts+100]),'r')
plt.plot(abs(sound_fft[0:  20000]),'r')

plt.figure(5)
plt.title('Hidden')
plt.xlabel('k')
plt.ylabel('Amplitude')
#plt.plot(abs(mix_rev_fft[(length_mix_rev // length)*nu*Ts-100:(length_mix_rev // length)*nu*Ts+100]),'r')
plt.plot(abs(hidden_fft[0:  20000]),'r')


plt.figure(6)
plt.title('Fourier transform on reversed time sound with sin')
plt.xlabel('k')
plt.ylabel('Amplitude')
#plt.plot(abs(mix_rev_fft[(length_mix_rev // length)*nu*Ts-100:(length_mix_rev // length)*nu*Ts+100]),'r')
plt.plot(abs(mix_rev_converted_fft[0:  20000]),'r')


plt.figure(7)
plt.title("Mix rev")
plt.xlabel('k')
plt.ylabel('Amplitude')
#plt.plot(abs(mix_rev_fft[(length_mix_rev // length)*nu*Ts-100:(length_mix_rev // length)*nu*Ts+100]),'r')
plt.plot(abs(mix_rev_fft[0:  20000]),'r')

plt.figure(8)
plt.title("Clear signal rev")
plt.xlabel('k')
plt.ylabel('Amplitude')
#plt.plot(abs(mix_rev_fft[(length_mix_rev // length)*nu*Ts-100:(length_mix_rev // length)*nu*Ts+100]),'r')
plt.plot(abs(sound_rev_fft[0:  20000]),'r')

plt.figure(9)
plt.title("Hidden rev")
plt.xlabel('k')
plt.ylabel('Amplitude')
#plt.plot(abs(mix_rev_fft[(length_mix_rev // length)*nu*Ts-100:(length_mix_rev // length)*nu*Ts+100]),'r')
plt.plot(abs(hidden_rev_fft[0:  20000]),'r')

plt.show()





