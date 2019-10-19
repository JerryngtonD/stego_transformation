from scipy.io import wavfile
import matplotlib.pyplot as plt
import wave
import math
import cmath
from numpy import array
import random
import numpy as np
from scipy.fftpack import fft
from pydub import AudioSegment

# TODO: Набор частот передачи вместо одной частоты (передача целого массива данных)
# TODO: Автоматическое определение нулевого урвоня до передачи и введение калибровочных частот для любого типа звука

'''
Конвертор для преобразования mp3-сигнала в wav
'''


def convert_to_wav():
    sound = AudioSegment.from_file('file.mp3')
    sound.export("result_file.wav", format="wav")
    print("Converting to wav was finished")


'''
Конвертор для преобразования wav-сигнала в mp3
'''


def convert_to_mp3():
    sound = AudioSegment.from_file('with_mix_250.wav')
    sound.export("file.mp3", format="mp3", bitrate="128k")
    print("Converting to mp3 was finished")


'''
Необходимо задать инициализацию генератора случайных чисел, чтобы при раскодировке
и дальнейшем дебаге не сохранять массив случайных значений в processTau
'''
random.seed(30)  # random initializing

'''
Загрузка файла: fs - частота дискретизации
                data - информационные байты ауди-сигнала
'''
fs, data = wavfile.read('./arfa.wav')

'''
Аудио файл может иметь несколько дорожек или каналов,
в нашем случае одна дорожка
'''
a = data.T[0]

'''
Быстрое преобразование Фурье для определения спектра частот
Получаем лист комплексных значений
'''
c = fft(a)

'''
Для Фурье-образа действительнозначной функции (т.е. любого “реального” сигнала) 
амплитудный спектр всегда является четной функцией, нам необходима лишь половина для работы
'''
d = len(c) // 2

'''
Исходный размер файла
'''
data_size = len(a)

'''
Средняя частота сигнала слышимая человеческоим ухом
'''
nu = 500

'''
Значение круговой частоты
'''
om = 2 * math.pi * nu

'''
Величина шага дискретизации
'''
dt = 1. / fs

'''
Длительность звуковой дорожки исходного файла
'''
Ts = len(a) // fs

'''
Коэффициент модуляции отклонения
'''
k = 1

'''
Граничный коэффициент - модуль отклонения (границы скорости течения времени)
'''
bound = 3

'''
Частота хода
'''
dtau = 1 / (nu * k)

'''
Коэффициент для расчета параметра tau
'''
alpha = 2 / (bound + 1 / bound)


'''
Длительность сигнала
'''
length = Ts * fs

'''
Длина единиц сигнала с учетом интерполяции по частоте дискретизации
'''
length1 = math.ceil(Ts * k * nu)

'''
Нижняя граница диапазона служебных частот для кодирования
'''
min_freq = 500

'''
Верхняя граница диапазона служебных частот для кодирования
'''
max_freq = 1000


'''
Коэффициент для рапознавания правильности амплитуды сигнала
'''
threshold_ratio = 0.5

'''
Фактор-коэффициент для уровня 0
'''
false_factor = 1.1

'''
Фактор-коэффициент для уровня 1
'''
true_factor = 1.3

'''
Будем передавать 8 бит - для них 8 рабочих частот
'''
process_freq_count = 40

'''
2 частоты на алфавит передачи - в данном случае бинарный [0, 1]
'''
fit_freq_count = 2

'''
Стегосообщение
'''
cipher_bits = [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]

'''
Все биты которые кодируем в сообщении
'''
all_provided_bits = cipher_bits + [0, 1]

'''
Шараметр дельты по частоте - шаг, с которым будем делить полосу частот [500; 1000]
'''
delta_freq = (max_freq - min_freq) / (process_freq_count + fit_freq_count - 1)

'''
Получаем все служебные частоты для передачи
'''
all_work_freq = [math.ceil(min_freq + i * delta_freq) for i in range(process_freq_count + fit_freq_count)]

spf = wave.open('./arfa.wav', 'r')


'''
Функция генерации рандомного скачка в границах для скорорости течения время 
'''
def get_random_velocity():
    return (1 / bound) + (bound - 1 / bound) * random.random()


'''
Параметр дельты по частоте - шаг, с которым будем делить полосу частот [500; 1000]
'''
def processTau(l):
    result = []
    tau = 0
    for i in range(l):
        result.append(tau)
        tau += alpha * dtau * get_random_velocity()

    return result


# s- value of fft component of signal, a - value of component of hidden signal, r - module of wished level of mix signal

'''
Функция получения добавочной амплитуды для достижения уровня 1
'''
def get_amplitude_value(s, a, r):
    if r == 0:
        return 0
    phi = cmath.phase(s)
    psi = cmath.phase(a)

    return abs(r / a) * math.cos(math.asin(abs(s / r)) * math.sin(phi - psi)) - abs(s / a) * math.cos(phi - psi)


'''
Функция интерполяции диапазона значений 
'''
def interpolaite(args, x_values, y_values):
    m = 0
    l = len(x_values)
    interpolaite_array = [0] * len(args)
    is_exit = False
    for n in range(len(args)):
        if is_exit:
            break
        while (x_values[m + 1] < args[n]):
            m += 1
            if m + 1 >= l - 1:
                is_exit = True
                break
        else:
            value = y_values[m] + (y_values[m + 1] - y_values[m]) * (args[n] - x_values[m]) / (
                    x_values[m + 1] - x_values[m])
            interpolaite_array[n] = value
    return interpolaite_array


'''
Функция получения массива сигналов для встраивания
'''
def get_signal_with_freq(freq_array, time, amplitudes_array):
    process = 0
    for i in range(len(all_work_freq)):
        process += amplitudes_array[i] * math.sin(2 * math.pi * freq_array[i] * time)

    return process

'''
Получаем массив преобразованного времени
'''
transformed_time_array = [[i * dtau for i in range(2 * length1)], processTau(2 * length1)]

'''
Преобразовании координаты времени и значений от времени
'''
x = np.array(transformed_time_array[0][0:2000])
y = np.array(transformed_time_array[1][0:2000])

'''
Для построения графиков достаточно меньшей области интервала
'''
xvals_short = np.linspace(0, Ts * 1000 / length, 200)
xvals = np.linspace(0, Ts, length)
yinterp0 = interpolaite(xvals_short, x.tolist(), y.tolist())

plt.title('Interpolaited transformed time array of coordinates')
plt.plot(xvals_short, yinterp0, '-x')
plt.show()

fs, data = wavfile.read('./arfa.wav')

sound = data.T[0]
x = transformed_time_array[0]
y = transformed_time_array[1]

xvals_ext = np.linspace(0, 2 * Ts, 2 * length)
yinterp = interpolaite(xvals_ext, x, y)
y_rev = interpolaite(xvals_ext, y, x)

'''
Звук с координатами в преобразованном времени
'''
sound_rev = interpolaite(y_rev, xvals, sound)

'''
Фурье преобразование от звука в преобразованном времени
'''
sound_rev_fft = fft(sound_rev)

'''

Получаем значения амплитуд сигналов, которые уже имеются в спектре, для того чтобы узнать сколько необходимо
до уверенного сигнала
'''
signal_amplitudes_on_freq_array = [sound_rev_fft[2 * Ts * all_work_freq[i]] for i in range(len(all_provided_bits))]

'''
Максимальная амплитуда из всех сигналов со служебными частотами, ее возьмем за эталон
'''
max_amplitude_on_signal = max(np.absolute(np.array(signal_amplitudes_on_freq_array)))
false_value = max_amplitude_on_signal * false_factor
true_value = false_value * true_factor

'''
Смотрим максимальные уровни шумов на всех служебных частотах
'''
test_all_true_values_array = [1] * len(all_provided_bits)
test_true_array_signal = [get_signal_with_freq(all_work_freq, yinterp[n], test_all_true_values_array) for n in
                          range(data_size)]
test_true_array_signal_rev = interpolaite(y_rev, xvals, test_true_array_signal)
test_true_array_signal_rev_fft = fft(test_true_array_signal_rev)

test_true_amplitudes_on_freq_array = [test_true_array_signal_rev_fft[2 * Ts * all_work_freq[i]] for i in
                                      range(len(all_provided_bits))]


r_amplitudes_array = [true_value if (cipher_bits[i] > 0) else 0 for i in range(len(cipher_bits))]
r_amplitudes_service_bits = [false_value, true_value]
r_amplitudes_array += r_amplitudes_service_bits


'''
Вычисляем сколько нужно добавить амплитуд к сигналам, чтобы в обратном преобразовании к обычному времени видеть пики
выше эталонной амплитуды для уверенного приема
'''
amplitudes_array_for_adding = [
    get_amplitude_value(signal_amplitudes_on_freq_array[i], test_true_amplitudes_on_freq_array[i],
                        r_amplitudes_array[i]) for i in range(len(signal_amplitudes_on_freq_array))]

print("amplitudes for adding")
print(amplitudes_array_for_adding)

print("test")
print([abs(signal_amplitudes_on_freq_array[i] + amplitudes_array_for_adding[i] * test_true_amplitudes_on_freq_array[i])
       for i in range(len(signal_amplitudes_on_freq_array))])
print("freq")
print(test_true_amplitudes_on_freq_array)
print("r_ampl")
print(r_amplitudes_array)

# stego-signal 10 freq  sin (2 for work)
hidden = [get_signal_with_freq(all_work_freq, yinterp[n], amplitudes_array_for_adding) for n in range(data_size)]

hidden_before_transform = [get_signal_with_freq(all_work_freq, xvals_ext[n], amplitudes_array_for_adding) for n in
                           range(data_size)]

plt.figure(1)
plt.title('Hidden before transform')
plt.xlabel('k')
plt.ylabel('Amplitude')
plt.plot(abs(array(fft(hidden_before_transform))[0: len(hidden_before_transform) // 44]), 'r')
plt.show()

plt.figure(1)
plt.title('Sin after transform of time')
plt.xlabel('k')
plt.ylabel('Amplitude')
plt.plot((array(hidden)[0: 1000]), 'r')
plt.show()

mix = (hidden + sound) // 2

output_signal = array(mix).astype("h")
wavfile.write('with_mix_250.wav', fs, output_signal)

c = fft(mix)  # calculate fourier transform (complex numbers list)
d = len(c) // 2  # rewrite d caused by c (meaning: you only need half of the fft list (real signal symmetry))

plt.figure(2)
plt.title('Fourier transform on sound with sin')
plt.xlabel('k')
plt.ylabel('Amplitude')
plt.plot(abs(c[nu * Ts - 100:nu * Ts + 100]), 'r')
plt.show()

'''
Перекодировка сигнала
'''
convert_to_mp3()
convert_to_wav()

fs, sound_after_converting_wav_mp3 = wavfile.read('./output/result_file.wav')

input_sound_length = len(sound_after_converting_wav_mp3)
graph_length = input_sound_length // 22

mix_rev_after_convering = interpolaite(y_rev, xvals, sound_after_converting_wav_mp3)
mix_rev = interpolaite(y_rev, xvals, mix)
hidden_rev = interpolaite(y_rev, xvals, hidden)

length_mix_rev = len(mix_rev_after_convering)

mix_rev_converted_fft = fft(mix_rev_after_convering)
mix_rev_fft = fft(mix_rev)

hidden_rev_fft = fft(hidden_rev)

mix_fft = fft(mix)
sound_fft = fft(sound)
hidden_fft = fft(hidden)

'''
После преобразования Фурье смешанного сиганла, вычисляем значения амплитуд сигналов по служеьным синусоидам для 0 и 1
'''

true_level = abs(mix_rev_converted_fft[2 * Ts * all_work_freq[-1]])
false_level = abs(mix_rev_converted_fft[2 * Ts * all_work_freq[-2]])
threshold_level = (true_level + false_level) / 2

print("threshold level data")
print(true_level)
print(false_level)
print(threshold_level)

for i in range(len(all_provided_bits)):
    print("mix_rev_on", "{" + str(i) + "}" + " ", abs(mix_rev_converted_fft[2 * Ts * all_work_freq[i]]))

'''
Распознаем переданную информацию имея ключи - значения амплитуд для 0 и 1
'''
decoded_array = [abs(mix_rev_converted_fft[2 * Ts * all_work_freq[i]]) > threshold_level for i in
                 range(len(all_provided_bits))]

print("Decoded array of secret message bits:")
print(decoded_array)

plt.subplot(3, 3, 1)
plt.title('Mix')
plt.xlabel('k')
plt.ylabel('Amplitude')
# plt.plot(abs(mix_rev_fft[(length_mix_rev // length)*nu*Ts-100:(length_mix_rev // length)*nu*Ts+100]),'r')
plt.plot(abs(mix_fft[0:  graph_length]), 'r')

plt.subplot(3, 3, 2)
plt.title('Sound')
plt.xlabel('k')
plt.ylabel('Amplitude')
# plt.plot(abs(mix_rev_fft[(length_mix_rev // length)*nu*Ts-100:(length_mix_rev // length)*nu*Ts+100]),'r')
plt.plot(abs(sound_fft[0:  graph_length]), 'r')

plt.subplot(3, 3, 3)
plt.title('Hidden')
plt.xlabel('k')
plt.ylabel('Amplitude')
# plt.plot(abs(mix_rev_fft[(length_mix_rev // length)*nu*Ts-100:(length_mix_rev // length)*nu*Ts+100]),'r')
plt.plot(abs(hidden_fft[0:  graph_length]), 'r')

plt.subplot(3, 3, 4)
plt.title('Fourier transform on reversed time sound with sin')
plt.xlabel('k')
plt.ylabel('Amplitude')
# plt.plot(abs(mix_rev_fft[(length_mix_rev // length)*nu*Ts-100:(length_mix_rev // length)*nu*Ts+100]),'r')
plt.plot(abs(mix_rev_converted_fft[2 * Ts * nu - 100:  2 * Ts * nu + 100]), 'r')
# plt.plot(abs(mix_rev_converted_fft[0:  graph_length]),'r')


plt.subplot(3, 3, 5)
plt.title("Mix rev")
plt.xlabel('k')
plt.ylabel('Amplitude')
# plt.plot(abs(mix_rev_fft[(length_mix_rev // length)*nu*Ts-100:(length_mix_rev // length)*nu*Ts+100]),'r')
plt.plot(abs(mix_rev_fft[2 * Ts * nu - 100:  2 * Ts * nu + 100]), 'r')
# plt.plot(abs(mix_rev_fft[0:  graph_length]),'r')

plt.subplot(3, 3, 6)
plt.title("Clear signal rev")
plt.xlabel('k')
plt.ylabel('Amplitude')
plt.plot(abs(sound_rev_fft[(length_mix_rev // length) * nu * Ts - 100:(length_mix_rev // length) * nu * Ts + 100]), 'r')
# plt.plot(abs(sound_rev_fft[0:  graph_length]),'r')

plt.subplot(3, 3, 7)
plt.title("Hidden rev")
plt.xlabel('k')
plt.ylabel('Amplitude')
# plt.plot(abs(mix_rev_fft[(length_mix_rev // length)*nu*Ts-100:(length_mix_rev // length)*nu*Ts+100]),'r')
# plt.plot(abs(hidden_rev_fft[2*Ts*nu - 100:  2*Ts*nu  + 100]),'r')
plt.plot(abs(hidden_rev_fft[0:  graph_length]), 'r')

plt.show()
