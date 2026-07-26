import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
plt.interactive(True)

def adaptive_threshold(signal, window_size):

    threshold = np.zeros_like(signal)
    for i in range(len(signal)):
        start = max(0, i - window_size // 2)
        end = min(len(signal), i + window_size // 2)
        window = signal[start:end]
        median_val = np.median(window)
        percentile_val = np.percentile(window, 99)
        threshold[i] = median_val + 1.7 * (percentile_val - median_val)
    return threshold

def coarse_cfo(seg,sync_word,fs,f_step: float = 1.0):

    n = len(sync_word)
    t = np.arange(n) / fs
    freqs = np.arange(-fs/2, fs/2 + f_step, f_step)
    val = np.zeros(len(freqs))
    counter = 0
    for f in freqs:
        sync_shift = sync_word * np.exp(1j * 2 * np.pi * f * t)
        val[counter] = np.abs(np.vdot(seg,sync_shift))
        counter+=1
    freq_est = freqs[np.argmax(val)]
    return freq_est

def cfo_fft_parabolic(seg,ref,fs,zero_pad = 4):

    mul = seg * np.conj(ref)
    FFT = np.fft.fftshift(np.fft.fft(mul, len(mul)*zero_pad))
    PSD = np.abs(FFT)
    max_idx = np.argmax(PSD)
    if 1 <= max_idx < len(PSD) - 1:
        P1 = PSD[max_idx-1]
        P2 = PSD[max_idx]
        P3 = PSD[max_idx+1]
        if abs(P1 - 2*P2 + P3) > 1e-20:
            delta = 0.5 * (P1 - P3) / (P1 - 2*P2 + P3)
        else:
            delta = 0.0
    else:
        delta = 0.0
    idx_true = max_idx + delta
    f_hat = -fs / 2 + idx_true * fs / len(PSD)
    return f_hat

Data_1 = np.fromfile("Raw__14085_432225000_19200__19200__19_46_20_369.raw", dtype=np.int16)
Data_2 = np.fromfile("Raw__14076_432225000_19200__19200__19_46_20_400.raw", dtype=np.int16)

Data_1_complex = (Data_1[::2].astype(np.float64) + 1j * Data_1[1::2].astype(np.float64))
Data_2_complex = (Data_2[::2].astype(np.float64) + 1j * Data_2[1::2].astype(np.float64))

fs = 19200
t1 = np.arange(len(Data_1_complex))/fs
t2 = np.arange(len(Data_2_complex))/fs

plt.figure()
plt.plot(t1, np.abs(Data_1_complex), label="Регистрация 1")
plt.plot(t2, np.abs(Data_2_complex), label="Регистрация 2", c="green")

plt.legend()
plt.xlabel("Время, с")
plt.ylabel("Модуль значений")
plt.grid(linestyle="--")

dibit_to_freq = {
    (0, 1): 1944.0,
    (0, 0): 648.0,
    (1, 0): -648.0,
    (1, 1): -1944.0
}

N_symbol = 24
samples_per_symbol = fs // 4800

SYNC_PATTERNS_HEX = {
    "MS_DATA":        0xD5D7F77FD757,
    "MS_VOICE":       0x7F7D5DD57DFD
}

Kr_1 = []
Kr_2 = []
sync = []
frequencies = np.zeros(N_symbol)
sync_signal = np.zeros(N_symbol * samples_per_symbol, dtype=np.complex128)

for k in SYNC_PATTERNS_HEX.keys():
    sync_hex = SYNC_PATTERNS_HEX[k]
    sync_bits = np.array([int(x) for x in f"{sync_hex:048b}"])
    phase = 0
    for i in range(N_symbol):
        bit1 = sync_bits[2 * i]
        bit0 = sync_bits[2 * i + 1]
        frequencies[i] = dibit_to_freq[(bit1, bit0)]
        for j in range(samples_per_symbol):
            phase += 2 * np.pi * frequencies[i] / fs
            sync_signal[i * samples_per_symbol + j] = np.exp(1j * phase)

    sync.append(np.copy(sync_signal))
    Kr_temp_1 = np.correlate(Data_1_complex,np.copy(sync_signal), mode="valid")
    Kr_temp_2 = np.correlate(Data_2_complex,np.copy(sync_signal), mode="valid")
    Kr_1.append(Kr_temp_1)
    Kr_2.append(Kr_temp_2)
    if k=="MS_VOICE":
        plt.figure()
        plt.plot(np.abs(Kr_temp_1), label = "Модуль ККФ для регистрации 1 (голос)")
        plt.plot(np.abs(Kr_temp_2), label = "Модуль ККФ для регистрации 2 (голос)", c="red")
        # plt.plot(np.roll(np.abs(Kr_temp_2),600), label = "Модуль ККФ для регистрации 2 (голос) - сдвиг",c="black")
        plt.xlabel("Отсчеты")
        plt.ylabel("|ККФ|")
        plt.legend()
        plt.grid(linestyle="--")
    else:
        plt.figure()
        plt.plot(np.abs(Kr_temp_1), label="Модуль ККФ для регистрации 1 (данные)")
        plt.plot(np.abs(Kr_temp_2), label="Модуль ККФ для регистрации 2 (данные)", c="red")
        plt.xlabel("Отсчеты")
        plt.ylabel("|ККФ|")
        plt.legend()
        plt.grid(linestyle="--")

threshold_percentile_1 = adaptive_threshold(np.abs(Kr_1[1]), int(30e-3*fs))
threshold_percentile_2 = adaptive_threshold(np.abs(Kr_2[1]), int(30e-3*fs))
plt.plot(threshold_percentile_1, label = "Порог 1", c="purple")
plt.plot(threshold_percentile_2, label = "Порог 2")
plt.legend()
peak_idx_voice_1 = np.abs(Kr_1[1])>threshold_percentile_1
peak_idx_voice_1 = np.where(peak_idx_voice_1)[0]
peak_idx_voice_2 = np.abs(Kr_2[1])>threshold_percentile_2
peak_idx_voice_2 = np.where(peak_idx_voice_2)[0]
counter_1 = 0
idx_temp_1 = []
dict_idx_1 = {}

for i in range(len(peak_idx_voice_1)-1):
    if (peak_idx_voice_1[i+1]-peak_idx_voice_1[i])<(int(360e-3*fs)-10):
        idx_temp_1.append(peak_idx_voice_1[i])
        continue
    dict_idx_1[counter_1] = np.copy(idx_temp_1)
    idx_temp_1 = []
    counter_1+=1

counter_2 = 0
idx_temp_2 = []
dict_idx_2 = {}
for i in range(len(peak_idx_voice_2)-1):
    if (peak_idx_voice_2[i+1]-peak_idx_voice_2[i])<(int(360e-3*fs)-10):
        idx_temp_2.append(peak_idx_voice_2[i])
        continue
    dict_idx_2[counter_2] = np.copy(idx_temp_2)
    idx_temp_2 = []
    counter_2+=1

true_idx_1 = np.array([np.max(dict_idx_1[i]) for i in list(dict_idx_1.keys())])
sync_shift_1 = np.array([Data_1_complex[p:p+N_symbol*samples_per_symbol] for p in true_idx_1])
freq_1 = np.array([coarse_cfo(m,sync[1],fs) for m in sync_shift_1])

true_idx_2 = np.array([np.max(dict_idx_2[i]) for i in list(dict_idx_2.keys())])
sync_shift_2 = np.array([Data_2_complex[p:p+N_symbol*samples_per_symbol] for p in true_idx_2])
freq_2 = np.array([coarse_cfo(m,sync[1],fs) for m in sync_shift_2])

##Альтернативный метод определения CFO через FFT и интерполяцию параболой##
freq_FFT_parabolic_1 = []
for i in sync_shift_1:
    freq_FFT_parabolic_1.append(cfo_fft_parabolic(i,sync[1],fs))
freq_FFT_parabolic_1 = np.array(freq_FFT_parabolic_1)
freq_FFT_parabolic_2 = []
for i in sync_shift_2:
    freq_FFT_parabolic_2.append(cfo_fft_parabolic(i,sync[1],fs))
freq_FFT_parabolic_2 = np.array(freq_FFT_parabolic_2)

min_len = min(len(true_idx_2), len(true_idx_1))
idx_shift = true_idx_1[:min_len]-true_idx_2[:min_len]

x1_a = Data_1_complex[idx_shift[0]:]
x2_a = Data_2_complex
min_len_data = min(len(x1_a), len(x2_a))

mult_sig = x1_a[:min_len_data]*np.conj(x2_a[:min_len_data])
phi = np.unwrap(np.angle(mult_sig))
df = np.diff(phi)*fs/(2*np.pi)

window = int(0.1*fs)
w = np.abs(mult_sig[:-1])**2
num = np.convolve(w*df, np.ones(window), 'same')
den = np.convolve(w, np.ones(window), 'same')
df_smooth = num/den

freq_interp_1 = np.interp(np.arange(len(Data_1_complex)),true_idx_1,freq_1)
freq_interp_2 = np.interp(np.arange(len(Data_2_complex)),true_idx_2,freq_2)

plt.figure()
plt.plot(t1,freq_interp_1,label = "Зависимость f(t) для регистрации 1")
plt.plot(t2,freq_interp_2,label = "Зависимость f(t) для регистрации 2", c="green")
plt.xlabel("Время, с")
plt.ylabel("Частота, Гц")
plt.legend()
plt.grid(linestyle='--')

freq_len = min(len(freq_interp_1), len(freq_interp_2))
freq_diff = freq_interp_1[:freq_len] - freq_interp_2[:freq_len]
plt.figure()
plt.plot(t2[:len(df_smooth)],df_smooth, label = "Сглаженное значение df(t)")
plt.plot(t2[:len(df_smooth)],freq_diff[:len(df_smooth)], c="purple", label = "Разность f1-f2")
plt.xlabel("Время, с")
plt.ylabel("Частота, Гц")
plt.legend()
plt.grid(linestyle='--')
print("")