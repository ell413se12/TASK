clc
clear
close all

fid_1 = fopen('Raw__14085_432225000_19200__19200__19_46_20_369.raw');
fid_2 = fopen('Raw__14076_432225000_19200__19200__19_46_20_400.raw');

Data_1 = fread(fid_1,"int16");
fclose(fid_1);
Data_2 = fread(fid_2,"int16");
fclose(fid_2);
Data_1_complex = Data_1(1:2:end) + 1j*Data_1(2:2:end);
Data_2_complex = Data_2(1:2:end) + 1j*Data_2(2:2:end);
fs = 19200;
t1 = (1:length(Data_1_complex))./fs;
t2 = (1:length(Data_2_complex))./fs;
figure();
plot(t1,abs(Data_1_complex));
grid on;
figure();
plot(t2,abs(Data_2_complex),"r");
grid on;

bits = 48;
N_symbol = bits/2; %48 бит на синхрослово = 24 символа на синхрослово%
T_symbol = 1/4800; %Длительность символа%
samples_per_symbol = floor(fs*T_symbol);

SYNC_PATTERNS_HEX = dictionary(["MS_VOICE"],["0x7F7D5DD57DFD"]);

sync_bits = dec2bin(hex2dec(SYNC_PATTERNS_HEX(SYNC_PATTERNS_HEX.keys())));
KEYS = SYNC_PATTERNS_HEX.keys();
sync_word_0 = sync_word_gen(SYNC_PATTERNS_HEX(KEYS), ...
    N_symbol,samples_per_symbol,fs,bits);
sync_word = sync_word_gen_gauss(SYNC_PATTERNS_HEX(KEYS), ...
    N_symbol,samples_per_symbol,fs,bits);
sync_word_2 = sync_word_gen_gauss_2(SYNC_PATTERNS_HEX(KEYS), ...
    N_symbol,samples_per_symbol,fs,bits);
Kr_temp_1_0 = conv(Data_1_complex,conj(flip(sync_word_0)),"valid");
Kr_temp_1 = conv(Data_1_complex,conj(flip(sync_word)),"valid");
Kr_temp_1_2 = conv(Data_1_complex,conj(flip(sync_word_2)),"valid");
figure()
hold on;

plot(abs(Kr_temp_1_2),"r");
plot(abs(Kr_temp_1),"b");
plot(abs(Kr_temp_1_0),"k");
xlabel("Отсчеты");
ylabel("|ККФ|");
grid on;

Data_1_diff = Data_1_complex(2:end).*conj(Data_1_complex(1:end-1));
sync_word_diff = sync_word_0(2:end).*conj(sync_word_0(1:end-1));
Kr_diff = conv(Data_1_diff,conj(flip(sync_word_diff)),"valid");
L_sync = length(sync_word_diff);
energy_data = conv(abs(Data_1_diff).^2, ones(1, L_sync), "valid");
energy_sync = sum(abs(sync_word_diff).^2);
Kr_norm = Kr_diff ./ (sqrt(energy_data .* energy_sync));
figure()
plot(abs(Kr_norm));

function sync_word = sync_word_gen(hex,N_symbol,sps,fs, bits)
    
    dibit_to_freq = dictionary([0,1,2,3],[648,1944,-648,-1944]);
    sync_signal = zeros([1,N_symbol*sps]);
    sync_bits = reshape(dec2bin(hex2dec(hex),bits),2,[])';
    freqs = dibit_to_freq(bin2dec(sync_bits));
    phase = 0;
    for k = 1:length(freqs)
        for t = 1:sps
            phase = phase + 2*pi*freqs(k)/fs;
            sync_signal((k-1)*sps+t) = exp(1j*phase);
        end
    end
    sync_word = sync_signal;
end

function sync_word = sync_word_gen_gauss(hex,N_symbol,sps,fs, bits)
    
    dibit_to_freq = dictionary([0,1,2,3],[648,1944,-648,-1944]);
    sync_signal = zeros([1,N_symbol*sps]);
    sync_bits = reshape(dec2bin(hex2dec(hex),bits),2,[])';
    freqs = dibit_to_freq(bin2dec(sync_bits));
    
    raw_freqs = zeros(1,N_symbol*sps);
    for k = 1:length(freqs)
        for t=1:sps
            raw_freqs((k-1)*sps+t) = freqs(k);
        end
    end
    BT = 0.5;
    filter_span = 4;
    h_gauss = gaussdesign(BT,filter_span,sps);
    smooth_freqs = conv(raw_freqs,h_gauss,"full");
    group_delay = (filter_span*sps)/2;
    start_idx = group_delay + 1;
    end_idx = start_idx + (N_symbol*sps)-1;
    smooth_freqs = smooth_freqs(start_idx:end_idx);
    plot(raw_freqs);
    hold on;
    plot(smooth_freqs);
    phase = 0;
    for k = 1:length(smooth_freqs)
        phase = phase + 2*pi*smooth_freqs(k)/fs;
        sync_signal(k) = exp(1j*phase);
    end
    sync_word = sync_signal;
end

function sync_word = sync_word_gen_gauss_2(hex,N_symbol,sps,fs, bits)
    
    dibit_to_freq = dictionary([0,1,2,3],[648,1944,-648,-1944]);
    sync_bits = reshape(dec2bin(hex2dec(hex),bits),2,[])';
    freqs = dibit_to_freq(bin2dec(sync_bits));
    
    raw_freqs = zeros(1,N_symbol*sps);
    for k = 1:length(freqs)
        for t=1:sps
            raw_freqs((k-1)*sps+t) = freqs(k);
        end
    end

    BT = 0.5;
    filter_span = 4;
    h_gauss = gaussdesign(BT,filter_span,sps);
    smooth_freqs = conv(raw_freqs,h_gauss,"full");
    phase = 0;
    sync_signal = zeros([1,length(smooth_freqs)]);
    for k = 1:length(smooth_freqs)
        phase = phase + 2*pi*smooth_freqs(k)/fs;
        sync_signal(k) = exp(1j*phase);
    end

    group_delay = (filter_span*sps)/2;
    start_idx = group_delay+1;
    end_idx = start_idx+(N_symbol*sps)-1; 
    sync_word = sync_signal(start_idx:end_idx);
end