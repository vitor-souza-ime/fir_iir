import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

fs = 1000
f_sinal = 50
t = np.linspace(0, 1, fs, endpoint=False)

senoidal = np.sin(2 * np.pi * f_sinal * t)
ruido = np.random.normal(0, 0.5, fs)
impulso = np.zeros(fs); impulso[0] = 1
misto = senoidal + ruido

num_taps = 51
fir_coef = signal.firwin(num_taps, cutoff=100, fs=fs, window="hamming")
fir_sistema = (fir_coef, [1.0])

iir_b, iir_a = signal.butter(4, Wn=100, fs=fs, btype='low')
iir_sistema = (iir_b, iir_a)

def aplica_filtro(sistema, sinal):
    b, a = sistema
    return signal.lfilter(b, a, sinal)

def analise_resposta(b, a, label):
    w, h = signal.freqz(b, a, fs=fs)

    impulso = np.zeros(100)
    impulso[0] = 1
    resp_impulso = signal.lfilter(b, a, impulso)

    degrau = np.ones(100)
    resp_degrau = signal.lfilter(b, a, degrau)

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.stem(resp_impulso)  # <- Linha corrigida
    plt.title(f'Resposta ao Impulso ({label})')
    plt.xlabel('Amostras')

    plt.subplot(1, 3, 2)
    plt.plot(w, 20 * np.log10(abs(h)))
    plt.title(f'Resposta em Frequência ({label})')
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Magnitude (dB)')

    plt.subplot(1, 3, 3)
    plt.plot(resp_degrau)
    plt.title(f'Resposta ao Degrau ({label})')
    plt.xlabel('Amostras')

    plt.tight_layout()
    plt.show()

def compara_resposta_entrada(sinais, nomes, sistemas, nomes_filtros):
    for sinal, nome_sinal in zip(sinais, nomes):
        plt.figure(figsize=(12, 6))
        plt.plot(t, sinal, label='Original', alpha=0.6)
        for sistema, nome_filtro in zip(sistemas, nomes_filtros):
            filtrado = aplica_filtro(sistema, sinal)
            plt.plot(t, filtrado, label=f'{nome_filtro}')
        plt.title(f'Resposta ao sinal: {nome_sinal}')
        plt.xlabel('Tempo (s)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Executar análises
analise_resposta(*fir_sistema, label="FIR")
analise_resposta(*iir_sistema, label="IIR")

sinais = [senoidal, ruido, impulso, misto]
nomes = ['Senoidal', 'Ruído Branco', 'Impulso de Dirac', 'Sinal Misto']
sistemas = [fir_sistema, iir_sistema]
nomes_filtros = ['FIR', 'IIR']

compara_resposta_entrada(sinais, nomes, sistemas, nomes_filtros)
