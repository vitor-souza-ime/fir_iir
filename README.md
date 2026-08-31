# 🎧 FIR and IIR Filter Analysis with Python

This project presents a **quantitative and reproducible comparison of digital FIR and IIR low-pass filters** using `scipy.signal`, `numpy`, and `matplotlib`.

The revised implementation goes beyond a purely visual comparison by including **frequency-domain metrics, phase response, group delay, transient behavior, noise reduction, SNR analysis, stability, computational cost, execution-time benchmarking, and Monte Carlo experiments**.

---

## 📌 Overview

The script evaluates the filters using four types of discrete-time signals:

- 🔵 **Sine wave** at 50 Hz
- ⚪ **White Gaussian noise**
- ⚡ **Discrete unit impulse** (Kronecker delta)
- 🌀 **Mixed signal** (50 Hz sine wave + white noise)

Two comparison scenarios are considered.

### Scenario A — Nominal Design Comparison

- 🟩 **FIR filter**
  - 51 taps
  - Hamming window
  - Nominal cutoff: 100 Hz

- 🟥 **IIR filter**
  - 4th-order Butterworth
  - Nominal critical frequency: 100 Hz

This scenario preserves the original filter definitions and demonstrates that the same numerical frequency parameter does **not** imply equivalent frequency responses.

### Scenario B — Comparison Under Common Specifications

A second comparison is performed using common performance requirements:

- Passband edge: **80 Hz**
- Stopband edge: **150 Hz**
- Maximum passband loss/ripple criterion: **1 dB**
- Minimum stopband attenuation: **40 dB**

The original 51-tap FIR filter is compared with the **minimum-order Butterworth filter** obtained with `scipy.signal.buttord()` for the same specifications.

---

## 🛠️ Requirements

Python **3.10 or newer** is recommended.

Install the required libraries with:

```bash
pip install numpy scipy matplotlib
```

---

## 🚀 How to Run

Clone or download the repository and execute:

```bash
python main.py
```

The script automatically creates a folder named:

```text
resultados/
```

All figures and quantitative output files are stored in this directory.

---

## 🔁 Reproducibility Parameters

The experiment uses fixed parameters so that the numerical results can be reproduced.

```text
Sampling frequency:        1000 Hz
Signal duration:           1.0 s
Number of samples:         1000
Sine-wave frequency:       50 Hz
Sine-wave amplitude:       1.0
Noise standard deviation:  0.5
Random seed:               20260830
Monte Carlo repetitions:   500
Frequency-response points: 65536
```

The random-number generator uses:

```python
numpy.random.default_rng(20260830)
```

---

## 🧪 Features

### ✅ Filter Design

The script designs and evaluates:

- FIR low-pass filter with `scipy.signal.firwin()`
- 4th-order Butterworth IIR filter with `scipy.signal.butter()`
- Minimum-order Butterworth filter satisfying the common passband/stopband specifications with `scipy.signal.buttord()`

For IIR filtering, **second-order sections (SOS)** are used with `scipy.signal.sosfilt()`.

---

### ✅ Frequency-Response Analysis

The frequency response is calculated with:

- `scipy.signal.freqz()`
- `scipy.signal.sosfreqz()`

The magnitude is normalized by the DC response and converted to decibels using:

```text
20 log10(|H(f)|)
```

The analysis includes:

- Magnitude at 50, 80, 100, 120, 150, and 200 Hz
- −3 dB frequency
- −6 dB frequency
- Transition width up to −40 dB
- Passband variation
- Worst-case stopband attenuation

---

### ✅ Phase and Group Delay

The script calculates:

- Phase response
- Group delay
- Mean group delay
- Group-delay standard deviation
- Minimum and maximum group delay in the passband

For the symmetric 51-tap FIR filter, the theoretical group delay is:

```text
(51 - 1) / 2 = 25 samples
```

At a sampling frequency of 1000 Hz, this corresponds to:

```text
25 ms
```

---

### ✅ Impulse-Response Analysis

For each filter, the script evaluates:

- Impulse-response energy
- Effective response duration
- Effective duration in milliseconds

The FIR response has exactly 51 coefficients.

For the IIR filters, the response is theoretically infinite. Therefore, the script uses a numerical truncation criterion based on the remaining tail energy.

---

### ✅ Step-Response Analysis

The following transient-response metrics are calculated:

- Final value
- 10–90% rise time
- Settling time using a **±2% criterion**
- Percentage overshoot

---

### ✅ White-Noise Analysis

The same white-noise realization is filtered by the FIR and IIR filters.

The script generates a **Welch power spectral density estimate** for:

- Input noise
- FIR-filtered noise
- IIR-filtered noise

This allows the noise attenuation to be evaluated in the frequency domain.

---

### ✅ Monte Carlo Analysis

Noise and SNR results are not based on a single random realization.

The experiment performs **500 Monte Carlo repetitions** and calculates:

- Input-noise variance
- Output-noise variance
- Mean and standard deviation of the variances
- Input SNR
- Output SNR
- Mean and standard deviation of SNR
- SNR improvement after filtering

The first 0.2 s of each realization is discarded from the power estimates to reduce the influence of startup transients.

---

### ✅ Stability Analysis

The poles of the IIR filters are explicitly calculated and plotted in the **z-plane**.

The analysis includes:

- Pole locations
- Pole magnitudes
- Unit-circle visualization

Stability can therefore be verified directly by confirming that all poles lie inside the unit circle.

---

### ✅ Computational-Cost Analysis

The script reports theoretical arithmetic cost for different implementation structures.

For FIR filters:

- Direct implementation
- Symmetry-exploiting implementation

For IIR filters:

- Direct-form estimate
- Second-order-section estimate

This avoids treating a single operation count as universally representative of every possible implementation.

---

### ✅ Execution-Time Benchmark

Actual execution time is also measured.

For each filter:

- A random signal containing **200,000 samples** is filtered
- Five warm-up runs are performed
- Thirty timed repetitions are executed
- Median execution time is reported
- Interquartile range (IQR) is reported

The benchmark also records information about the execution environment.

---

## 📊 Generated Figures

The script automatically generates the following figures:

```text
figura_1_resposta_impulso.png
figura_2_resposta_frequencia_nominal.png
figura_3_resposta_degrau.png
figura_4_fase_atraso_grupo.png
figura_5_senoidal_zoom.png
figura_6_ruido_psd.png
figura_7_sinal_misto_zoom.png
figura_8_plano_z.png
figura_9_comparacao_especificacoes.png
```

These figures include:

- 📈 Impulse responses
- 📉 Frequency responses
- 🧱 Step responses
- 🔄 Phase response
- ⏱️ Group delay
- 🔍 Time-domain zooms
- 📊 Noise power spectral density
- 🌀 Mixed-signal filtering
- ⭕ Pole-zero diagrams
- ⚖️ Comparison under common specifications

---

## 📄 Numerical Output Files

In addition to the figures, the script creates:

### `metricas.csv`

Contains the quantitative metrics calculated for each filter, including:

- Frequency-response values
- Passband and stopband metrics
- Group delay
- Impulse-response energy
- Step-response metrics
- Noise variance
- SNR
- Monte Carlo statistics
- Runtime benchmark

### `coeficientes.txt`

Contains:

- Python version
- NumPy version
- SciPy version
- Matplotlib version
- Operating-system information
- CPU identification
- FIR coefficients
- IIR coefficients
- SOS coefficients
- IIR pole magnitudes

---

## 📂 Project Structure

```text
📁 fir_iir/
├── main.py
├── README.md
├── requirements.txt
└── resultados/
    ├── figura_1_resposta_impulso.png
    ├── figura_2_resposta_frequencia_nominal.png
    ├── figura_3_resposta_degrau.png
    ├── figura_4_fase_atraso_grupo.png
    ├── figura_5_senoidal_zoom.png
    ├── figura_6_ruido_psd.png
    ├── figura_7_sinal_misto_zoom.png
    ├── figura_8_plano_z.png
    ├── figura_9_comparacao_especificacoes.png
    ├── metricas.csv
    └── coeficientes.txt
```

> The `resultados/` directory is created automatically when the script is executed.

---

## 📘 Topics Covered

- Digital signal processing
- FIR and IIR digital filters
- Hamming-window FIR design
- Butterworth filter design
- Frequency-response analysis
- Phase response
- Group delay
- Impulse response
- Step response
- Passband and stopband specifications
- White-noise filtering
- Power spectral density
- Signal-to-noise ratio
- Monte Carlo simulation
- Pole-zero analysis
- Digital-filter stability
- Computational complexity
- Execution-time benchmarking
- Reproducible scientific computing with Python

---

## 🔬 Scientific Reproducibility

This repository accompanies a comparative study of FIR and IIR digital filters.

The revised experimental procedure was designed to make the comparison quantitatively reproducible by explicitly defining:

- Filter-design parameters
- Test-signal parameters
- Random seed
- Frequency-response calculation procedure
- Passband and stopband criteria
- Transient-response criteria
- Noise and SNR metrics
- Monte Carlo repetition count
- Numerical implementation structure
- Execution environment

---

## 🏁 License

This project is open-source and free to use under the MIT License.
