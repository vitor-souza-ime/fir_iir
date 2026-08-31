from __future__ import annotations

import csv
import platform
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import signal

# ----------------------------- Reproducibility -----------------------------
FS = 1000.0
DURATION_S = 1.0
N_SAMPLES = int(FS * DURATION_S)
SIGNAL_FREQ_HZ = 50.0
SIGNAL_AMPLITUDE = 1.0
NOISE_STD = 0.5
RNG_SEED = 20260830
N_MONTE_CARLO = 500
FREQ_POINTS = 65536
DB_FLOOR = -160.0

# Common objective bands used for quantitative comparison.
FP_HZ = 80.0      # passband edge
FSB_HZ = 150.0    # stopband edge
RP_DB = 1.0       # maximum passband loss/ripple criterion
RS_DB = 40.0      # minimum stopband attenuation criterion

OUT = Path(__file__).resolve().parent / "resultados"
OUT.mkdir(parents=True, exist_ok=True)


def db20(x: np.ndarray | float) -> np.ndarray | float:
    """20*log10 magnitude with numerical floor only to avoid log(0)."""
    x_arr = np.asarray(x)
    result = 20.0 * np.log10(np.maximum(np.abs(x_arr), np.finfo(float).tiny))
    if np.isscalar(x):
        return float(result)
    return result


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close()


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "não identificado"


def freq_response_ba(b: np.ndarray, a: np.ndarray):
    f, h = signal.freqz(b, a, worN=FREQ_POINTS, fs=FS)
    # Normalize magnitude by the DC response. For these low-pass designs |H(0)|≈1,
    # but the explicit normalization makes the reported procedure unambiguous.
    h_norm = h / np.abs(h[0])
    mag_db = db20(h_norm)
    return f, h_norm, mag_db


def freq_response_sos(sos: np.ndarray):
    f, h = signal.sosfreqz(sos, worN=FREQ_POINTS, fs=FS)
    h_norm = h / np.abs(h[0])
    mag_db = db20(h_norm)
    return f, h_norm, mag_db


def value_at(f: np.ndarray, y: np.ndarray, freq_hz: float) -> float:
    return float(y[np.argmin(np.abs(f - freq_hz))])


def first_crossing(f: np.ndarray, mag_db: np.ndarray, threshold_db: float, start_hz=0.0) -> float:
    mask = f >= start_hz
    idxs = np.flatnonzero(mask & (mag_db <= threshold_db))
    return float(f[idxs[0]]) if idxs.size else float("nan")


def group_delay_stats(b: np.ndarray, a: np.ndarray, fp: float = FP_HZ):
    # Avoid the exact Nyquist endpoint and use a dense frequency grid.
    w = np.linspace(0, 2 * np.pi * 250.0 / FS, 8192)
    with np.errstate(divide="ignore", invalid="ignore"):
        w_out, gd = signal.group_delay((b, a), w=w)
    f = w_out * FS / (2 * np.pi)
    mask = (f >= 0) & (f <= fp) & np.isfinite(gd)
    vals = gd[mask]
    return f, gd, {
        "gd_mean_samples": float(np.mean(vals)),
        "gd_std_samples": float(np.std(vals)),
        "gd_min_samples": float(np.min(vals)),
        "gd_max_samples": float(np.max(vals)),
    }


def impulse_metrics(apply_filter, exact_fir_len: int | None = None):
    n = 4096
    impulse = np.zeros(n)
    impulse[0] = 1.0
    h = apply_filter(impulse)
    total_energy = float(np.sum(h * h))
    if exact_fir_len is not None:
        n_eff = exact_fir_len
        energy = float(np.sum(h[:n_eff] ** 2))
    else:
        # Effective truncation: remaining tail energy below 1e-12 of total.
        reverse_tail = np.cumsum((h[::-1] ** 2))[::-1]
        threshold = total_energy * 1e-12
        candidates = np.flatnonzero(reverse_tail <= threshold)
        n_eff = int(candidates[0]) if candidates.size else n
        n_eff = max(n_eff, 1)
        energy = float(np.sum(h[:n_eff] ** 2))
    return h, {
        "impulse_energy": energy,
        "impulse_effective_samples": n_eff,
        "impulse_effective_ms": 1000.0 * n_eff / FS,
    }


def step_metrics(apply_filter):
    n = 500
    y = apply_filter(np.ones(n))
    final = float(np.mean(y[-100:]))
    lo10, hi90 = 0.1 * final, 0.9 * final
    i10s = np.flatnonzero(y >= lo10)
    i90s = np.flatnonzero(y >= hi90)
    i10 = int(i10s[0]) if i10s.size else 0
    i90 = int(i90s[0]) if i90s.size else 0
    rise_ms = 1000.0 * max(0, i90 - i10) / FS
    overshoot_pct = max(0.0, (float(np.max(y)) - final) / abs(final) * 100.0)
    tol = 0.02 * abs(final)
    outside = np.abs(y - final) > tol
    outside_idxs = np.flatnonzero(outside)
    settling_idx = int(outside_idxs[-1] + 1) if outside_idxs.size else 0
    settling_ms = 1000.0 * settling_idx / FS
    return y, {
        "step_final": final,
        "rise_time_ms_10_90": rise_ms,
        "settling_time_ms_2pct": settling_ms,
        "overshoot_pct": overshoot_pct,
    }


def spectral_metrics(f: np.ndarray, mag_db: np.ndarray):
    pass_mask = (f >= 0) & (f <= FP_HZ)
    stop_mask = (f >= FSB_HZ) & (f < FS / 2 - 1e-9)
    pb = mag_db[pass_mask]
    sb = mag_db[stop_mask]
    f3 = first_crossing(f, mag_db, -3.0)
    f6 = first_crossing(f, mag_db, -6.0)
    f40 = first_crossing(f, mag_db, -40.0, start_hz=FP_HZ)
    return {
        "f_minus3_db_hz": f3,
        "f_minus6_db_hz": f6,
        "transition_to_40db_hz": f40 - FP_HZ,
        "passband_variation_db_0_80": float(np.max(pb) - np.min(pb)),
        "worst_stopband_atten_db_150_500": float(-np.max(sb)),
        "mag_50_db": value_at(f, mag_db, 50.0),
        "mag_80_db": value_at(f, mag_db, 80.0),
        "mag_100_db": value_at(f, mag_db, 100.0),
        "mag_120_db": value_at(f, mag_db, 120.0),
        "mag_150_db": value_at(f, mag_db, 150.0),
        "mag_200_db": value_at(f, mag_db, 200.0),
    }


def monte_carlo_metrics(apply_filter, n_mc=N_MONTE_CARLO):
    rng = np.random.default_rng(RNG_SEED)
    t = np.arange(N_SAMPLES) / FS
    sine = SIGNAL_AMPLITUDE * np.sin(2 * np.pi * SIGNAL_FREQ_HZ * t)
    discard = 200  # discard 0.2 s to remove startup transient from power estimates

    var_in, var_out, snr_in, snr_out = [], [], [], []
    y_sig = apply_filter(sine)
    p_sig_in = float(np.mean(sine[discard:] ** 2))
    p_sig_out = float(np.mean(y_sig[discard:] ** 2))

    for _ in range(n_mc):
        noise = rng.normal(0.0, NOISE_STD, N_SAMPLES)
        y_noise = apply_filter(noise)
        p_noise_in = float(np.mean(noise[discard:] ** 2))
        p_noise_out = float(np.mean(y_noise[discard:] ** 2))
        var_in.append(float(np.var(noise[discard:], ddof=1)))
        var_out.append(float(np.var(y_noise[discard:], ddof=1)))
        snr_in.append(10 * np.log10(p_sig_in / p_noise_in))
        snr_out.append(10 * np.log10(p_sig_out / p_noise_out))

    def mean_std(vals):
        return float(np.mean(vals)), float(np.std(vals, ddof=1))

    vin_m, vin_s = mean_std(var_in)
    vout_m, vout_s = mean_std(var_out)
    si_m, si_s = mean_std(snr_in)
    so_m, so_s = mean_std(snr_out)
    return {
        "noise_var_in_mean": vin_m,
        "noise_var_in_std": vin_s,
        "noise_var_out_mean": vout_m,
        "noise_var_out_std": vout_s,
        "snr_in_db_mean": si_m,
        "snr_in_db_std": si_s,
        "snr_out_db_mean": so_m,
        "snr_out_db_std": so_s,
        "snr_gain_db": so_m - si_m,
    }


def benchmark(apply_filter):
    rng = np.random.default_rng(12345)
    x = rng.standard_normal(200_000)
    for _ in range(5):
        apply_filter(x)
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        apply_filter(x)
        times.append(time.perf_counter() - t0)
    arr = np.asarray(times)
    return {
        "runtime_ms_200k_median": float(np.median(arr) * 1000),
        "runtime_ms_200k_iqr": float((np.percentile(arr, 75) - np.percentile(arr, 25)) * 1000),
    }


def theoretical_cost(kind: str, order_or_taps: int, sos_sections: int | None = None):
    if kind == "FIR":
        taps = order_or_taps
        naive_mult = taps
        naive_add = taps - 1
        sym_mult = (taps + 1) // 2
        sym_add = taps - 1  # pair pre-additions + accumulator additions
        return naive_mult, naive_add, sym_mult, sym_add
    order = order_or_taps
    direct_mult = 2 * order + 1
    direct_add = 2 * order
    if sos_sections is None:
        sos_sections = int(np.ceil(order / 2))
    sos_mult = 5 * sos_sections
    sos_add = 4 * sos_sections
    return direct_mult, direct_add, sos_mult, sos_add


def main():
    # ------------------------- Signals -------------------------
    t = np.arange(N_SAMPLES) / FS
    sine = SIGNAL_AMPLITUDE * np.sin(2 * np.pi * SIGNAL_FREQ_HZ * t)
    rng = np.random.default_rng(RNG_SEED)
    noise = rng.normal(0.0, NOISE_STD, N_SAMPLES)
    mixed = sine + noise

    # ------------------- Scenario A: nominal -------------------
    fir_b = signal.firwin(51, cutoff=100.0, fs=FS, window="hamming")
    fir_a = np.array([1.0])
    iir4_b, iir4_a = signal.butter(4, Wn=100.0, fs=FS, btype="low", output="ba")
    iir4_sos = signal.butter(4, Wn=100.0, fs=FS, btype="low", output="sos")

    # ---------------- Scenario B: matched specs ----------------
    # The original FIR already meets Ap<=1 dB at 80 Hz and As>=40 dB at 150 Hz.
    # Design the minimum-order Butterworth filter for the same criteria.
    iir_eq_order, iir_eq_wn = signal.buttord(FP_HZ, FSB_HZ, RP_DB, RS_DB, fs=FS)
    iir_eq_b, iir_eq_a = signal.butter(iir_eq_order, iir_eq_wn, fs=FS, btype="low", output="ba")
    iir_eq_sos = signal.butter(iir_eq_order, iir_eq_wn, fs=FS, btype="low", output="sos")

    filters = {
        "FIR51_Hamming_100": {
            "b": fir_b,
            "a": fir_a,
            "apply": lambda x: signal.lfilter(fir_b, fir_a, x),
            "sos": None,
            "kind": "FIR",
            "order": 50,
            "taps": 51,
        },
        "IIR4_Butter_100": {
            "b": iir4_b,
            "a": iir4_a,
            "apply": lambda x: signal.sosfilt(iir4_sos, x),
            "sos": iir4_sos,
            "kind": "IIR",
            "order": 4,
            "taps": None,
        },
        "IIR8_Butter_matched": {
            "b": iir_eq_b,
            "a": iir_eq_a,
            "apply": lambda x: signal.sosfilt(iir_eq_sos, x),
            "sos": iir_eq_sos,
            "kind": "IIR",
            "order": int(iir_eq_order),
            "taps": None,
        },
    }

    # ---------------------- Quantitative metrics ----------------------
    all_metrics = {}
    responses = {}
    for name, cfg in filters.items():
        if cfg["sos"] is None:
            f, h, mag_db = freq_response_ba(cfg["b"], cfg["a"])
        else:
            f, h, mag_db = freq_response_sos(cfg["sos"])
        responses[name] = (f, h, mag_db)
        m = spectral_metrics(f, mag_db)
        _, gd, gd_stats = group_delay_stats(cfg["b"], cfg["a"])
        m.update(gd_stats)
        _, im = impulse_metrics(cfg["apply"], exact_fir_len=51 if cfg["kind"] == "FIR" else None)
        m.update(im)
        _, sm = step_metrics(cfg["apply"])
        m.update(sm)
        m.update(monte_carlo_metrics(cfg["apply"]))
        m.update(benchmark(cfg["apply"]))
        all_metrics[name] = m

    # ---------------------- Poles and coefficients ----------------------
    poles_iir4 = np.roots(iir4_a)
    poles_iir8 = np.roots(iir_eq_a)

    # ---------------------- Figures ----------------------
    # Fig. 1: impulse response, zoomed to relevant samples
    h_fir, _ = impulse_metrics(filters["FIR51_Hamming_100"]["apply"], exact_fir_len=51)
    h_iir4, _ = impulse_metrics(filters["IIR4_Butter_100"]["apply"])
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.4), sharex=True)
    axes[0].stem(np.arange(70), h_fir[:70], basefmt=" ")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("FIR, 51 coeficientes")
    axes[0].grid(True, alpha=0.3)
    axes[1].stem(np.arange(70), h_iir4[:70], basefmt=" ")
    axes[1].set_xlabel("Amostras")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("IIR Butterworth, ordem 4")
    axes[1].grid(True, alpha=0.3)
    savefig("figura_1_resposta_impulso.png")

    # Fig. 2: nominal frequency response
    plt.figure(figsize=(7.2, 4.6))
    for name, label in [("FIR51_Hamming_100", "FIR 51 taps"), ("IIR4_Butter_100", "IIR Butterworth ordem 4")]:
        f, _, mag = responses[name]
        plt.plot(f, np.maximum(mag, DB_FLOOR), label=label)
    plt.axvline(FP_HZ, linestyle="--", linewidth=1, label="Limite passante: 80 Hz")
    plt.axvline(FSB_HZ, linestyle=":", linewidth=1, label="Início rejeição: 150 Hz")
    plt.ylim(-120, 5)
    plt.xlim(0, 500)
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    savefig("figura_2_resposta_frequencia_nominal.png")

    # Fig. 3: step response
    y_fir, _ = step_metrics(filters["FIR51_Hamming_100"]["apply"])
    y_iir4, _ = step_metrics(filters["IIR4_Butter_100"]["apply"])
    nplot = 80
    time_ms = np.arange(nplot) * 1000 / FS
    plt.figure(figsize=(7.2, 4.5))
    plt.plot(time_ms, y_fir[:nplot], label="FIR 51 taps")
    plt.plot(time_ms, y_iir4[:nplot], label="IIR Butterworth ordem 4")
    plt.axhline(1.02, linestyle="--", linewidth=0.8)
    plt.axhline(0.98, linestyle="--", linewidth=0.8)
    plt.xlabel("Tempo (ms)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig("figura_3_resposta_degrau.png")

    # Fig. 4: phase and group delay
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True)
    for name, label in [("FIR51_Hamming_100", "FIR 51 taps"), ("IIR4_Butter_100", "IIR Butterworth ordem 4")]:
        f, h, _ = responses[name]
        mask = f <= 200
        phase = np.unwrap(np.angle(h))
        axes[0].plot(f[mask], np.rad2deg(phase[mask]), label=label)
        fg, gd, _ = group_delay_stats(filters[name]["b"], filters[name]["a"])
        maskg = (fg <= 200) & np.isfinite(gd) & (np.abs(gd) < 200)
        axes[1].plot(fg[maskg], gd[maskg], label=label)
    axes[0].set_ylabel("Fase (graus)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].set_xlabel("Frequência (Hz)")
    axes[1].set_ylabel("Atraso de grupo (amostras)")
    axes[1].grid(True, alpha=0.3)
    savefig("figura_4_fase_atraso_grupo.png")

    # Fig. 5: sine response with time alignment for visualization
    y_fir_sine = filters["FIR51_Hamming_100"]["apply"](sine)
    y_iir_sine = filters["IIR4_Butter_100"]["apply"](sine)
    fir_delay = 25
    iir_delay_50 = int(round(value_at(*group_delay_stats(iir4_b, iir4_a)[:2], 50.0))) if False else 5
    # Use 25 samples for FIR (exact) and 5 samples for IIR only for this 50 Hz single-tone visualization.
    def advance(y, d):
        out = np.full_like(y, np.nan)
        if d > 0:
            out[:-d] = y[d:]
        else:
            out[:] = y
        return out
    y_fir_align = advance(y_fir_sine, fir_delay)
    y_iir_align = advance(y_iir_sine, iir_delay_50)
    mask = (t >= 0.20) & (t <= 0.30)
    plt.figure(figsize=(7.2, 4.3))
    plt.plot(t[mask], sine[mask], label="Original")
    plt.plot(t[mask], y_fir_align[mask], label="FIR alinhado (25 amostras)")
    plt.plot(t[mask], y_iir_align[mask], label="IIR alinhado em 50 Hz (~5 amostras)")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    savefig("figura_5_senoidal_zoom.png")

    # Fig. 6: PSD of white-noise filtering (same realization)
    y_fir_noise = filters["FIR51_Hamming_100"]["apply"](noise)
    y_iir_noise = filters["IIR4_Butter_100"]["apply"](noise)
    f_psd, p0 = signal.welch(noise[200:], fs=FS, nperseg=512)
    _, p1 = signal.welch(y_fir_noise[200:], fs=FS, nperseg=512)
    _, p2 = signal.welch(y_iir_noise[200:], fs=FS, nperseg=512)
    plt.figure(figsize=(7.2, 4.5))
    plt.semilogy(f_psd, p0, label="Ruído de entrada")
    plt.semilogy(f_psd, p1, label="Após FIR")
    plt.semilogy(f_psd, p2, label="Após IIR")
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Densidade espectral de potência")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig("figura_6_ruido_psd.png")

    # Fig. 7: mixed signal, representative time zoom with FIR delay compensation
    y_fir_mixed = filters["FIR51_Hamming_100"]["apply"](mixed)
    y_iir_mixed = filters["IIR4_Butter_100"]["apply"](mixed)
    y_fir_mixed_al = advance(y_fir_mixed, fir_delay)
    y_iir_mixed_al = advance(y_iir_mixed, iir_delay_50)
    mask = (t >= 0.40) & (t <= 0.50)
    plt.figure(figsize=(7.2, 4.5))
    plt.plot(t[mask], mixed[mask], label="Sinal misto", alpha=0.55)
    plt.plot(t[mask], y_fir_mixed_al[mask], label="FIR alinhado")
    plt.plot(t[mask], y_iir_mixed_al[mask], label="IIR alinhado em 50 Hz")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    savefig("figura_7_sinal_misto_zoom.png")

    # Fig. 8: pole-zero plot for IIR filters
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.0))
    theta = np.linspace(0, 2 * np.pi, 400)
    for ax, b, a, title in [
        (axes[0], iir4_b, iir4_a, "IIR ordem 4 (nominal)"),
        (axes[1], iir_eq_b, iir_eq_a, f"IIR ordem {iir_eq_order} (especificações comuns)"),
    ]:
        z = np.roots(b)
        p = np.roots(a)
        ax.plot(np.cos(theta), np.sin(theta), linestyle="--", linewidth=1)
        ax.scatter(z.real, z.imag, marker="o", facecolors="none", label="Zeros")
        ax.scatter(p.real, p.imag, marker="x", label="Polos")
        ax.axhline(0, linewidth=0.6)
        ax.axvline(0, linewidth=0.6)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Parte real")
        ax.set_ylabel("Parte imaginária")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    savefig("figura_8_plano_z.png")

    # Fig. 9: fair comparison under common requirements
    plt.figure(figsize=(7.2, 4.6))
    for name, label in [("FIR51_Hamming_100", "FIR 51 taps"), ("IIR8_Butter_matched", f"Butterworth ordem {iir_eq_order}")]:
        f, _, mag = responses[name]
        plt.plot(f, np.maximum(mag, DB_FLOOR), label=label)
    plt.axvline(FP_HZ, linestyle="--", linewidth=1)
    plt.axvline(FSB_HZ, linestyle="--", linewidth=1)
    plt.axhline(-RP_DB, linestyle=":", linewidth=1, label="-1 dB")
    plt.axhline(-RS_DB, linestyle=":", linewidth=1, label="-40 dB")
    plt.xlim(0, 250)
    plt.ylim(-100, 5)
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    savefig("figura_9_comparacao_especificacoes.png")

    # ---------------------- CSV outputs ----------------------
    metric_keys = sorted({k for m in all_metrics.values() for k in m})
    with open(OUT / "metricas.csv", "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["filtro"] + metric_keys)
        for name, m in all_metrics.items():
            writer.writerow([name] + [m.get(k, "") for k in metric_keys])

    with open(OUT / "coeficientes.txt", "w", encoding="utf-8") as ftxt:
        ftxt.write("VERSOES\n")
        ftxt.write(f"Python {sys.version.split()[0]}\n")
        ftxt.write(f"NumPy {np.__version__}\n")
        ftxt.write(f"SciPy {scipy.__version__}\n")
        ftxt.write(f"Matplotlib {matplotlib.__version__}\n")
        ftxt.write(f"Plataforma {platform.platform()}\n")
        ftxt.write(f"CPU {cpu_model()}\n\n")
        ftxt.write("FIR 51 taps, Hamming, cutoff=100 Hz\n")
        ftxt.write(np.array2string(fir_b, precision=12, separator=", ") + "\n\n")
        ftxt.write("IIR Butterworth ordem 4, Wn=100 Hz, b\n")
        ftxt.write(np.array2string(iir4_b, precision=12, separator=", ") + "\n")
        ftxt.write("IIR Butterworth ordem 4, a\n")
        ftxt.write(np.array2string(iir4_a, precision=12, separator=", ") + "\n\n")
        ftxt.write(f"IIR Butterworth equivalentes: ordem={iir_eq_order}, Wn={iir_eq_wn:.12f} Hz\n")
        ftxt.write("SOS\n")
        ftxt.write(np.array2string(iir_eq_sos, precision=12, separator=", ") + "\n")
        ftxt.write("Polos IIR4 (modulos):\n")
        ftxt.write(np.array2string(np.abs(poles_iir4), precision=12, separator=", ") + "\n")
        ftxt.write("Polos IIR equivalente (modulos):\n")
        ftxt.write(np.array2string(np.abs(poles_iir8), precision=12, separator=", ") + "\n")

    # Summary on screen
    print("Environment:")
    print(f"Python {sys.version.split()[0]}, NumPy {np.__version__}, SciPy {scipy.__version__}, Matplotlib {matplotlib.__version__}")
    print(f"CPU: {cpu_model()}")
    print(f"Matched Butterworth: order={iir_eq_order}, Wn={iir_eq_wn:.6f} Hz")
    print("\nCoefficients (FIR unique half, b[0]..b[25]):")
    for i, c in enumerate(fir_b[:26]):
        print(f"b[{i:02d}] = {c:.12g}")
    print("\nIIR4 b:", np.array2string(iir4_b, precision=12))
    print("IIR4 a:", np.array2string(iir4_a, precision=12))
    print("IIR4 pole magnitudes:", np.abs(poles_iir4))
    print("\nMetrics:")
    for name, m in all_metrics.items():
        print(f"\n{name}")
        for k, v in m.items():
            print(f"  {k}: {v:.12g}")

    print("\nTheoretical cost per output sample:")
    print("FIR51 (naive mult/add; symmetric mult/add):", theoretical_cost("FIR", 51))
    print("IIR4 (direct mult/add; SOS mult/add):", theoretical_cost("IIR", 4, iir4_sos.shape[0]))
    print(f"IIR{iir_eq_order} (direct mult/add; SOS mult/add):", theoretical_cost("IIR", int(iir_eq_order), iir_eq_sos.shape[0]))


if __name__ == "__main__":
    main()
