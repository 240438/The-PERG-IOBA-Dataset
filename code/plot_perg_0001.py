import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional filtering (matches paper's idea: 4th order, zero-phase, 1–45 Hz)
from scipy.signal import butter, filtfilt

CSV_PATH = "./../perg-ioba-dataset/csv/0001.csv"   # change if needed
APPLY_FILTER = True

# PERG-IOBA: 255 samples over ~150 ms => fs ~ 1700 Hz (paper states 1700 Hz)
FS = 1700.0

def bandpass_1_45hz(x, fs=FS, low=1.0, high=45.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="bandpass")
    return filtfilt(b, a, x)

def baseline_correct(x, t_ms, baseline_end_ms=20.0):
    mask = t_ms <= baseline_end_ms
    return x - np.mean(x[mask])

def find_components(x, t_ms):
    """
    Very simple peak picking inside fixed windows:
    - N35: min in 25–45 ms
    - P50: max in 45–70 ms
    - N95: min in 80–130 ms
    """
    def window_idx(a, b):
        return np.where((t_ms >= a) & (t_ms <= b))[0]

    n35_idx = window_idx(25, 45)
    p50_idx = window_idx(45, 70)
    n95_idx = window_idx(80, 130)

    i_n35 = n35_idx[np.argmin(x[n35_idx])]
    i_p50 = p50_idx[np.argmax(x[p50_idx])]
    i_n95 = n95_idx[np.argmin(x[n95_idx])]

    return i_n35, i_p50, i_n95

def compute_metrics(x, t_ms, i_n35, i_p50, i_n95):
    n35_lat, p50_lat, n95_lat = t_ms[i_n35], t_ms[i_p50], t_ms[i_n95]
    n35_val, p50_val, n95_val = x[i_n35], x[i_p50], x[i_n95]

    amp_n35_p50 = p50_val - n35_val          # N35->P50 rise
    amp_p50_n95 = p50_val - n95_val          # P50->N95 drop magnitude (positive if N95 is negative)

    ratio = amp_p50_n95 / amp_n35_p50 if amp_n35_p50 != 0 else np.nan

    # PTP and RMS over 0–150 ms
    ptp = np.max(x) - np.min(x)
    rms = np.sqrt(np.mean(x**2))

    return {
        "N35_latency_ms": n35_lat,
        "P50_latency_ms": p50_lat,
        "N95_latency_ms": n95_lat,
        "N35_value": n35_val,
        "P50_value": p50_val,
        "N95_value": n95_val,
        "Amp_N35_P50": amp_n35_p50,
        "Amp_P50_N95": amp_p50_n95,
        "Ratio_(P50-N95)/(N35-P50)": ratio,
        "PTP_0_150": ptp,
        "RMS_0_150": rms,
    }

# --- Load ---
df = pd.read_csv(CSV_PATH)
t = pd.to_datetime(df["TIME_1"])
t0 = t.iloc[0]
t_ms = (t - t0).dt.total_seconds().to_numpy() * 1000.0

signals = {
    "RE_1": df["RE_1"].to_numpy(dtype=float),
    "LE_1": df["LE_1"].to_numpy(dtype=float),
}

plt.figure(figsize=(12, 6))

for name, x_raw in signals.items():
    x = x_raw.copy()

    if APPLY_FILTER:
        x = bandpass_1_45hz(x)
        x = baseline_correct(x, t_ms, baseline_end_ms=20.0)

    i_n35, i_p50, i_n95 = find_components(x, t_ms)
    metrics = compute_metrics(x, t_ms, i_n35, i_p50, i_n95)

    # Print metrics
    print("\n===", name, "===")
    for k, v in metrics.items():
        print(f"{k:28s}: {v}")

    # Plot waveform
    plt.plot(t_ms, x, label=f"{name}")

    # Mark components
    plt.scatter([metrics["N35_latency_ms"]], [metrics["N35_value"]], s=60, marker="v")
    plt.scatter([metrics["P50_latency_ms"]], [metrics["P50_value"]], s=60, marker="^")
    plt.scatter([metrics["N95_latency_ms"]], [metrics["N95_value"]], s=60, marker="v")

    # Annotate
    plt.annotate(f"N35\n{metrics['N35_latency_ms']:.1f} ms",
                 (metrics["N35_latency_ms"], metrics["N35_value"]),
                 textcoords="offset points", xytext=(0, -30), ha="center")
    plt.annotate(f"P50\n{metrics['P50_latency_ms']:.1f} ms",
                 (metrics["P50_latency_ms"], metrics["P50_value"]),
                 textcoords="offset points", xytext=(0, 10), ha="center")
    plt.annotate(f"N95\n{metrics['N95_latency_ms']:.1f} ms",
                 (metrics["N95_latency_ms"], metrics["N95_value"]),
                 textcoords="offset points", xytext=(0, -30), ha="center")

plt.axvspan(25, 45, alpha=0.08)
plt.axvspan(45, 70, alpha=0.08)
plt.axvspan(80, 130, alpha=0.08)

plt.title("PERG waveform for 0001.csv (RE_1 and LE_1) with N35/P50/N95 detection")
plt.xlabel("Time (ms from trial start)")
plt.ylabel("Amplitude (raw units; after optional filtering/baseline correction)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 150)
plt.tight_layout()

out_path = "perg_0001_plot.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"\nSaved plot to: {out_path}")