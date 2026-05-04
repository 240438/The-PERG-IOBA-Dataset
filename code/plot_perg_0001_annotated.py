import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

CSV_PATH = "./../perg-ioba-dataset/csv/0001.csv"
OUT_PATH = "perg_0001_annotated.png"

# Paper reports ~1700 Hz sampling
FS = 1700.0
APPLY_FILTER = True

def bandpass_1_45hz(x, fs=FS, low=1.0, high=45.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="bandpass")
    return filtfilt(b, a, x)

def baseline_correct(x, t_ms, baseline_end_ms=20.0):
    mask = t_ms <= baseline_end_ms
    return x - np.mean(x[mask])

def find_components(x, t_ms):
    def idx_between(a, b):
        return np.where((t_ms >= a) & (t_ms <= b))[0]

    n35_idx = idx_between(25, 45)   # N35 window
    p50_idx = idx_between(45, 70)   # P50 window
    n95_idx = idx_between(80, 130)  # N95 window

    i_n35 = n35_idx[np.argmin(x[n35_idx])]
    i_p50 = p50_idx[np.argmax(x[p50_idx])]
    i_n95 = n95_idx[np.argmin(x[n95_idx])]
    return i_n35, i_p50, i_n95

def compute_metrics(x, t_ms, i_n35, i_p50, i_n95):
    n35_lat, p50_lat, n95_lat = t_ms[i_n35], t_ms[i_p50], t_ms[i_n95]
    n35_val, p50_val, n95_val = x[i_n35], x[i_p50], x[i_n95]

    amp_n35_p50 = p50_val - n35_val
    amp_p50_n95 = p50_val - n95_val
    ratio = amp_p50_n95 / amp_n35_p50 if amp_n35_p50 != 0 else np.nan

    ptp = np.max(x) - np.min(x)
    rms = np.sqrt(np.mean(x**2))

    return {
        "N35_latency_ms": n35_lat,
        "P50_latency_ms": p50_lat,
        "N95_latency_ms": n95_lat,
        "N35_value": n35_val,
        "P50_value": p50_val,
        "N95_value": n95_val,
        "Amp_N35P50": amp_n35_p50,
        "Amp_P50N95": amp_p50_n95,
        "Ratio_(P50-N95)/(N35-P50)": ratio,
        "PTP_0_150": ptp,
        "RMS_0_150": rms,
    }

def draw_bracket(ax, x, y0, y1, text, color="black", lw=2, text_dx=1.5):
    """
    Draw a vertical bracket at time x from y0 to y1 with label.
    """
    ax.plot([x, x], [y0, y1], color=color, lw=lw)
    cap = 0.8  # small horizontal cap (in ms units visually; ok for plotting)
    ax.plot([x - cap, x + cap], [y0, y0], color=color, lw=lw)
    ax.plot([x - cap, x + cap], [y1, y1], color=color, lw=lw)

    y_mid = (y0 + y1) / 2
    ax.text(x + text_dx, y_mid, text, color=color, va="center")

# ---- Load data ----
df = pd.read_csv(CSV_PATH)

t = pd.to_datetime(df["TIME_1"])
t0 = t.iloc[0]
t_ms = (t - t0).dt.total_seconds().to_numpy() * 1000.0

signals = {
    "RE_1": df["RE_1"].to_numpy(float),
    "LE_1": df["LE_1"].to_numpy(float),
}

fig, ax = plt.subplots(figsize=(14, 6))

for name, x_raw in signals.items():
    x = x_raw.copy()
    if APPLY_FILTER:
        x = bandpass_1_45hz(x)
        x = baseline_correct(x, t_ms, baseline_end_ms=20.0)

    i_n35, i_p50, i_n95 = find_components(x, t_ms)
    m = compute_metrics(x, t_ms, i_n35, i_p50, i_n95)

    # Print metrics
    print("\n===", name, "===")
    for k, v in m.items():
        print(f"{k:28s}: {v}")

    # Plot waveform
    ax.plot(t_ms, x, label=name)

    # Mark N35/P50/N95 points
    ax.scatter([m["N35_latency_ms"]], [m["N35_value"]], s=60, marker="v")
    ax.scatter([m["P50_latency_ms"]], [m["P50_value"]], s=60, marker="^")
    ax.scatter([m["N95_latency_ms"]], [m["N95_value"]], s=60, marker="v")

    ax.annotate(f"{name} N35\n{m['N35_latency_ms']:.1f} ms",
                (m["N35_latency_ms"], m["N35_value"]),
                textcoords="offset points", xytext=(0, -35), ha="center")
    ax.annotate(f"{name} P50\n{m['P50_latency_ms']:.1f} ms",
                (m["P50_latency_ms"], m["P50_value"]),
                textcoords="offset points", xytext=(0, 10), ha="center")
    ax.annotate(f"{name} N95\n{m['N95_latency_ms']:.1f} ms",
                (m["N95_latency_ms"], m["N95_value"]),
                textcoords="offset points", xytext=(0, -35), ha="center")

    # Draw amplitude brackets at P50 time (visually clear)
    x_br = m["P50_latency_ms"]

    # Amp_N35P50: from N35_value up to P50_value
    draw_bracket(
        ax,
        x=x_br + (1.5 if name == "RE_1" else 4.0),  # small offset so RE/LE brackets don't overlap
        y0=m["N35_value"],
        y1=m["P50_value"],
        text=f"Amp_N35P50 = {m['Amp_N35P50']:.2f}",
        color=("tab:blue" if name == "RE_1" else "tab:orange"),
        text_dx=1.5,
    )

    # Amp_P50N95: from N95_value up to P50_value
    draw_bracket(
        ax,
        x=x_br + (8.0 if name == "RE_1" else 10.5),
        y0=m["N95_value"],
        y1=m["P50_value"],
        text=f"Amp_P50N95 = {m['Amp_P50N95']:.2f}",
        color=("tab:blue" if name == "RE_1" else "tab:orange"),
        text_dx=1.5,
    )

# Shade detection windows
ax.axvspan(25, 45, alpha=0.08, color="gray")
ax.axvspan(45, 70, alpha=0.08, color="gray")
ax.axvspan(80, 130, alpha=0.08, color="gray")

ax.set_title("PERG 0001.csv with N35/P50/N95 markers + amplitude brackets")
ax.set_xlabel("Time (ms from trial start)")
ax.set_ylabel("Amplitude (after optional filter + baseline correction)")
ax.set_xlim(0, 150)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
print(f"\nSaved annotated plot to: {OUT_PATH}")