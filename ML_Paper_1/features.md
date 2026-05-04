# Machine Learning paper – features used vs ignored

This document summarizes which fields/features are **used** and **ignored** by the paper:

> **“Detection of Retinal Dysfunction with Multimodal PERG Analysis: A Patient-Level Hybrid Machine Learning Framework”**

It is based on the paper text and on the columns available in `participants_info.csv`:

```
id_record	date	age_years	sex	diagnosis1	diagnosis2	diagnosis3	va_re_logMar	va_le_logMar	unilateral	rep_record	comments
```

---

## 1) Target label (what the models predict)

### Used
- **`diagnosis1`** → converted into a binary label:
  - `diagnosis1 == "Normal"` → **Normal** (y = 0)
  - `diagnosis1 != "Normal"` → **Pathological** (y = 1)

### Ignored / not used for label definition
- **`diagnosis2`**, **`diagnosis3`**
  - The paper says they are kept only as comorbidity information and are not used to define the main groups.

---

## 2) Clinical + demographic input features (metadata)

These are used as model inputs together with PERG waveform features.

### Used
- **`age_years`** (Age)
- **`sex`** (Sex)
  - Encoded with one-hot encoding
- **Eye-specific visual acuity in logMAR**
  - For a **right-eye** sample: **`va_re_logMar`**
  - For a **left-eye** sample: **`va_le_logMar`**

### Ignored / not used as ML inputs (mentioned in dataset but not described as model inputs)
- **`date`**
- **`unilateral`**
- **`rep_record`** (follow-up linkage; discussed for dataset structure, not used as input feature)
- **`comments`** (free-text narrative comments)

---

## 3) PERG waveform-derived input features (from each eye’s signal)

The paper does **feature-based ML** (not raw waveform deep learning).

### Preprocessing (before feature extraction)
- Band-pass filtering: **1–45 Hz**, 4th-order, zero-phase Butterworth
- Baseline correction: subtract mean of first **20 ms**
- Z-normalization/scaling is applied (paper states z-normalization to reduce inter-individual amplitude differences)

### PERG components detected (time windows)
- **N35**: 25–45 ms
- **P50**: 45–70 ms
- **N95**: 80–130 ms

### Derived features computed

**A) Latencies (ms)**
- `N35_ms`
- `P50_ms`
- `N95_ms`

**B) Amplitude differences**
- `Amp_N35P50` (often reported as z-normalized: `Amp_N35P50_z`)
- `Amp_P50N95` (often reported as z-normalized: `Amp_P50N95_z`) — primary biomarker

**C) Ratio**
- `Ratio_N95P50` (paper naming)  
  - described as the ratio between the two amplitude differences (N35–P50 and P50–N95)

**D) Whole-wave summaries (0–150 ms)**
- `RMS_0_150`
- `PTP_0_150` (often reported as `PTP_0_150_z`)

### Special note: RMS excluded in Model 1
- In **Model 1 (Dual-stream feature fusion)** the paper explicitly says it **excludes RMS** because it “remains constant across eyes”.
  - Model 1 PERG inputs (7): `N35_ms`, `P50_ms`, `N95_ms`, `Amp_N35P50_z`, `Amp_P50N95_z`, `Ratio_N95P50`, `PTP_0_150_z`

### Ignored / not used
- The **raw waveform samples** (full 255-point time series) are not used as ML input.

---

## 4) Handling multiple trials (aggregation to eye-level rows)

If multiple trials exist for the same `(Patient_ID, id_record, Eye)` (e.g., `RE_1`, `RE_2`, `RE_3`):
- Features are computed per trial, then one eye-level row is created by taking the **median** of each feature across trials.

This is how record-level sessions become eye-level samples.

---

## 5) Model input summary (by model)

### Model 1: Dual-stream feature fusion
- **PERG stream (7 features):**
  - `N35_ms`, `P50_ms`, `N95_ms`
  - `Amp_N35P50_z`, `Amp_P50N95_z`
  - `Ratio_N95P50`
  - `PTP_0_150_z`
- **Clinical stream:**
  - `age_years`, `sex`, and VA for the corresponding eye (`va_re_logMar` or `va_le_logMar`)

### Model 2: Stacking ensemble
- Uses a combined feature vector (PERG + clinical/VA). The paper describes the combined vector but does not list additional exclusions in the same explicit way as Model 1.

### Model 3: Two-stage cascade
- Stage 1: fast “gate” model (e.g., XGBoost)
- Stage 2: richer hybrid model (Model 1 or Model 2)

---

## 6) Quick checklist

### Used in ML inputs
- PERG-derived: N35/P50/N95 latencies, N35–P50 amplitude, P50–N95 amplitude, ratio, PTP (and sometimes RMS)
- Clinical: age, sex, corresponding-eye logMAR VA

### Ignored (not used as ML inputs)
- Raw waveform samples
- date, unilateral, rep_record, comments
- diagnosis2, diagnosis3 (not used for primary label definition)