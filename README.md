# IoT Washing Machine Vibration Analysis — Project README

**Course:** AAI 530 — Internet of Things (IoT) and Data Analytics  
**Project:** IoT Edge-Analytics for Smart Appliance Monitoring  
**Hardware:** [IndusBoard Coin V2](https://indus.electronicsforu.com) (ESP32-S2) with LSM303AGR Accelerometer  

This repository contains data collection firmware, Jupyter notebooks for exploratory data analysis (EDA), stage classification (Random Forest and LSTM), and drum failure/anomaly prediction. All models use vibration data collected at various washing machine stages and labeled with a **status** code (0, 1, 2).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Data Collection](#2-data-collection)
3. [Status Map and Datasets](#3-status-map-and-datasets)
4. [Notebooks and Workflows](#4-notebooks-and-workflows)
5. [Stage Classification](#5-stage-classification)
6. [Part 2: Drum Failure and Anomaly Detection](#6-part-2-drum-failure-and-anomaly-detection)
7. [Key Insights and Future Work](#7-key-insights-and-future-work)
8. [Project Structure](#8-project-structure)

---

## 1. Overview

The project turns a standard washing machine into a monitorable appliance by:

- **Collecting** 3-axis accelerometer data (X, Y, Z) at ~100 Hz from the IndusBoard Coin.
- **Classifying** the current stage (stopped, rinsing/slow spin, high speed/dry) from vibration patterns using **Random Forest** and **LSTM**.
- **Predicting** drum failures and anomalies using a **Bidirectional LSTM** trained on synthetic fault data.

Data is logged via Serial (e.g. CoolTerm), merged with manually annotated stage labels, and stored as `merged_train.csv` and `merged_test.csv`. The same status mapping is used in both the assignment notebook and the main analysis notebook.

---

## 2. Data Collection

### 2.1 Firmware: `src/sensor_data.cpp`

Vibration data is acquired on the **IndusBoard Coin V2** using the sketch in `src/sensor_data.cpp`.

**Behavior:**

- Initializes the **LSM303AGR** accelerometer over I²C (pins 8 and 9).
- In `loop()`, reads X, Y, Z via `Acc.GetAxes(acc)` and prints CSV lines: `Acc_X, Acc_Y, Acc_Z`.
- Uses a **10 ms** delay to approximate **100 Hz** sampling (ODR is fixed in the library; rate is effectively controlled in software).
- Serial baud rate: **115200** (for logging to PC or SD).

**Relevant snippet:**

```cpp
void loop() {
  int32_t acc[3];
  Acc.GetAxes(acc);
  Serial.print(acc[0]); Serial.print(",");
  Serial.print(acc[1]); Serial.print(",");
  Serial.println(acc[2]);
  delay(10);  // ~100 Hz
}
```

Data is collected during different washing machine phases; the **status** (0, 1, or 2) is assigned offline when building the merged CSVs.

### 2.2 Data Pipeline (from project design)

- **Sampling:** 100 Hz, 3 axes (X, Y, Z) in milli-Gs.
- **Magnitude:** Computed in the notebooks as \(\|a\| = \sqrt{X^2 + Y^2 + Z^2}\) for orientation-independent intensity.
- **Windowing:** For LSTM/RNN, data is grouped into fixed-length sequences (e.g. 50 samples = 0.5 s) with configurable stride.

---

## 3. Status Map and Datasets

### 3.1 Status Map

All modeling uses this mapping of the integer **status** column to stages:

| Status | Label               | Description                          |
|--------|---------------------|--------------------------------------|
| **0**  | **STOPPED**         | Machine on, drum stationary          |
| **1**  | **RINSING_SLOW_SPIN** | Rinsing or low-speed spin         |
| **2**  | **HIGH_SPEED_DRY**  | High-speed spin or dry cycle         |

In `washing_machine_vibration_analysis.ipynb`, this is implemented as:

```python
status_map = {
    0: 'STOPPED',
    1: 'RINSING_SLOW_SPIN',
    2: 'HIGH_SPEED_DRY'
}
df['label'] = df['status'].map(status_map)
```

### 3.2 Merged Datasets

- **`data/merged_train.csv`** — Training set. Columns: **X, Y, Z, status** (no header in file; names are set when loading).
- **`data/merged_test.csv`** — Test set. Same schema.

Typical sizes (from notebook runs): ~16,020 training rows, ~11,425 test rows. Class distributions (0, 1, 2) are reported in the notebooks after loading.

### 3.3 Data Dictionary

| Column   | Type    | Description                                  |
|----------|---------|----------------------------------------------|
| X        | int     | X-axis acceleration (milli-G)                |
| Y        | int     | Y-axis acceleration (milli-G)                |
| Z        | int     | Z-axis acceleration (milli-G)                |
| status   | int     | Stage label: 0, 1, or 2                      |

The main analysis notebook adds:

- **magnitude** = \(\sqrt{X^2 + Y^2 + Z^2}\)
- **label** = string version of status (e.g. `'STOPPED'`, `'RINSING_SLOW_SPIN'`, `'HIGH_SPEED_DRY'`)

---

## 4. Notebooks and Workflows

### 4.1 `Project_Assignment_AAI 530.ipynb`

- **Purpose:** Assignment notebook for AAI 530 (washing machine project).
- **Data:** Uses `merged_train_csv.csv` / `merged_test_csv.csv` with columns `x_acc`, `y_acc`, `z_acc`, `status` (same semantics as status map above).
- **Content:**
  - Load and audit train/test data (shapes, status counts, missing values).
  - **Random Forest** for **stage classification** (predict status 0/1/2 from `x_acc`, `y_acc`, `z_acc`).
  - **LSTM** for **future vibration prediction** (e.g. next 10 Z-axis values from a window of 50), with scaling, sliding windows, and MSE loss.
- **Result (RF):** The notebook reports **100% test accuracy**; the write-up attributes this to distinct mechanical signatures per stage and high-fidelity sampling of the IndusBoard IMU.

### 4.2 `washing_machine_vibration_analysis.ipynb`

- **Purpose:** Full pipeline: EDA, stage classification with LSTM, and failure/anomaly prediction.
- **Data:** `data/merged_train.csv`, `data/merged_test.csv` (X, Y, Z, status); adds magnitude and label.
- **Flow:**
  1. **Load & explore** — Load merged CSVs, apply status map, compute magnitude, show label distribution and basic stats.
  2. **EDA** — Statistics by stage, box plots (X, Y, Z, magnitude), time-series plots, correlation.
  3. **Synthetic fault generation** — Generate fault patterns (unbalanced, bearing, motor, belt, misalignment) for Part 2.
  4. **Preprocessing** — `create_sequences()` with `sequence_length=50`, optional **stride** (default `sequence_length` for non-overlapping windows to keep training fast). Features: **X, Y, Z, magnitude**. Scaling with `StandardScaler` fit on training data only; `LabelEncoder` for stage labels.
  5. **Part 1 — Stage classification:** LSTM 128→64, dropout, dense layers, softmax for 3 classes. Train/validation split from training data; test on held-out test sequences. Reports accuracy, classification report, confusion matrix, training history.
  6. **Part 2 — Failure prediction:** Binary normal vs fault. Combines normal sequences (from train+test) with fault sequences from synthetic fault data; Bidirectional LSTM (128→64), class weights, ROC–AUC, precision/recall.
  7. **Real-time simulation** — Example of predicting both stage and fault status for a single sequence.
  8. **Summary** — Project summary, metrics, key insights, future work.

---

## 5. Stage Classification

Stage classification predicts the current washing machine stage (0, 1, or 2) from vibration. Two approaches are implemented in this repo.

### 5.1 Random Forest (`Project_Assignment_AAI 530.ipynb`)

- **Input:** One row per sample: features = `x_acc`, `y_acc`, `z_acc`; target = `status` (0, 1, 2).
- **Model:** `RandomForestClassifier(n_estimators=100, random_state=42)`.
- **Train/Test:** Fit on `train_df`, evaluate on `test_df`.
- **Reported result:** 100% accuracy on the test set.
- **Explanation (from notebook):** Labels (Stop, Spin, Dry Spin) are well separated by motor RPM and vibration levels (e.g. Z-axis bands). Random Forest finds effective splits; the high accuracy is attributed to the **distinct mechanical signatures** of the appliance and the **high-fidelity sampling** of the IndusBoard IMU.

### 5.2 LSTM (`washing_machine_vibration_analysis.ipynb`)

- **Input:** Sequences of length **50** (0.5 s at 100 Hz), features **X, Y, Z, magnitude**. Stride is set to `sequence_length` by default to avoid redundant overlapping windows and to keep training time reasonable (e.g. ~300–500 sequences instead of tens of thousands).
- **Model:** Two-layer LSTM (**128 → 64** units), dropout (0.2), recurrent dropout (0.2), then Dense(64), Dropout(0.3), Dense(32), Dropout(0.3), Dense(3, softmax). Optimizer: Adam; loss: sparse categorical crossentropy. Callbacks: EarlyStopping, ReduceLROnPlateau.
- **Rationale (from notebook):**
  - **Why LSTM:** Vibration is temporal; each stage has evolving patterns (e.g. STOP = low/steady, RINSING_SLOW_SPIN = moderate/periodic, HIGH_SPEED_DRY = high amplitude). LSTM captures these dependencies.
  - **Why 128→64:** First layer learns rich temporal features; second compresses and abstracts. Progressive reduction helps generalization and is suitable for edge (IndusBoard) deployment.
  - **Why two layers:** Hierarchical features (low-level patterns → high-level stage).
  - **Why sequence length 50:** 0.5 s captures typical cycles and keeps inference efficient.
  - **Alternatives:** CNN (weaker temporal context), simple RNN (vanishing gradients), GRU (less capacity), Transformer (overkill for this size).

Stage labels in the LSTM notebook use the same status map (STOPPED, RINSING_SLOW_SPIN, HIGH_SPEED_DRY). Example reported metrics: high precision/recall per class and high overall accuracy on the test sequences.

---

## 6. Part 2: Drum Failure and Anomaly Detection

Part 2 is implemented only in **`washing_machine_vibration_analysis.ipynb`**. Because no defective appliance was available, **synthetic fault data** is generated and used to train a binary **normal vs fault** classifier.

### 6.1 Synthetic Fault Types

Fault patterns are generated from normal-stage statistics (mean/std of X, Y, Z) and labeled as fault. Types (as in the notebook):

1. **Unbalanced drum** — Irregular, high-amplitude spikes (e.g. 30% chance of large random offsets).
2. **Bearing failure** — High-frequency component (e.g. 20 Hz) with amplitude increasing over time.
3. **Motor malfunction** — Erratic variance and random jumps in X, Y, Z.
4. **Belt slippage** — Random “drops” where vibration amplitude is reduced (e.g. 20% of samples).
5. **Drum misalignment** — Asymmetric pattern: one axis (X, Y, or Z) is amplified.

Faults are generated for **HIGH_SPEED_DRY** (all five types) and **RINSING_SLOW_SPIN** (e.g. unbalanced, motor). The notebook combines these into `df_faults` and then builds fault sequences with the same `create_sequences()` (same length and stride as Part 1).

### 6.2 Failure Prediction Model

- **Task:** Binary: **normal** (0) vs **fault** (1). Normal = all sequences from merged train/test; fault = sequences from synthetic fault data.
- **Input:** Same as Part 1: 50 timesteps × 4 features (X, Y, Z, magnitude), scaled with the **same** `StandardScaler` fit on normal training data.
- **Model:** **Bidirectional LSTM**: Bidirectional(LSTM(128, return_sequences=True)), then Bidirectional(LSTM(64)), Dense(64), Dropout(0.3), Dense(32), Dropout(0.3), Dense(1, sigmoid). Binary crossentropy; class weights for imbalance; same callbacks as Part 1.
- **Evaluation:** Accuracy, precision, recall, ROC curve, AUC, confusion matrix.

Part 2 is intended for **predicting drum failures and detecting anomalies** (unbalanced load, bearing/motor/belt issues, misalignment). Results are indicative; validation on real fault data is recommended when available.

---

## 7. Key Insights and Future Work

**Insights (from `washing_machine_vibration_analysis.ipynb`):**

- LSTM/RNN models capture **temporal patterns** in vibration well; stages have distinct signatures.
- **Synthetic fault generation** allows training a failure detector without a faulty machine.
- The **128→64** (and Bidirectional 128→64) designs balance accuracy and size for potential **edge deployment** on IndusBoard Coin V2.

**Future work (from the same notebook):**

- Collect **real fault data** for validation and retraining.
- **Model quantization** for edge deployment.
- Extend fault types (e.g. water pump, door sensor).
- **Alerting** or dashboard for predictive maintenance.

---

## 8. Project Structure

```
aai-iot-vibration-analysis-project/
├── README.md                           # This file
├── project_details.txt                 # Project definition & design (AAI 530)
├── washing_machine_vibration_analysis.ipynb   # EDA, LSTM stage classification, Part 2 failure model
├── Project_Assignment_AAI 530.ipynb    # Assignment: RF stage classification, LSTM future prediction
├── src/
│   └── sensor_data.cpp                 # IndusBoard firmware: sample X,Y,Z at ~100 Hz
└── data/
    ├── merged_train.csv                # Training: X, Y, Z, status
    ├── merged_test.csv                 # Test: X, Y, Z, status
    └── (other stage-specific CSVs if used for earlier experiments)
```

### Running the Notebooks

1. **Environment:** Python 3 with pandas, numpy, matplotlib, seaborn, scikit-learn, TensorFlow (Keras). Install with pip or conda as needed.
2. **Paths:** In `washing_machine_vibration_analysis.ipynb`, merged files are loaded from `data/merged_train.csv` and `data/merged_test.csv`. Adjust if your files live elsewhere (e.g. `../merged_train.csv`).
3. **Execution order:** Run cells top to bottom; data loading and EDA must run before sequence creation, scaling, and model training. Part 2 depends on synthetic fault generation and the same scaler/sequence length as Part 1.

### References

- **IndusBoard Coin:** [IndusBoard — Electronics For You](https://indus.electronicsforu.com)
- **Course:** AAI 530 — Internet of Things (IoT) and Data Analytics
