# ⚛️ Higgs Boson Classifier — Team B

> Neural network-based signal/background classifier for the HiggsML challenge (BlackSwan dataset, FAIR Universe framework).

---

## 📁 Project Structure

```
higgsteam-B/
├── main.py                            # Entry point
├── data.py                            # Data loading, cleaning & splitting
├── NeuralNetwork_Training_Collab.ipynb  # Main training notebook (Colab)
├── distribution.py                    # Signal/background distribution plots
├── learning_curve.py                  # Loss & accuracy curves (PyTorch)
├── significance.py                    # Poisson significance optimization
└── weighted_roc.py                    # Weighted ROC & AUC computation
```

---

## 🎯 Objective

Classify particle collision events from the ATLAS detector as either:

- **Signal (S = 1)** — Higgs boson decay via H → ττ
- **Background (B = 0)** — Standard Model background processes

The classifier is optimised to maximise the physics significance **s / √(s + b)**, where *s* and *b* are the weighted signal and background yields after a score cut.

---

## 🔬 Dataset

| Property | Value |
|---|---|
| Source | `HiggsML` — `blackSwan_data` |
| Train / Test split | 75% / 25% (sequential, CERN protocol) |
| Features used | `PRI_met`, `PRI_met_phi`, `PRI_lep_pt`, `PRI_lep_phi`, `PRI_had_pt`, `PRI_had_phi` |
| Preprocessing | `StandardScaler` + class-weight equalisation on train only |
| Test weights | **Untouched** — physical sums preserved (S ≈ 1 015, B ≈ 1 050 370) |

---

## 🧠 Model Architecture (Keras)

```
Input (6 features)
  └─ Dense(64, swish) → Dropout(0.2)
  └─ Dense(64, swish) → Dropout(0.2)
  └─ Dense(64, swish) → Dropout(0.2)
  └─ Dense(32, swish)
  └─ Dense(1, sigmoid)          ← binary output
```

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam (lr = 0.001) |
| Loss | Binary Cross-Entropy |
| Epochs | 30 |
| Batch size | 2 048 |

---

## 📊 Evaluation Metrics

Three physics-standard diagnostics are computed on the test set:

1. **Score separability** — weighted histogram overlay of S vs B classifier scores
2. **Weighted ROC curve** — AUC computed with physical event weights
3. **Significance scan** — threshold sweep to find the optimal cut maximising s / √(s + b)

All plots are saved as `.png` files and displayed in a dashboard at the end of the notebook.

---

## 🚀 Quickstart (Google Colab)

1. Open `NeuralNetwork_Training_Collab.ipynb` in Colab.
2. The notebook will automatically clone the repo and install dependencies.
3. Run all cells in order.

**Manual install:**

```bash
pip install HiggsML==0.1.5 xgboost tensorflow scikit-learn matplotlib
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `HiggsML` | Dataset download & loading |
| `TensorFlow / Keras` | Neural network (primary classifier) |
| `PyTorch` | Learning curve demonstration |
| `scikit-learn` | Preprocessing, ROC, train/test split |
| `NumPy / Pandas` | Data manipulation |
| `Matplotlib` | All plots |
| `OpenCV` (`cv2`) | Dashboard figure composition |

---

## 👥 Team

**ST4 — BlackSwan · Team B**
Repository: [ST4-BlackSwan/Higgs-TeamB](https://github.com/ST4-BlackSwan/Higgs-TeamB)
