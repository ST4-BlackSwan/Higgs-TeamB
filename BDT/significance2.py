import numpy as np
import matplotlib.pyplot as plt

from HiggsML.datasets import download_dataset
from sklearn.model_selection import train_test_split

from boosted_decision_tree_scale_pos_weight import BoostedDecisionTreeScalePosWeight
data = download_dataset("blackSwan_data")
data.load_train_set()
data_set = data.get_train_set()

train_data = data_set.drop(columns=["labels", "weights", "detailed_labels"])
labels = data_set["labels"]
weights = data_set["weights"]


X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    train_data,
    labels,
    weights,
    test_size=0.3,
    random_state=42,
    stratify=labels
)

# here we use the hyperparameters found with the bayesian method.

bdt = BoostedDecisionTreeScalePosWeight(
    n_estimators=1000,
    max_depth=9,
    learning_rate=0.08501869815505453,
    subsample=1.0,
    colsample_bytree=0.6,
    min_child_weight=1,
    tree_method="hist",
    random_state=31415,
    early_stopping_rounds=15
)

bdt.fit(X_train, y_train, w_train)

predictions = bdt.predict(X_test)


sort_indices = np.argsort(predictions)[::-1]
scores_sorted = predictions[sort_indices]
y_sorted = y_test.to_numpy()[sort_indices]
w_sorted = w_test.to_numpy()[sort_indices]

# Somme cumulative des poids pour obtenir S et B à chaque seuil possible
S_cumulative = np.cumsum(w_sorted * (y_sorted == 1))
B_cumulative = np.cumsum(w_sorted * (y_sorted == 0))

# Filtre de sécurité pour éviter les divisions par zéro
valid_mask = (S_cumulative > 0) & (B_cumulative > 0)

Z_cumulative = np.zeros_like(S_cumulative, dtype=float)
S_v = S_cumulative[valid_mask]
B_v = B_cumulative[valid_mask]

# Formule de l'Approximate Median Significance (AMS) du CERN
Z_cumulative[valid_mask] = np.sqrt(2 * ((S_v + B_v) * np.log(1 + S_v / B_v) - S_v))

# Trouver le Working Point optimal
best_idx = np.argmax(Z_cumulative)
best_threshold = scores_sorted[best_idx]
best_significance = Z_cumulative[best_idx]

print(f"\n=== RÉSULTATS ===")
print(f"Seuil BDT optimal (Cut) : {best_threshold:.4f}")
print(f"Signification Maximale Z : {best_significance:.3f}")

# ==========================================
# 4. TRACÉ DE LA "SIGNIFICANCE CURVE" DUAL
# ==========================================
print("-> Génération du graphique...")


print(f"Au pic (Z={best_significance:.2f}) :")
print(f" -> Signal restant (S) = {S_cumulative[best_idx]:.2f}")
print(f" -> Bruit restant (B) = {B_cumulative[best_idx]:.2f}")

fig, ax1 = plt.subplots(figsize=(10, 6))

# Histogramme de distribution du signal et du bruit (Axe de gauche)
ax1.hist(predictions[y_test == 1], bins=50, range=(0, 1), weights=w_test[y_test == 1],
         alpha=0.35, color="dodgerblue", label="Signal (Poids physiques)", density=True)
ax1.hist(predictions[y_test == 0], bins=50, range=(0, 1), weights=w_test[y_test == 0],
         alpha=0.35, color="crimson", label="Background (Poids physiques)", density=True)

ax1.set_xlabel("Score continu du BDT", fontsize=11, fontweight="bold")
ax1.set_ylabel("Densité de probabilité (Normalisée)", color="black", fontsize=11)
ax1.tick_params(axis='y', labelcolor="black")
ax1.grid(True, linestyle=":", alpha=0.5)

# Courbe d'évolution de la signification (Axe de droite)
ax2 = ax1.twinx()
ax2.plot(scores_sorted, Z_cumulative, color="darkviolet", lw=2.5, label="Signification $Z$ (AMS)")
ax2.set_ylabel("Signification Statistique $Z$", color="darkviolet", fontsize=11, fontweight="bold")
ax2.tick_params(axis='y', labelcolor="darkviolet")

# Ligne du cut optimal
ax2.axvline(best_threshold, linestyle="--", color="black", lw=1.5,
            label=f"Cut optimal = {best_threshold:.3f}\n$Z_{{max}}$ = {best_significance:.2f}")

# Fusion des légendes des deux axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", framealpha=0.9)

plt.title("Optimisation de la Signification Statistique (CERN AMS) sur le BDT", fontsize=12, pad=15)
plt.xlim(0, 1)
plt.tight_layout()
plt.show()