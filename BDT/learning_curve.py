import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from HiggsML.datasets import download_dataset

from boosted_decision_tree_scale_pos_weight import BoostedDecisionTreeScalePosWeight


print("Loading dataset...", flush=True)

data = download_dataset("blackSwan_data")
data.load_train_set()
data_set = data.get_train_set()

print("Dataset loaded:", data_set.shape, flush=True)



train_data = data_set.drop(columns=["labels", "weights", "detailed_labels"])
labels = data_set["labels"]
weights = data_set["weights"]


# On garde une version complète du train set.

X_train_full, X_test, y_train_full, y_test, w_train_full, w_test = train_test_split(
    train_data,
    labels,
    weights,
    test_size=0.3,
    random_state=42,
    stratify=labels
)



train_fractions = np.linspace(0.1, 1.0, 5)

train_aucs = []
test_aucs = []
train_sizes = []

for frac in train_fractions:
    print(f"Training with {frac:.0%} of the training data...", flush=True)

    # À chaque tour, on repart du train set complet
    if frac < 1.0:
        X_train, _, y_train, _, w_train, _ = train_test_split(
            X_train_full,
            y_train_full,
            w_train_full,
            train_size=frac,
            random_state=42,
            stratify=y_train_full
        )
    else:
        X_train = X_train_full
        y_train = y_train_full
        w_train = w_train_full

    print("Training size:", len(X_train), flush=True)

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

    train_scores = bdt.predict(X_train)
    test_scores = bdt.predict(X_test)

    train_auc = roc_auc_score(y_train, train_scores, sample_weight=w_train)
    test_auc = roc_auc_score(y_test, test_scores, sample_weight=w_test)

    train_aucs.append(train_auc)
    test_aucs.append(test_auc)
    train_sizes.append(len(X_train))

    print(f"Train AUC = {train_auc:.4f}", flush=True)
    print(f"Test AUC  = {test_auc:.4f}", flush=True)
    print()




os.makedirs("figures", exist_ok=True)

plt.figure(figsize=(8, 5))

plt.plot(train_sizes, train_aucs, "o-", label="Training AUC")
plt.plot(train_sizes, test_aucs, "o-", label="Test AUC")

plt.xlabel("Training set size")
plt.ylabel("AUC")
plt.title("Learning curve - Hyperparameter BDT")
plt.legend()
plt.grid()

output_path = "figures/learning_curve_hyperparameter_bdt.png"

plt.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"Figure saved to: {output_path}", flush=True)

plt.show()