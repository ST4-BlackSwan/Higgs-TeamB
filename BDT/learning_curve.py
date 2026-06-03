import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from HiggsML.datasets import download_dataset

from boosted_decision_tree_hyperparameters import BoostedDecisionTreeHyperParameters


data = download_dataset("blackSwan_data")

data.load_train_set()
data_set = data.get_train_set()

train_data = data_set.drop(columns=["labels", "weights", "detailed_labels"])
labels = data_set["labels"]
weights = data_set["weights"]


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
    print(f"Training with {frac:.0%} of the training data...")

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

    model = BoostedDecisionTreeHyperParameters()

    model.fit(X_train, y_train, w_train)

    train_scores = model.predict(X_train)
    test_scores = model.predict(X_test)

    train_auc = roc_auc_score(y_train, train_scores, sample_weight=w_train)
    test_auc = roc_auc_score(y_test, test_scores, sample_weight=w_test)

    train_aucs.append(train_auc)
    test_aucs.append(test_auc)
    train_sizes.append(len(X_train))

    print(f"Train AUC = {train_auc:.4f}")
    print(f"Test AUC  = {test_auc:.4f}")
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

plt.savefig("figures/learning_curve_hyperparameter_bdt.png", dpi=300, bbox_inches="tight")
plt.show()



