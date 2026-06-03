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

np.savetxt("predictions.csv", predictions, delimiter=",", fmt="%d")

thresholds = np.linspace(0, 1, 200)

significances = []
S_values = []
B_values = []

for threshold in thresholds:
    selected = predictions > threshold

    S = np.sum(w_test[(y_test == 1) & selected])
    B = np.sum(w_test[(y_test == 0) & selected])

    if S > 0 and B > 0:
        Z = np.sqrt(2 * ((S + B) * np.log(1 + S / B) - S))
    else:
        Z = 0

    significances.append(Z)
    S_values.append(S)
    B_values.append(B)

significances = np.array(significances)

best_index = np.argmax(significances)
best_threshold = thresholds[best_index]
best_significance = significances[best_index]

print("Best threshold:", best_threshold)
print("Best Poisson significance:", best_significance)



plt.figure(figsize=(8, 5))
plt.plot(thresholds, significances, label="Tuned BDT Poisson significance")
plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Best threshold = {best_threshold:.3f}"
)

plt.xlabel("BDT score threshold")
plt.ylabel("Significance Z")
plt.title("Poisson significance curve - Tuned BDT")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()




plt.figure(figsize=(8, 5))

plt.hist(
    predictions[y_test == 1],
    bins=50,
    range=(0, 1),
    weights=w_test[y_test == 1],
    density=True,
    alpha=0.5,
    label="Signal S"
)

plt.hist(
    predictions[y_test == 0],
    bins=50,
    range=(0, 1),
    weights=w_test[y_test == 0],
    density=True,
    alpha=0.5,
    label="Background B"
)

plt.xlabel("BDT score")
plt.ylabel("Normalized weighted density")
plt.title("Normalized BDT score distribution for Signal and Background")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()