import numpy as np
import matplotlib.pyplot as plt

from boosted_decision_tree_hyperparameters import BoostedDecisionTreeHyperParameters
from HiggsML.datasets import download_dataset
from sklearn.model_selection import train_test_split

from boosted_decision_tree import BoostedDecisionTree

data = download_dataset("blackSwan_data")
data.load_train_set()
data_set = data.get_train_set()

train_data = data_set.drop(columns=["labels", "weights", "detailed_labels"])
labels = data_set["labels"]
weights = data_set["weights"]

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    train_data, labels, weights, test_size=0.3, random_state=42, stratify=labels
)

thresholds = np.linspace(0, 1, 200)

significances = []
S_values = []
B_values = []


bdt = BoostedDecisionTreeHyperParameters()
bdt.fit(X_train, y_train, w_train)
predictions = bdt.predict(X_test)
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
plt.plot(thresholds, significances, label="Poisson significance")
plt.axvline(
    best_threshold, linestyle="--", label=f"Best threshold = {best_threshold:.3f}"
)
plt.xlabel("BDT score threshold")
plt.ylabel("Significance Z")
plt.title("Poisson significance curve")
plt.legend()
plt.grid()
plt.show()
