from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from HiggsML.datasets import download_dataset
from boosted_decision_tree_scale_pos_weight import BoostedDecisionTreeScalePosWeight
import numpy as np

data = download_dataset(
    "blackSwan_data")
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

print("--- Total weights ---")
print("signal :", np.sum(weights[labels == 0]))
print("noise :", np.sum(weights[labels == 1]))

print("--- Test weights ---")
print("signal :", np.sum(w_test[labels == 0]))
print("noise :", np.sum(w_test[labels == 1]))

bdt = BoostedDecisionTreeScalePosWeight()
bdt.fit(X_train, y_train, w_train)

predictions = bdt.predict(X_test)
auc = roc_auc_score(y_test, predictions, sample_weight=w_test)

print("AUC =", auc)

fpr, tpr, thresholds = roc_curve(
    y_test,
    predictions,
    sample_weight=w_test
)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"BDT ROC curve, AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()