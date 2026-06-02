from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from HiggsML.datasets import download_dataset

from BDT.boosted_decision_tree import BoostedDecisionTree

data = download_dataset("blackSwan_data")
data.load_train_set()
data_set = data.get_train_set()

train_data = data_set.drop(columns=["labels", "weights", "detailed_labels"])
labels = data_set["labels"]
weights = data_set["weights"]

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    train_data, labels, weights, test_size=0.3, random_state=42, stratify=labels
)

bdt = BoostedDecisionTree(X_train)
bdt.fit(X_train, y_train, w_train)

predictions = bdt.predict(X_test)
auc = roc_auc_score(y_test, predictions, sample_weight=w_test)
print("AUC =", auc)
