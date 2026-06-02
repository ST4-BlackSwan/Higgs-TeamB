from HiggsML.datasets import download_dataset
from boosted_decision_tree import BoostedDecisionTree
from sklearn.metrics import roc_auc_score




data = download_dataset(
    "blackSwan_data"
)



data.load_train_set()
data_set = data.get_train_set()
train_data = data_set.drop(columns=["labels", "weights", "detailed_labels"])

bdt = BoostedDecisionTree(data_set)


bdt.fit(train_data, data_set["labels"], data_set["weights"])

data.load_test_set()
test_data = data.get_test_set()
predictions = bdt.predict(test_data["ztautau"].drop(columns=["weights"]))
pred_train = bdt.predict(train_data)
auc = roc_auc_score(
    data_set["labels"],
    pred_train,
    sample_weight=data_set["weights"]
)

pred_z = bdt.predict(
    test_data["ztautau"].drop(columns=["weights"])
)

pred_higgs = bdt.predict(
    test_data["htautau"].drop(columns=["weights"])
)

from xgboost import plot_importance
import matplotlib.pyplot as plt

plot_importance(
    bdt.model,
    max_num_features=15
)

print(list(train_data.columns))
for i, col in enumerate(train_data.columns):
    print(i, col)

plt.tight_layout()
plt.show()
print(auc)
print("Mean Higgs =", pred_higgs.mean())
print("Mean Z =", pred_z.mean())
print(pred_higgs[:20])
print(pred_higgs.mean())
print(predictions)