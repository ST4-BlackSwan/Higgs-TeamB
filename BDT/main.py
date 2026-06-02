from HiggsML.datasets import download_dataset
from boosted_decision_tree import BoostedDecisionTree

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

print(predictions)