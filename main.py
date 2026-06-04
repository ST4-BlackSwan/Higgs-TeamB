import numpy as np

from HiggsML.datasets import download_dataset
from sklearn.model_selection import train_test_split

from boosted_decision_tree_scale_pos_weight import BoostedDecisionTreeScalePosWeight


# Getting training and testing data
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

# Here we use the hyperparameters found with the bayesian method.
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

# Training the model
bdt.fit(X_train, y_train, w_train)

# Using the model
predictions = bdt.predict(X_test)

print(predictions)