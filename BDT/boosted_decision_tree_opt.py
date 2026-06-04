import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

class BoostedDecisionTree:
    def __init__(self,
                 n_estimators=1500,
                 max_depth=6,
                 learning_rate=0.02,
                 subsample=0.8,
                 colsample_bytree=0.8,
                 min_child_weight=6,
                 gamma=0,
                 tree_method='hist',
                 random_state=31415,
                 early_stopping_rounds=25,
                 scale_factor=1.0):

        self.scale_factor = scale_factor

        # Instancing model
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            tree_method=tree_method,
            random_state=random_state,
            early_stopping_rounds=early_stopping_rounds
        )

    def fit(self, train_data, labels, weights=None):
        # Counting background and signal events
        sum_w_background = np.sum(weights[labels == 0])
        sum_w_signal = np.sum(weights[labels == 1])

        # Dynamicly computing the ratio
        ratio = (sum_w_background / sum_w_signal) * self.scale_factor
        self.model.set_params(scale_pos_weight=ratio)
        print(f"[INFO] scale_pos_weight configuré à {ratio:.2f}")

        # Getting training and validation data
        X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
            train_data, labels, weights, test_size=0.2, random_state=42, stratify=labels
        )

        # Training model
        self.model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_val, y_val)],
            sample_weight_eval_set=[w_val],
            verbose=False
        )

    def predict(self, test_data):
        return self.model.predict_proba(test_data)[:, 1]