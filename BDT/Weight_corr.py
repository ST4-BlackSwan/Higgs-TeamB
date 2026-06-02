from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

class BoostedDecisionTree2:
    def __init__(self):
        self.model = XGBClassifier()
        self.scaler = StandardScaler()

    def fit(self, train_data, labels, weights=None):
        self.scaler.fit_transform(train_data)
        X_train_data = self.scaler.transform(train_data)
        self.model.fit(X_train_data, labels, sample_weight=weights)

    def predict(self, test_data):
        test_data = self.scaler.transform(test_data)
        return self.model.predict_proba(test_data)[:, 1]