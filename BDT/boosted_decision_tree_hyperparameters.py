from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

class BoostedDecisionTreeHyperParameters:
    def __init__(self, n_estimators=1000,        
            max_depth=7,             
            learning_rate=0.02,
            subsample=0.8,           
            colsample_bytree=0.8,    
            min_child_weight=5,      
            tree_method='hist',      
            random_state=31415,
            early_stopping_rounds=15):
        self.model = XGBClassifier(
            n_estimators=n_estimators,        
            max_depth=max_depth,             
            learning_rate=learning_rate,
            subsample=subsample,           
            colsample_bytree=colsample_bytree,    
            min_child_weight=min_child_weight,      
            tree_method=tree_method,      
            random_state=random_state,
            early_stopping_rounds=early_stopping_rounds
        )

    def fit(self, train_data, labels, weights=None):

        X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
            train_data, labels, weights, test_size=0.2, random_state=42, stratify=labels
        )

        self.model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_val, y_val)],
            sample_weight_eval_set=[w_val],
            verbose=False 
        )
                            
    def predict(self, test_data):
        return self.model.predict_proba(test_data)[:, 1]