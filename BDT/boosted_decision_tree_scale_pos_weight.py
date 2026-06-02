import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

class BoostedDecisionTreeScalePosWeight:
    def __init__(self,
            n_estimators=1500,       
            max_depth=6,             
            learning_rate=0.02,
            subsample=0.8,           
            colsample_bytree=0.8,    
            min_child_weight=6,      
            gamma=0.15,
            tree_method='hist',      
            random_state=31415,
            early_stopping_rounds=25):
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
        num_background = np.sum(labels == 0)
        num_signal = np.sum(labels == 1)
        ratio = num_background / num_signal
        
        self.model.set_params(scale_pos_weight=ratio)
        
        print(f"[INFO] scale_pos_weight configuré à {ratio:.2f}")

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