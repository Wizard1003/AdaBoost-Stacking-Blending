from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
import numpy as np
import pandas as pd

class StackingModel:
    def __init__(self, base_models, meta_learner=None):
        self.base_models = base_models  # dict from get_base_models()
        self.meta_learner = meta_learner or LogisticRegression(max_iter=1000)
        
        # Build sklearn StackingClassifier
        estimators = list(base_models.items())
        self.model = StackingClassifier(
            estimators=estimators,
            final_estimator=self.meta_learner,
            cv=5,                        # 5-fold cross validation
            stack_method='predict_proba',
            passthrough=False
        )
    
    def fit(self, X_train, y_train):
        print("Training stacking model with 5-fold CV...")
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def get_oof_predictions(self, X_train, y_train):
        """Get Out-of-Fold predictions for each base model"""
        oof_preds = {}
        for name, model in self.base_models.items():
            print(f"  Getting OOF predictions for: {name}")
            preds = cross_val_predict(
                model, X_train, y_train,
                cv=5, method='predict_proba'
            )
            oof_preds[name] = preds[:, 1]  # probability of class 1
        return pd.DataFrame(oof_preds)