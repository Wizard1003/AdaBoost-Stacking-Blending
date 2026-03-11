import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class BlendingModel:
    def __init__(self, base_models, meta_learner=None, holdout_size=0.2):
        self.base_models = base_models
        self.meta_learner = meta_learner or LogisticRegression(max_iter=1000)
        self.holdout_size = holdout_size
    
    def fit(self, X_train, y_train):
        # Split into train and holdout sets
        X_tr, X_hold, y_tr, y_hold = train_test_split(
            X_train, y_train,
            test_size=self.holdout_size,
            random_state=42
        )
        
        print(f"Train: {X_tr.shape}, Holdout: {X_hold.shape}")
        
        # Train base models on train split
        holdout_preds = []
        for name, model in self.base_models.items():
            print(f"  Training base model: {name}")
            model.fit(X_tr, y_tr)
            preds = model.predict_proba(X_hold)[:, 1]
            holdout_preds.append(preds)
        
        # Stack holdout predictions as new features
        meta_features = np.column_stack(holdout_preds)
        
        # Train meta-learner on holdout predictions
        print("Training meta-learner on holdout predictions...")
        self.meta_learner.fit(meta_features, y_hold)
        
        # Retrain base models on full training data
        print("Retraining base models on full data...")
        for name, model in self.base_models.items():
            model.fit(X_train, y_train)
        
        return self
    
    def predict(self, X_test):
        test_preds = []
        for name, model in self.base_models.items():
            preds = model.predict_proba(X_test)[:, 1]
            test_preds.append(preds)
        
        meta_features = np.column_stack(test_preds)
        return self.meta_learner.predict(meta_features)
    
    def predict_proba(self, X_test):
        test_preds = []
        for name, model in self.base_models.items():
            preds = model.predict_proba(X_test)[:, 1]
            test_preds.append(preds)
        meta_features = np.column_stack(test_preds)
        return self.meta_learner.predict_proba(meta_features)