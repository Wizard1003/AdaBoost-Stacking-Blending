from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class AdaBoostModel:
    def __init__(self, n_estimators=100, learning_rate=1.0, max_depth=1):
        self.model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=max_depth),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm='SAMME',
            random_state=42
        )
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def get_estimator_errors(self):
        """Returns error at each boosting round"""
        return self.model.estimator_errors_
    
    def get_estimator_weights(self):
        """Returns weight of each weak learner"""
        return self.model.estimator_weights_