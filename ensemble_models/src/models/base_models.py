from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def get_base_models():
    """Returns dictionary of base models for stacking/blending"""
    return {
        'logistic_regression': LogisticRegression(
            max_iter=1000, random_state=42
        ),
        'svm': SVC(
            probability=True, random_state=42
        ),
        'decision_tree': DecisionTreeClassifier(
            max_depth=5, random_state=42
        ),
        'knn': KNeighborsClassifier(
            n_neighbors=5
        ),
        'xgboost': XGBClassifier(
            n_estimators=100, random_state=42,
            eval_metric='logloss', verbosity=0
        ),
        'lightgbm': LGBMClassifier(
            n_estimators=100, random_state=42, verbose=-1
        )
    }