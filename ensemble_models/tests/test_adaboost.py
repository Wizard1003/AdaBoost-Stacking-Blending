import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.adaboost_model import AdaBoostModel
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def test_adaboost_trains():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    model = AdaBoostModel(n_estimators=10)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)  # basic sanity check
    print("AdaBoost test passed ✅")