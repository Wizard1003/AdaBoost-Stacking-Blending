import warnings
warnings.filterwarnings('ignore')

import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Ensemble-Models")


from src.data.preprocessor import DataPreprocessor
from src.models.base_models import get_base_models
from src.models.adaboost_model import AdaBoostModel
from src.models.stacking_model import StackingModel
from src.models.blending_model import BlendingModel
from src.evaluation.metrics import evaluate_model, compare_models
import mlflow
import mlflow.sklearn
import os

os.makedirs('outputs', exist_ok=True)

def main():
    # ── 1. Load & Preprocess Data ──────────────────────────
    print("\n📦 Loading data...")
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data('data/raw/breast_cancer.csv')
    X, y = preprocessor.preprocess(df, target_col='target')
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    all_results = []
    
    # ── 2. AdaBoost ────────────────────────────────────────
    print("\n🔶 Training AdaBoost...")
    with mlflow.start_run(run_name="AdaBoost"):
        ada = AdaBoostModel(n_estimators=100, learning_rate=0.5)
        ada.fit(X_train, y_train)
        results = evaluate_model(ada, X_test, y_test, "AdaBoost")
        all_results.append(results)
        mlflow.log_metrics({k: v for k, v in results.items() if k != 'model'})
        mlflow.sklearn.log_model(ada.model, "adaboost_model")

    # ── 3. Stacking ────────────────────────────────────────
    print("\n🔷 Training Stacking Ensemble...")
    with mlflow.start_run(run_name="Stacking"):
        base_models = get_base_models()
        stacker = StackingModel(base_models)
        stacker.fit(X_train, y_train)
        results = evaluate_model(stacker, X_test, y_test, "Stacking")
        all_results.append(results)
        mlflow.log_metrics({k: v for k, v in results.items() if k != 'model'})


    # ── 4. Blending ────────────────────────────────────────
    print("\n🟩 Training Blending Ensemble...")
    with mlflow.start_run(run_name="Blending"):
        base_models2 = get_base_models()
        blender = BlendingModel(base_models2, holdout_size=0.2)
        blender.fit(X_train, y_train)
        results = evaluate_model(blender, X_test, y_test, "Blending")
        all_results.append(results)
        mlflow.log_metrics({k: v for k, v in results.items() if k != 'model'})

    # ── 5. Compare All ─────────────────────────────────────
    compare_models(all_results)
    print("\n✅ Done! Check MLflow UI: mlflow ui")

if __name__ == "__main__":
    mlflow.set_experiment("Ensemble-Models")
    main()