"""
╔══════════════════════════════════════════════════════════════╗
║   AdaBoost + Stacking + Blending — Multi-Dataset Runner      ║
║   Run:  python main.py                    (interactive menu) ║
║   Run:  python main.py --dataset wine     (skip menu)        ║
╚══════════════════════════════════════════════════════════════╝
"""

import warnings, os, sys, argparse
warnings.filterwarnings('ignore')
os.makedirs('outputs', exist_ok=True)
os.makedirs('data/raw', exist_ok=True)

import pandas as pd
import numpy as np
import mlflow

from sklearn.datasets import (
    load_breast_cancer, load_wine, load_iris, load_digits
)

from src.data.preprocessor     import DataPreprocessor
from src.models.base_models     import get_base_models
from src.models.adaboost_model  import AdaBoostModel
from src.models.stacking_model  import StackingModel
from src.models.blending_model  import BlendingModel
from src.evaluation.metrics     import (
    evaluate_model, plot_confusion_matrix,
    plot_roc_curves, plot_model_comparison, compare_models
)


# ══════════════════════════════════════════════════════════════
#   DATASET LOADERS
# ══════════════════════════════════════════════════════════════

def load_wine_dataset():
    data = load_wine()
    df   = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df.to_csv('data/raw/wine.csv', index=False)
    return df, {
        'name':        'Wine',
        'target_col':  'target',
        'class_names': [str(n) for n in data.target_names],
        'binary':      False,
        'description': '3-class wine cultivar classification — 178 samples, 13 features'
    }

def load_iris_dataset():
    data = load_iris()
    df   = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df.to_csv('data/raw/iris.csv', index=False)
    return df, {
        'name':        'Iris',
        'target_col':  'target',
        'class_names': [str(n) for n in data.target_names],
        'binary':      False,
        'description': '3-class flower species classification — 150 samples, 4 features'
    }

def load_breast_cancer_dataset():
    data = load_breast_cancer()
    df   = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df.to_csv('data/raw/breast_cancer.csv', index=False)
    return df, {
        'name':        'Breast Cancer',
        'target_col':  'target',
        'class_names': ['Malignant', 'Benign'],
        'binary':      True,
        'description': '2-class cancer classification — 569 samples, 30 features'
    }

def load_digits_dataset():
    data = load_digits()
    df   = pd.DataFrame(data.data, columns=[f'pixel_{i}' for i in range(data.data.shape[1])])
    df['target'] = data.target
    df.to_csv('data/raw/digits.csv', index=False)
    return df, {
        'name':        'Digits',
        'target_col':  'target',
        'class_names': [str(i) for i in range(10)],
        'binary':      False,
        'description': '10-class handwritten digit recognition — 1797 samples, 64 features'
    }

def load_custom_csv_dataset(filepath, target_col, class_names=None):
    df       = pd.read_csv(filepath)
    n_cls    = df[target_col].nunique()
    names    = class_names or [str(c) for c in sorted(df[target_col].unique())]
    return df, {
        'name':        os.path.basename(filepath).replace('.csv','').replace('_',' ').title(),
        'target_col':  target_col,
        'class_names': names,
        'binary':      n_cls == 2,
        'description': f'Custom — {len(df)} samples, {len(df.columns)-1} features, {n_cls} classes'
    }


# ══════════════════════════════════════════════════════════════
#   DATASET REGISTRY
#   ➕ To add your own dataset:
#      1. Write a loader function above (see examples)
#      2. Add an entry here with key, loader, label
# ══════════════════════════════════════════════════════════════

DATASETS = {
    '1': {
        'key':    'wine',
        'loader':  load_wine_dataset,
        'label':  'Wine           — 3 classes  | 178 samples  | 13 features'
    },
    '2': {
        'key':    'iris',
        'loader':  load_iris_dataset,
        'label':  'Iris           — 3 classes  | 150 samples  |  4 features'
    },
    '3': {
        'key':    'breast_cancer',
        'loader':  load_breast_cancer_dataset,
        'label':  'Breast Cancer  — 2 classes  | 569 samples  | 30 features'
    },
    '4': {
        'key':    'digits',
        'loader':  load_digits_dataset,
        'label':  'Digits         — 10 classes | 1797 samples | 64 features'
    },
    '5': {
        'key':    'custom',
        'loader':  None,
        'label':  'Custom CSV     — your own dataset from data/raw/'
    },
    # ── ADD MORE DATASETS HERE ─────────────────────────────
    # '6': {
    #     'key':   'my_dataset',
    #     'loader': load_my_dataset,
    #     'label': 'My Dataset — X classes | Y samples | Z features'
    # },
}


# ══════════════════════════════════════════════════════════════
#   INTERACTIVE MENU
# ══════════════════════════════════════════════════════════════

def show_menu():
    print("\n" + "═"*62)
    print("   📂  SELECT A DATASET TO RUN")
    print("═"*62)
    for k, v in DATASETS.items():
        print(f"   [{k}]  {v['label']}")
    print("─"*62)

def get_user_choice():
    show_menu()
    while True:
        try:
            choice = input("\n   Enter number: ").strip()
            if choice in DATASETS:
                return choice
            print(f"   ❌  Enter a number between 1 and {len(DATASETS)}.")
        except (KeyboardInterrupt, EOFError):
            print("\n\n   Exiting.")
            sys.exit(0)

def get_custom_csv_input():
    print("\n   📂  Custom CSV Setup")
    print("   " + "─"*40)

    csv_files = [f for f in os.listdir('data/raw') if f.endswith('.csv')]
    if csv_files:
        print("   Files available in data/raw/:")
        for f in csv_files:
            print(f"     • {f}")

    filename   = input("\n   Filename (from data/raw/): ").strip()
    filepath   = filename if os.path.isabs(filename) else os.path.join('data/raw', filename)

    if not os.path.exists(filepath):
        print(f"   ❌  File not found: {filepath}")
        sys.exit(1)

    df_preview = pd.read_csv(filepath, nrows=3)
    print(f"\n   Columns: {list(df_preview.columns)}")
    target_col = input("   Target column name: ").strip()

    if target_col not in df_preview.columns:
        print(f"   ❌  Column '{target_col}' not found.")
        sys.exit(1)

    return filepath, target_col


# ══════════════════════════════════════════════════════════════
#   MAIN RUNNER
# ══════════════════════════════════════════════════════════════

def main(dataset_key=None):
    print("\n" + "═"*62)
    print("   ENSEMBLE LEARNING — AdaBoost | Stacking | Blending")
    print("═"*62)

    # ── 1. Dataset Selection ──────────────────────────────
    if dataset_key:
        matched = next((k for k, v in DATASETS.items() if v['key'] == dataset_key), None)
        choice  = matched if matched else get_user_choice()
    else:
        choice = get_user_choice()

    selected = DATASETS[choice]

    # ── 2. Load Dataset ───────────────────────────────────
    if selected['key'] == 'custom':
        filepath, target_col = get_custom_csv_input()
        df, meta = load_custom_csv_dataset(filepath, target_col)
    else:
        df, meta = selected['loader']()

    print(f"\n   ✅  Dataset     : {meta['name']}")
    print(f"   📋  Info        : {meta['description']}")
    print(f"   🏷️   Classes     : {meta['class_names']}")

    vc = df[meta['target_col']].value_counts().sort_index()
    for idx, count in vc.items():
        lbl = meta['class_names'][int(idx)] if int(idx) < len(meta['class_names']) else str(idx)
        print(f"                  {lbl:>15} = {count}")

    # ── 3. Preprocess & Split ─────────────────────────────
    preprocessor = DataPreprocessor()
    X, y         = preprocessor.preprocess(df, target_col=meta['target_col'])
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)

    print(f"\n   Train : {X_train.shape[0]} samples")
    print(f"   Val   : {X_val.shape[0]}  samples")
    print(f"   Test  : {X_test.shape[0]}  samples")

    class_names    = meta['class_names']
    is_binary      = meta['binary']
    all_results    = []
    trained_models = {}

    # MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(f"Ensemble-{meta['name'].replace(' ','-')}")

    # ══════════════════════════════════════════════════════
    #   STEP A — AdaBoost
    # ══════════════════════════════════════════════════════
    print("\n" + "─"*62)
    print("🔶  TRAINING  →  AdaBoost")
    print("─"*62)
    with mlflow.start_run(run_name="AdaBoost"):
        ada = AdaBoostModel(n_estimators=100, learning_rate=0.5, max_depth=1)
        ada.fit(X_train, y_train)
        print("   Training complete ✅")
        r = evaluate_model(ada, X_test, y_test, "AdaBoost", class_names)
        mlflow.log_metrics({k: v for k, v in r.items() if k != 'model'})
        mlflow.log_param("dataset", meta['name'])
    all_results.append(r)
    trained_models['AdaBoost'] = ada
    plot_confusion_matrix(ada, X_test, y_test, "AdaBoost", class_names)

    # ══════════════════════════════════════════════════════
    #   STEP B — Stacking
    # ══════════════════════════════════════════════════════
    print("\n" + "─"*62)
    print("🔷  TRAINING  →  Stacking Ensemble")
    print("─"*62)
    with mlflow.start_run(run_name="Stacking"):
        stacker = StackingModel(get_base_models())
        stacker.fit(X_train, y_train)
        print("   Training complete ✅")
        r = evaluate_model(stacker, X_test, y_test, "Stacking", class_names)
        mlflow.log_metrics({k: v for k, v in r.items() if k != 'model'})
        mlflow.log_param("dataset", meta['name'])
    all_results.append(r)
    trained_models['Stacking'] = stacker
    plot_confusion_matrix(stacker, X_test, y_test, "Stacking", class_names)

    # ══════════════════════════════════════════════════════
    #   STEP C — Blending
    # ══════════════════════════════════════════════════════
    print("\n" + "─"*62)
    print("🟩  TRAINING  →  Blending Ensemble")
    print("─"*62)
    with mlflow.start_run(run_name="Blending"):
        blender = BlendingModel(get_base_models(), holdout_size=0.2)
        blender.fit(X_train, y_train)
        print("   Training complete ✅")
        r = evaluate_model(blender, X_test, y_test, "Blending", class_names)
        mlflow.log_metrics({k: v for k, v in r.items() if k != 'model'})
        mlflow.log_param("dataset", meta['name'])
    all_results.append(r)
    trained_models['Blending'] = blender
    plot_confusion_matrix(blender, X_test, y_test, "Blending", class_names)

    # ══════════════════════════════════════════════════════
    #   STEP D — Charts & Summary
    # ══════════════════════════════════════════════════════
    print("\n" + "─"*62)
    print("📊  GENERATING CHARTS...")
    print("─"*62)
    plot_roc_curves(trained_models, X_test, y_test, is_binary)
    plot_model_comparison(all_results)
    compare_models(all_results)

    pd.DataFrame(all_results).to_csv('outputs/model_results.csv', index=False)

    print("\n" + "═"*62)
    print("   ✅  ALL DONE!")
    print("─"*62)
    print(f"   Dataset : {meta['name']}")
    print("   outputs/ → confusion matrices, ROC, comparison, CSV")
    print("   MLflow   → mlflow ui → http://127.0.0.1:5000")
    print("═"*62 + "\n")


# ══════════════════════════════════════════════════════════════
#   USAGE:
#     python main.py                      → interactive menu
#     python main.py --dataset wine
#     python main.py --dataset iris
#     python main.py --dataset breast_cancer
#     python main.py --dataset digits
#     python main.py --dataset custom     → prompts for CSV
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', type=str, default=None,
        choices=['wine','iris','breast_cancer','digits','custom'],
        help='Skip menu and use this dataset directly'
    )
    args = parser.parse_args()
    main(dataset_key=args.dataset)