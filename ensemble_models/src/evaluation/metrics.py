from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    results = {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'auc_roc': roc_auc_score(y_test, y_proba)
    }
    
    print(f"\n{'='*50}")
    print(f"  {model_name} Results")
    print(f"{'='*50}")
    print(f"  Accuracy : {results['accuracy']:.4f}")
    print(f"  F1 Score : {results['f1_score']:.4f}")
    print(f"  AUC-ROC  : {results['auc_roc']:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")
    
    return results

def plot_confusion_matrix(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix — {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'outputs/{model_name}_confusion_matrix.png')
    plt.show()

def compare_models(results_list):
    """Takes list of result dicts and prints comparison table"""
    print(f"\n{'='*60}")
    print(f"  MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Accuracy':>10} {'F1':>10} {'AUC-ROC':>10}")
    print(f"{'-'*60}")
    for r in results_list:
        print(f"{r['model']:<25} {r['accuracy']:>10.4f} {r['f1_score']:>10.4f} {r['auc_roc']:>10.4f}")