import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve
)
from sklearn.preprocessing import LabelBinarizer

os.makedirs('outputs', exist_ok=True)


# ──────────────────────────────────────────────────────────────
#  Core evaluation — works for binary AND multiclass
# ──────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name="Model", class_names=None):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    n_cls   = y_proba.shape[1]

    # AUC-ROC: binary vs multiclass
    if n_cls == 2:
        auc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')

    results = {
        'model':    model_name,
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'f1_score': round(f1_score(y_test, y_pred, average='weighted'), 4),
        'auc_roc':  round(auc, 4),
    }

    labels = class_names or [str(i) for i in sorted(set(y_test))]

    print(f"\n{'='*55}")
    print(f"   {model_name} — Results")
    print(f"{'='*55}")
    print(f"   Accuracy :  {results['accuracy']}")
    print(f"   F1 Score :  {results['f1_score']}")
    print(f"   AUC-ROC  :  {results['auc_roc']}")
    print(f"\n{classification_report(y_test, y_pred, target_names=labels)}")

    return results


# ──────────────────────────────────────────────────────────────
#  Confusion Matrix
# ──────────────────────────────────────────────────────────────

def plot_confusion_matrix(model, X_test, y_test, model_name, class_names=None):
    y_pred  = model.predict(X_test)
    cm      = confusion_matrix(y_test, y_pred)
    labels  = class_names or [str(i) for i in sorted(set(y_test))]
    figsize = max(5, len(labels))

    plt.figure(figsize=(figsize, figsize - 1))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix — {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    path = f"outputs/{model_name.replace(' ','_')}_confusion_matrix.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"   Saved → {path}")


# ──────────────────────────────────────────────────────────────
#  ROC Curves — binary and multiclass (OvR)
# ──────────────────────────────────────────────────────────────

def plot_roc_curves(models_dict, X_test, y_test, is_binary=True):
    plt.figure(figsize=(9, 6))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    if is_binary:
        # ── Binary: one curve per model ───────────────────
        for (name, model), color in zip(models_dict.items(), colors):
            y_proba       = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _   = roc_curve(y_test, y_proba)
            auc           = roc_auc_score(y_test, y_proba)
            plt.plot(fpr, tpr, color=color, lw=2, label=f'{name}  (AUC={auc:.4f})')
    else:
        # ── Multiclass: macro-average curve per model ─────
        lb      = LabelBinarizer()
        y_bin   = lb.fit_transform(y_test)
        n_cls   = y_bin.shape[1]

        for (name, model), color in zip(models_dict.items(), colors):
            y_proba = model.predict_proba(X_test)

            # Compute macro-avg ROC
            all_fpr = np.linspace(0, 1, 200)
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_cls):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                mean_tpr   += np.interp(all_fpr, fpr, tpr)
            mean_tpr /= n_cls
            auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            plt.plot(all_fpr, mean_tpr, color=color, lw=2,
                     label=f'{name}  (AUC={auc:.4f}, macro-avg OvR)')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves — All Models')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('outputs/roc_curves_all_models.png', dpi=120)
    plt.close()
    print("   Saved → outputs/roc_curves_all_models.png")


# ──────────────────────────────────────────────────────────────
#  Bar Chart Comparison
# ──────────────────────────────────────────────────────────────

def plot_model_comparison(results_list):
    names    = [r['model']    for r in results_list]
    accuracy = [r['accuracy'] for r in results_list]
    f1       = [r['f1_score'] for r in results_list]
    auc      = [r['auc_roc']  for r in results_list]

    x, w = np.arange(len(names)), 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w, accuracy, w, label='Accuracy', color='#3498db')
    ax.bar(x,     f1,       w, label='F1 Score',  color='#2ecc71')
    ax.bar(x + w, auc,      w, label='AUC-ROC',   color='#e74c3c')

    min_val = min(accuracy + f1 + auc)
    ax.set_ylim(max(0, min_val - 0.05), 1.01)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.legend()
    plt.tight_layout()
    plt.savefig('outputs/model_comparison.png', dpi=120)
    plt.close()
    print("   Saved → outputs/model_comparison.png")


# ──────────────────────────────────────────────────────────────
#  Terminal Summary Table
# ──────────────────────────────────────────────────────────────

def compare_models(results_list):
    print(f"\n{'='*62}")
    print(f"   FINAL MODEL COMPARISON")
    print(f"{'='*62}")
    print(f"{'Model':<22} {'Accuracy':>10} {'F1 Score':>10} {'AUC-ROC':>10}")
    print(f"{'-'*62}")
    best = max(results_list, key=lambda r: r['auc_roc'])
    for r in results_list:
        tag = '  ← BEST' if r['model'] == best['model'] else ''
        print(f"{r['model']:<22} {r['accuracy']:>10} {r['f1_score']:>10} {r['auc_roc']:>10}{tag}")
    print(f"{'='*62}")