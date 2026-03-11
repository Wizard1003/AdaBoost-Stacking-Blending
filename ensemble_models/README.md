ensemble-models/
│
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned & feature-engineered data
│   └── splits/                 # Train/val/test splits
│
├── notebooks/
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis
│   ├── 02_adaboost.ipynb       # AdaBoost experiments
│   ├── 03_stacking.ipynb       # Stacking experiments
│   └── 04_blending.ipynb       # Blending experiments
│
├── src/
│   ├── data/
│   │   ├── loader.py           # Dataset loading utilities
│   │   └── preprocessor.py     # Scaling, encoding, imputation
│   │
│   ├── models/
│   │   ├── base_models.py      # LR, SVM, DT, KNN definitions
│   │   ├── adaboost_model.py   # Custom AdaBoost + sklearn wrapper
│   │   ├── stacking_model.py   # StackingClassifier logic
│   │   └── blending_model.py   # Blending pipeline logic
│   │
│   ├── ensemble/
│   │   ├── meta_learner.py     # Meta-model (LogReg / XGB)
│   │   ├── cross_val.py        # OOF (Out-of-Fold) predictions
│   │   └── weight_optimizer.py # Optimizing blend weights
│   │
│   ├── evaluation/
│   │   ├── metrics.py          # Accuracy, F1, AUC-ROC, etc.
│   │   └── visualizer.py       # Plots: ROC, confusion matrix
│   │
│   └── utils/
│       ├── logger.py           # Logging setup
│       └── config.py           # Config loader
│
├── configs/
│   ├── adaboost.yaml           # n_estimators, learning_rate
│   ├── stacking.yaml           # Base model list, meta-learner
│   └── blending.yaml           # Holdout ratio, base models
│
├── tests/
│   ├── test_adaboost.py
│   ├── test_stacking.py
│   └── test_blending.py
│
├── mlruns/                     # MLflow experiment logs
├── requirements.txt
├── setup.py
└── README.md










```

---

## Data Flow Architecture
```
Raw Data
   │
   ▼
Preprocessing (scaling, encoding, imputation)
   │
   ├──────────────────────────────────┐
   │                                  │
   ▼                                  ▼
[ADABOOST]                    [STACKING / BLENDING]
Sequential weak learners       Level-0 Base Models:
   • Stump 1 (low weight)        • Logistic Regression
   • Stump 2 (higher weight)     • SVM
   • Stump 3 ...                 • Decision Tree
   │                             • KNN / XGBoost
   ▼                             │
Weighted Majority Vote           ▼
   │                       OOF Predictions (Stacking)
   │                       OR Holdout Predictions (Blending)
   │                             │
   │                             ▼
   │                       Level-1 Meta-Learner
   │                       (Logistic Regression / XGBoost)
   │                             │
   └──────────────┬──────────────┘
                  ▼
           Final Predictions
                  │
                  ▼
           Evaluation Report
     (Accuracy, AUC, F1, Confusion Matrix)




# in bash 
# Step 1 — Go into the correct folder
cd "C:\Ruchi\Mini Project\aai\ensemble_models"

# Step 2 — Activate venv
venv\Scripts\activate

source venv/Scripts/activate

# Step 3 — Replace your files with the downloaded ones
# (copy each downloaded .py file into the correct src/ subfolder)

# Step 4 — Run the model
python main.py

# Step 5 — Run tests
pytest tests/ -v

1. Run Tests — confirm everything works:
pytest tests/ -v
# Expected: 7 passed in ~30s ✅
2. View MLflow Dashboard:
mlflow ui
# Open browser → http://127.0.0.1:5000
3. Open Jupyter to experiment:
jupyter lab
# Try changing n_estimators, learning_rate, base models