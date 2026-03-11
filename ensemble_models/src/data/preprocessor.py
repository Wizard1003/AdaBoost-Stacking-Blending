import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
    
    def load_data(self, filepath):
        return pd.read_csv(filepath)
    
    def preprocess(self, df, target_col):
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Handle missing values
        X = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=X.columns
        )
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42
        )
        return X_train, X_val, X_test, y_train, y_val, y_test