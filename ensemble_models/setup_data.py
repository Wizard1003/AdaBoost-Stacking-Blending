# Run this in Python terminal or a .py file
from sklearn.datasets import load_breast_cancer, load_iris, fetch_california_housing
import pandas as pd

# Classification Dataset (Breast Cancer)
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.to_csv('data/raw/breast_cancer.csv', index=False)
print("Breast cancer dataset saved ✅")

# Regression Dataset (California Housing)  
housing = fetch_california_housing()
df2 = pd.DataFrame(housing.data, columns=housing.feature_names)
df2['target'] = housing.target
df2.to_csv('data/raw/california_housing.csv', index=False)
print("Housing dataset saved ✅")