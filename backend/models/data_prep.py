import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# 1. Load the training data
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')

# 2. Drop redundant/collinear features
if 'DAYS_BIRTH' in X_train.columns:
    X_train = X_train.drop(columns=['DAYS_BIRTH'])

# 3. Isolate continuous numerical columns for scaling
# Filtering out any column with <= 2 unique values prevents scaling one-hot encoded flags
continuous_cols = [col for col in X_train.columns if X_train[col].nunique() > 2]

# 4. Standardize numerical features
scaler = StandardScaler()
X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
joblib.dump(scaler, 'credit_scaler.pkl')

# 5. Apply SMOTE to correct class imbalance
# Ensure y_train is aligned perfectly with X_train for synthetic generation
smote = SMOTE(random_state=42)
X_train_modified, y_train_modified = smote.fit_resample(X_train, y_train)

# 6. Export the fully preprocessed datasets
X_train_modified.to_csv('X_train_modified.csv', index=False)
y_train_modified.to_csv('y_train_modified.csv', index=False)

print(f"Original shape: {X_train.shape} | New shape: {X_train_modified.shape}")


# 1. Load the raw test data
X_test = pd.read_csv('/content/drive/MyDrive/X_test.csv')
y_test = pd.read_csv('/content/drive/MyDrive/y_test.csv')

# 2. Drop the redundant collinear feature to match X_train
if 'DAYS_BIRTH' in X_test.columns:
    X_test = X_test.drop(columns=['DAYS_BIRTH'])

# 3. Identify the continuous numerical columns
# This must match the logic used on your training set
continuous_cols = [col for col in X_test.columns if X_test[col].nunique() > 2]

# 4. Load the fitted scaler and transform (DO NOT fit again)
scaler = joblib.load('/content/drive/MyDrive/credit_scaler.pkl')
X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

# 5. Export the preprocessed test datasets directly to Drive
X_test.to_csv('/content/drive/MyDrive/X_test_modified.csv', index=False)
y_test.to_csv('/content/drive/MyDrive/y_test_modified.csv', index=False)

print(f"Test features shape: {X_test.shape} | Test labels shape: {y_test.shape}")