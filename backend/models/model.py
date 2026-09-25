import pandas as pd
import lightgbm as lgb

# 1. Load the modified datasets from Google Drive
X_train = pd.read_csv('/content/drive/MyDrive/X_train_modified.csv')
y_train = pd.read_csv('/content/drive/MyDrive/y_train_modified.csv')
X_test = pd.read_csv('/content/drive/MyDrive/X_test_modified.csv')
y_test = pd.read_csv('/content/drive/MyDrive/y_test_modified.csv')

# Ensure labels are 1D arrays (LightGBM expects this format)
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# 2. Define the LightGBM model with aggressive anti-overfitting parameters
lgb_model = lgb.LGBMClassifier(
    n_estimators=1500,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=35,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

# 3. Train the model using early stopping
print("Starting LightGBM model training...")
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='auc',
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50)
    ]
)
print(f"Training complete. Best iteration: {lgb_model.best_iteration_}")

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Generate hard predictions and prediction probabilities
y_pred = lgb_model.predict(X_test)
y_proba = lgb_model.predict_proba(X_test)[:, 1]

# 2. Calculate core evaluation metrics
auc = roc_auc_score(y_test, y_proba)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n--- Test Data Evaluation Metrics ---")
print(f"ROC-AUC Score: {auc:.4f}")
print(f"Precision:     {precision:.4f}")
print(f"Recall:        {recall:.4f}")
print(f"F1-Score:      {f1:.4f}")

# 3. Generate the full classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 4. Visualize the Confusion Matrix for your project presentation
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Approved (0)', 'Default (1)'],
            yticklabels=['Approved (0)', 'Default (1)'])
plt.ylabel('Actual Outcome')
plt.xlabel('Predicted Outcome')
plt.title('LightGBM Confusion Matrix')
plt.show()

# Evaluate a new threshold (e.g., 0.08) to balance precision and recall
custom_threshold = 0.08
y_pred_adjusted = (y_proba >= custom_threshold).astype(int)

# Recalculate metrics based on the new threshold
print(classification_report(y_test, y_pred_adjusted))