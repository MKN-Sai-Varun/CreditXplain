**Baselines and Final Model on the Test Set**

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic regression | 0.722 | 0.220 | 0.395 | 0.283 |
| Random forest | 0.701 | 0.184 | 0.394 | 0.251 |
| XGBoost | 0.748 | 0.225 | 0.428 | 0.295 |
| LightGBM | 0.747 | 0.229 | 0.424 | 0.297 |
| Final model (LightGBM, FE v2) | 0.767 | – | – | – |

Baselines: SMOTE on training data, metrics at each model's F1-optimal threshold. Final model PR-AUC 0.2549; gender excluded (cost 0.0015 AUC). Ten bootstrap-retrained models: 0.7609 ± 0.0009.
