**Dataset and Evaluation Split**

| Property | Value |
|---|---|
| Applicants (Home Credit, application table) | 307,511 |
| Columns as received | 122 |
| Default rate | 8.07% |
| Columns missing for more than half of applicants | 41 |
| Training / validation / test applicants | 196,806 / 49,202 / 61,503 |
| Engineered features used by the model | 300 |
| Age range (years) | 20.5–69.1 |

Stratified 80/20 train/test split; validation is 20% of the training part. Cleaning statistics are learned from the training rows only.
