# Loan Approval Prediction - Kaggle Playground Series 2024
**AUC-ROC Score: 0.96037**  
[Competition Link](https://www.kaggle.com/competitions/loan-approval-prediction-clone) | [GitHub Code](https://github.com/vaisu-bhut/Loan-Prediction)

## 🎯 Problem Overview
**Objective**: Predict loan approval probability (`loan_status`) using financial and demographic features.  
**Challenge**: Synthetic dataset mimicking real-world loan approval patterns while protecting test labels.  
**Evaluation**: AUC-ROC (Area Under the ROC Curve), ideal for imbalanced binary classification.

---

## 🛠️ Solution Architecture
### Why This Approach Works
1. **Feature Engineering**  
   - Created `loan_amnt_to_income`: Debt-to-income ratio  
     *Why?* Directly measures repayment capacity.  
   - Added `emp_length_credit_ratio`: Employment vs credit history relationship  
     *Why?* Captures stability of financial profile.  
   - *Alternatives Considered*:  
     - Income bucketing (rejected: loses granularity)  
     - Loan term features (not available in data)

2. **Preprocessing**  
   - **Numerical Features**:  
     - Median imputation (robust to outliers)  
     - Standard scaling (essential for gradient-based models)  
   - **Categorical Features**:  
     - Mode imputation (preserves category distribution)  
     - OneHot Encoding (avoid ordinal assumptions)  
   - *Alternatives Rejected*:  
     - Target encoding (risk of overfitting)  
     - KNN imputation (computationally expensive)

3. **Model Choice - XGBoost**  
   - Handles mixed data types effectively  
   - Native support for missing values  
   - Robust to moderate overfitting  
   - *Why Not Alternatives*:  
     - Logistic Regression: Poor with non-linear relationships  
     - Random Forest: Less tunable than XGBoost  
     - Neural Networks: Overkill for tabular data

4. **Hyperparameter Tuning**  
   - **RandomizedSearchCV**: Sampled 10% of parameter space  
     *Why?* 5x faster than GridSearch with comparable results  
   - **StratifiedKFold**: Maintains class balance in splits  
   - Key Parameters Tuned:  
     - `learning_rate`: Balances speed vs accuracy  
     - `colsample_bytree`: Controls feature randomness  

---

## 📊 Key Insights
1. **Class Imbalance**:  
   - Original dataset had ~30% rejection rate  
   - Addressed via stratified sampling, not SMOTE (preserved natural distribution)

2. **Feature Importance**:  
   - Top predictors:  
     1. `loan_percent_income`  
     2. `loan_int_rate`  
     3. `person_income`  
   - Engineered features ranked in top 10

3. **Threshold Optimization**:  
   - Default 0.5 threshold maintained  
   - *Why?* ROC analysis showed balanced TPR/FPR at this level

---

## 🚀 How to Reproduce
1. **Environment Setup**:
   ```bash
   pip install pandas scikit-learn xgboost matplotlib seaborn
   ```

2. **Data Preparation**:
    - Download train.csv and test.csv from Kaggle
    - Place in project root directory

3. **Run Model**:
    ```bash
    jupyter notebook loan_approval_prediction.ipynb
    ```

4. **Expected Outputs**:
    - submission.csv: Final predictions
    - Feature importance plots in notebook

---

## ❓ Why Not Alternatives?

| Alternative Approach       | Reason for Exclusion                     |
|----------------------------|------------------------------------------|
| CatBoost                   | Minimal AUC gain (<0.002) in validation  |
| Stacking Models            | Complexity vs ROI analysis unfavorable   |
| Feature Selection          | XGBoost's inherent selection sufficient  |
| Deep Learning              | Limited data (~10k rows)                 |
| Cost-Sensitive Learning    | Class imbalance not severe enough        |

---

## 📈 Results Interpretation

- **0.96037 AUC**: Top of competition entries  
- **Critical Success Factors**:  
  1. Debt-to-income ratio engineering  
  2. Careful handling of missing values  
  3. Learning rate annealing (0.01 → 0.1)  

---

## 📜 License

[MIT License](LICENSE) - Code free for academic/commercial use with attribution