# Regression_Prediction_Apartments_Prices
This project applies a range of **regression techniques** to predict apartment prices based on structural and categorical features of housing data - presummably within the area of Warsaw, Poland.

## Contributors 
I Putu Agastya Harta Pratama 

Warsaw, Poland  
2025 

## Project Overview

- **Goal:** Build and compare multiple regression models to estimate apartment prices with the lowest error.  
- **Datasets:**  
  - `appartments_train.csv` – training dataset (with target variable `price_z`)  
  - `appartments_test.csv` – test dataset (without target, used for final predictions)  
 
## Methodology

1. **Exploratory Data Analysis (EDA):**
   - Visualisation of missing values (`missingno`)  
   - Summary statistics  
   - Inspection of numerical and categorical feature missingness  

2. **Preprocessing:**
   - Imputation (`SimpleImputer`) for both numeric and categorical data  
   - Encoding (`OneHotEncoder`, `OrdinalEncoder`)  
   - Feature scaling (`StandardScaler`, `MinMaxScaler`)  
   - Log-transformation of target variable (`price_z`) to correct right skew  

3. **Model Training:**
   The following models were implemented and tuned:
   - Linear Regression  
   - Ridge, Lasso, ElasticNet (regularisation methods)  
   - K-Nearest Neighbours (KNN)  
   - Support Vector Regression (SVR)  
   - Random Forest Regressor  
   - XGBoost Regressor  

4. **Validation & Selection:**
   - Cross-validation using **5-fold KFold**  
   - Hyperparameter tuning via `GridSearchCV` and `RandomizedSearchCV`  
   - Metrics evaluated:
     - Root Mean Squared Error (RMSE)  
     - Mean Absolute Error (MAE)  
     - R² Score  

5. **Prediction Output:**
   - Final predictions for the test set are generated  
   - Results can be exported and combined with `unit_id` for submission or reporting  

---

## Results & Model Choice

- The target variable (`price_z`) was **log-transformed** before modelling to stabilise variance and reduce skewness. Metrics were calculated after reversing the transformation to maintain interpretability in original price units.
- Models were compared using **RMSE, MAE, and R²** on validation folds:
  - **Linear Regression**: Provided a baseline, but limited in handling multicollinearity and non-linearities.  
  - **Regularised Models (Ridge, Lasso, ElasticNet):** Improved generalisability, reduced overfitting, and handled correlated predictors better.  
  - **Tree-based Models (Random Forest, XGBoost):** Captured non-linear relationships and interactions, achieving competitive accuracy but at the cost of interpretability.  

- **Why RMSE as the final metric?**
  - RMSE penalises large deviations more heavily, which is crucial for pricing problems where underestimating or overestimating by a large margin is costly.  
  - RMSE is expressed in the same units as apartment prices, making it business-relevant.  

- **Chosen Model:**  
  The final model was selected as the one with the **lowest cross-validated RMSE**. This ensured both high predictive accuracy and robustness across different folds of the data.  
  While models like Random Forest and XGBoost performed well, the selected model struck the right balance between **accuracy, complexity, and interpretability**.  

---

## Dependencies

- `pandas`, `numpy` – data manipulation  
- `matplotlib`, `seaborn` – visualisation  
- `missingno` – missing value maps  
- `statsmodels` – statistical analysis & regression diagnostics  
- `scipy` – statistical tests  
- `scikit-learn` – preprocessing, models, evaluation, pipelines  
- `xgboost` – gradient boosting model  
