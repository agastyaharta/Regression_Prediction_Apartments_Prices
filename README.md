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

## Dataset Explanation:
The dataset consists of apartment records with the following features:

* unit_id – Unique (and anonymized) identifier for each apartment.
* obj_type – Type of apartment or object (categorical, anonymized).
* dim_m2 – Apartment size in square meters.
* n_rooms – Number of rooms.
* floor_no – The floor on which the apartment is located.
* floor_max – Total number of floors in the building.
* year_built – The year the building was constructed.
* dist_centre – Distance from the apartment to the city center.
* n_poi – Number of points of interest nearby.
* dist_sch – Distance to the nearest school.
* dist_clinic – Distance to the nearest clinic.
* dist_post – Distance to the nearest post office.
* dist_kind – Distance to the nearest kindergarten.
* dist_rest – Distance to the nearest restaurant.
* dist_uni – Distance to the nearest college or university.
* dist_pharma – Distance to the nearest pharmacy.
* own_type – Ownership type (categorical, anonymized).
* build_mat – Building material (categorical, anonymized).
* cond_class – Condition or quality class of the apartment (categorical, anonymized).
* has_park – Whether the apartment has a parking space (boolean).
* has_balcony – Whether the apartment has a balcony (boolean).
* has_lift – Whether the apartment building has an elevator (boolean).
* has_sec – Whether the apartment has security features (boolean).
* has_store – Whether the apartment has a storage room (boolean).
* price_z – Target variable: Apartment price (in appropriate monetary units) to be predicted – only in the training sample
* src_month – Source month (time attribute).
* loc_code – Anonymized location code of the apartment.
* market_volatility – Simulated market fluctuation affecting the apartment price.
* infrastructure_quality – Indicator of the building’s infrastructure quality, partially based on the building’s age.
* neighborhood_crime_rate – Random index simulating local crime rate.
* popularity_index – Randomly generated measure of the apartment’s attractiveness.
* green_space_ratio – Proxy variable representing the amount of nearby green space, inversely related to the distance from the city center.
* estimated_maintenance_cost – Estimated cost of maintaining the apartment, based on its size.
* global_economic_index – Simulated economic index with minor fluctuations across entries, reflecting broader market conditions.
 
## Methodology

_**further explanations of each correspondins steps are available within the notebook file**_

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
   - Results can be exported and combined with corresponding `unit_id` for submission or reporting (in the format of CSV)   

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

