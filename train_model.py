import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error

# 1. Load the enriched data
df = pd.read_csv('train_ready.csv')

# 2. Separate features (X) and target variable (y)
X = df.drop(columns=['total_sales', 'id', 'product_code', 'store_code'])
y = df['total_sales']

# 3. Split the data for validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Define the XGBoost Engine and the Tuning Grid
print("--- Initializing Hyperparameter Tuning (This may take a minute)... ---")
xgb = XGBRegressor(random_state=42)

# These are the engine settings we are going to test
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# 5. Run the Search to find the mathematical sweet spot
random_search = RandomizedSearchCV(
    estimator=xgb, 
    param_distributions=param_grid, 
    n_iter=10, 
    scoring='neg_root_mean_squared_error', 
    cv=3, 
    verbose=1, 
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

# 6. Evaluate the Best Engine
best_model = random_search.best_estimator_
predictions = best_model.predict(X_test)
rmse = root_mean_squared_error(y_test, predictions)

print(f"\n--- Best Parameters Found: {random_search.best_params_} ---")
print(f"--- Optimized XGBoost Validation RMSE: {rmse:.2f} ---")