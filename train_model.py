import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

# 1. Load the enriched dataset
df = pd.read_csv('train_ready.csv')

# 2. Separate features (X) and target (y)
X = df.drop(columns=['total_sales', 'id', 'product_code', 'store_code'])
y = df['total_sales']

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train XGBoost Engine
print("--- Training XGBoost Engine... ---")
xgb_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=1.0,
    colsample_bytree=0.9,
    random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

# 5. Train LightGBM Engine
print("--- Training LightGBM Engine... ---")
lgb_model = LGBMRegressor(
    n_estimators=150,
    learning_rate=0.05,
    num_leaves=15,
    random_state=42,
    verbose=-1
)
lgb_model.fit(X_train, y_train)
lgb_preds = lgb_model.predict(X_test)

# 6. ENSEMBLE BLEND: Average the predictions
print("--- Blending Predictions... ---")
ensemble_preds = (xgb_preds + lgb_preds) / 2
ensemble_rmse = root_mean_squared_error(y_test, ensemble_preds)

print(f"--- Standalone XGBoost RMSE: {root_mean_squared_error(y_test, xgb_preds):.2f} ---")
print(f"--- Standalone LightGBM RMSE: {root_mean_squared_error(y_test, lgb_preds):.2f} ---")
print(f"--- ENSEMBLE Validation RMSE: {ensemble_rmse:.2f} ---")