import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# 1. Load the enriched training data
df_train = pd.read_csv('train_ready.csv')
X_train = df_train.drop(columns=['total_sales', 'id', 'product_code', 'store_code'])
y_train = df_train['total_sales']

# 2. Train XGBoost Engine on 100% of Data
print("--- Training XGBoost Engine on 100% of Historical Data... ---")
xgb_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=1.0,
    colsample_bytree=0.9,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# 3. Train LightGBM Engine on 100% of Data
print("--- Training LightGBM Engine on 100% of Historical Data... ---")
lgb_model = LGBMRegressor(
    n_estimators=150,
    learning_rate=0.05,
    num_leaves=15,
    random_state=42,
    verbose=-1
)
lgb_model.fit(X_train, y_train)

# 4. Load and Preprocess Test Data
print("--- Processing Test Data... ---")
test_data = pd.read_csv('test.csv')
submission_ids = test_data['id']

test_data['product_weight_kg'] = test_data['product_weight_kg'].fillna(test_data['product_weight_kg'].mean())
test_data['store_size'] = test_data['store_size'].fillna(test_data['store_size'].mode()[0])
test_data['price_per_kg'] = test_data['product_price'] / test_data['product_weight_kg']
test_data['price_tier'] = pd.qcut(test_data['product_price'], q=4, labels=['Low', 'Medium', 'High', 'Premium'])

columns_to_encode = ['store_format', 'store_location_tier', 'store_size', 'fat_content', 'price_tier', 'product_category']
X_test = pd.get_dummies(test_data, columns=columns_to_encode, drop_first=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# 5. Generate and Blend Predictions
print("--- Generating Ensemble Predictions... ---")
xgb_preds = xgb_model.predict(X_test)
lgb_preds = lgb_model.predict(X_test)
ensemble_preds = (xgb_preds + lgb_preds) / 2

# 6. Export to CSV
submission = pd.DataFrame({
    'id': submission_ids,
    'total_sales': ensemble_preds
})
submission.to_csv('final_submission.csv', index=False)
print("--- Success: V3 Ensemble 'final_submission.csv' generated! ---")