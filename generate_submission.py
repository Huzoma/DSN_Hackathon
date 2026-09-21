import pandas as pd
from xgboost import XGBRegressor

# 1. Load the enriched training data
df_train = pd.read_csv('train_ready.csv')
X_train = df_train.drop(columns=['total_sales', 'id', 'product_code', 'store_code'])
y_train = df_train['total_sales']

# 2. Train the Optimized XGBoost Engine on 100% of the Data
print("--- Training Optimized XGBoost Engine on 100% of Historical Data... ---")
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=1.0,
    colsample_bytree=0.9,
    random_state=42
)
model.fit(X_train, y_train)

# 3. Load and Preprocess the Test Data
print("--- Processing Test Data... ---")
test_data = pd.read_csv('test.csv')
submission_ids = test_data['id']

# Apply Imputation
test_data['product_weight_kg'] = test_data['product_weight_kg'].fillna(test_data['product_weight_kg'].mean())
test_data['store_size'] = test_data['store_size'].fillna(test_data['store_size'].mode()[0])

# Apply Advanced Feature Engineering (price_per_kg)
test_data['price_per_kg'] = test_data['product_price'] / test_data['product_weight_kg']
test_data['price_tier'] = pd.qcut(test_data['product_price'], q=4, labels=['Low', 'Medium', 'High', 'Premium'])

# Apply One-Hot Encoding
columns_to_encode = ['store_format', 'store_location_tier', 'store_size', 'fat_content', 'price_tier', 'product_category']
X_test = pd.get_dummies(test_data, columns=columns_to_encode, drop_first=True)

# 4. ARCHITECT GUARDRAIL: Align Schemas
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# 5. Generate Final Predictions
print("--- Generating V2 Predictions... ---")
predictions = model.predict(X_test)

# 6. Format for Kaggle Submission
submission = pd.DataFrame({
    'id': submission_ids,
    'total_sales': predictions
})

submission.to_csv('final_submission.csv', index=False)
print("--- Success: V2 'final_submission.csv' generated and ready for Kaggle! ---")