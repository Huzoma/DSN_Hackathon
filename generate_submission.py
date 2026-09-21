import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# 1. Load the training data
df_train = pd.read_csv('train_ready.csv')
X_train = df_train.drop(columns=['total_sales', 'id', 'product_code', 'store_code'])
y_train = df_train['total_sales']

# 2. Train the Final Model on 100% of the Data
# We no longer need to hold back 20% for a mock exam. We want the model as smart as possible.
print("--- Training Final Engine on 100% of Historical Data... ---")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. Load and Preprocess the Test Data
print("--- Processing Test Data... ---")
test_data = pd.read_csv('test.csv')

# We isolate the 'id' column because Kaggle requires it for the final upload
submission_ids = test_data['id']

# Apply Imputation (Patching holes)
test_data['product_weight_kg'] = test_data['product_weight_kg'].fillna(test_data['product_weight_kg'].mean())
test_data['store_size'] = test_data['store_size'].fillna(test_data['store_size'].mode()[0])

# Apply Feature Engineering (Price Tiers & One-Hot Encoding)
test_data['price_tier'] = pd.qcut(test_data['product_price'], q=4, labels=['Low', 'Medium', 'High', 'Premium'])
columns_to_encode = ['store_format', 'store_location_tier', 'store_size', 'fat_content', 'price_tier', 'product_category']
X_test = pd.get_dummies(test_data, columns=columns_to_encode, drop_first=True)

# 4. ARCHITECT GUARDRAIL: Align Schemas
# This forces the test columns to perfectly match the training columns. 
# If the test set lacks a specific category, this fills the missing column with 0 (False) to prevent crashes.
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# 5. Generate Final Predictions
print("--- Generating Predictions... ---")
predictions = model.predict(X_test)

# 6. Format for Kaggle Submission
submission = pd.DataFrame({
    'id': submission_ids,
    'total_sales': predictions
})

submission.to_csv('final_submission.csv', index=False)
print("--- Success: 'final_submission.csv' generated and ready for Kaggle! ---")