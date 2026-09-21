import pandas as pd

# 1. Load the raw data
train_data = pd.read_csv('train.csv')

# 2. Re-apply our established /audit fixes (Imputation)
train_data['product_weight_kg'] = train_data['product_weight_kg'].fillna(train_data['product_weight_kg'].mean())
train_data['store_size'] = train_data['store_size'].fillna(train_data['store_size'].mode()[0])

# 3. FEATURE ENGINEERING: Price Tiers (Binning)
# pd.qcut divides the numerical prices into 4 mathematically equal distribution buckets
train_data['price_tier'] = pd.qcut(train_data['product_price'], q=4, labels=['Low', 'Medium', 'High', 'Premium'])

# 4. FEATURE ENGINEERING: Text to Numbers (One-Hot Encoding)
# pd.get_dummies converts categorical text columns into True/False (1/0) binary columns
columns_to_encode = ['store_format', 'store_location_tier', 'store_size', 'fat_content', 'price_tier']
train_engineered = pd.get_dummies(train_data, columns=columns_to_encode, drop_first=True)

# 5. Output the results
print("--- New Dataset Shape (Rows, Columns) ---")
print(train_engineered.shape)

print("\n--- Newly Engineered Columns ---")
# Filtering to just show a few of our new binary columns to keep the terminal clean
new_cols = [col for col in train_engineered.columns if 'store_format' in col or 'price_tier' in col]
print(new_cols)

# 6. Save the final pipeline output
train_engineered.to_csv('train_ready.csv', index=False)
print("\n--- Success: Model-ready data saved to 'train_ready.csv' ---")