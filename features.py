import pandas as pd

# 1. Load raw training data
df = pd.read_csv('train.csv')

# 2. Handle Missing Values (Imputation)
df['product_weight_kg'] = df['product_weight_kg'].fillna(df['product_weight_kg'].mean())
df['store_size'] = df['store_size'].fillna(df['store_size'].mode()[0])

# 3. ADVANCED FEATURE ENGINEERING (New Addition)
# Calculate the exact price per kilogram to give the model a new relational metric
df['price_per_kg'] = df['product_price'] / df['product_weight_kg']

# Bin the raw prices into categorical tiers
df['price_tier'] = pd.qcut(df['product_price'], q=4, labels=['Low', 'Medium', 'High', 'Premium'])

# 4. One-Hot Encoding
columns_to_encode = ['store_format', 'store_location_tier', 'store_size', 'fat_content', 'price_tier', 'product_category']
df = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)

# 5. Export the enriched matrix for training
df.to_csv('train_ready.csv', index=False)
print("--- V2 Feature Engineering Complete: 'train_ready.csv' updated with Advanced Features ---")