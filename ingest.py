import pandas as pd

# 1. Load the CSV(Comma-Separated Values) files into DataFrames
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 2. Check the dimensions of the dataset
print("--- Training Data Shape (Rows, Columns) ---")
print(train_data.shape)

# 3. Output the exact column headers
print("\n--- Columns in Training Data ---")
print(train_data.columns.tolist())

# 4. Preview the first 3 rows to inspect the data types
print("\n--- First 3 Rows ---")
print(train_data.head(3))

# 5. Check for missing (null) values in every column
print("\n--- Missing Data Count ---")
print(train_data.isnull().sum())

# 6. Check the official data types for each column
print("\n--- Dataset Info ---")
print(train_data.info())

# 7. Impute missing numerical values (product weight) with the mean average
train_data['product_weight_kg'] = train_data['product_weight_kg'].fillna(train_data['product_weight_kg'].mean())

# 8. Impute missing categorical values (store size) with the mode (most frequent string)
mode_store_size = train_data['store_size'].mode()[0]
train_data['store_size'] = train_data['store_size'].fillna(mode_store_size)

# 9. Verify the fix
print("\n--- Missing Data Count After Fix ---")
print(train_data.isnull().sum())