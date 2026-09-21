import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 1. Load the fully engineered dataset
df = pd.read_csv('train_ready.csv')

# 2. Separate Features (X) from the Target (y)
# We drop arbitrary string IDs because they hold no mathematical predictive value
X = df.drop(columns=['total_sales', 'id', 'product_code', 'store_code'])
y = df['total_sales']

# 3. Split the data (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train the Model
print("--- Training the Random Forest Engine... ---")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Generate Predictions on the 20% test set
predictions = model.predict(X_test)

# 6. Evaluate the Model using Kaggle's RMSE metric
rmse = mean_squared_error(y_test, predictions) ** 0.5

print(f"--- Model Validation RMSE: {rmse:.2f} ---")