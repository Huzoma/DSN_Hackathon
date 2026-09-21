import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and clean the data
train_data = pd.read_csv('train.csv')
train_data['product_weight_kg'] = train_data['product_weight_kg'].fillna(train_data['product_weight_kg'].mean())
train_data['store_size'] = train_data['store_size'].fillna(train_data['store_size'].mode()[0])

# 2. Configure styling
sns.set_theme(style="whitegrid")

# 3. Create a scatter plot: Product Price vs. Total Sales
plt.figure(figsize=(10, 6))
# We use alpha=0.3 to make the dots slightly transparent, which helps us see where data clumps together
sns.scatterplot(x='product_price', y='total_sales', data=train_data, alpha=0.3, color='#2563eb')

# 4. Add labels and formatting
plt.title('Product Price vs. Total Sales', fontsize=14)
plt.xlabel('Product Price', fontsize=12)
plt.ylabel('Total Sales', fontsize=12)
plt.tight_layout()

# 5. Save the chart
plt.savefig('price_vs_sales.png')
print("--- Success: Chart saved as 'price_vs_sales.png' ---")