import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and clean the data (our established baseline)
train_data = pd.read_csv('train.csv')
train_data['product_weight_kg'] = train_data['product_weight_kg'].fillna(train_data['product_weight_kg'].mean())
train_data['store_size'] = train_data['store_size'].fillna(train_data['store_size'].mode()[0])

# 2. Configure the styling (similar to a CSS reset)
sns.set_theme(style="whitegrid")

# 3. Create a bar chart: Store Format vs. Total Sales
plt.figure(figsize=(10, 6))
sns.barplot(x='store_format', y='total_sales', data=train_data, errorbar=None, hue='store_format', legend=False, palette='viridis')

# 4. Add labels and formatting
plt.title('Average Total Sales by Store Format', fontsize=14)
plt.xlabel('DSN Mart Store Format', fontsize=12)
plt.ylabel('Average Total Sales', fontsize=12)
plt.xticks(rotation=45) # Rotates text so labels don't overlap
plt.tight_layout()

# 5. Save the chart as an image file in your folder
plt.savefig('sales_by_format.png')
print("--- Success: Chart saved as 'sales_by_format.png' ---")