import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

"""
House Pricing Regression
Predicts housing prices based on house area, bedrooms, furnishing,
and location.

Jodie Cheung

5/5/2026

"""

# Import data from the folder
df = pd.read_csv('C:/Users/night/OneDrive/Desktop/Datasets/Housing.csv')

# Clean up data
df.head() # Prints first 5 rows
df.shape # Rows : Columns
df.describe() # summary statistics
df.info() # Gets information on the type + counts
df.loc[df.duplicated()] # Looks for duplicates
df.columns # Get the column titles
df.isna().sum() # Looks for null values

# Create a histogram and boxplot of the price distribution + spread
fig, axes = plt.subplots(1, 2, figsize = (10, 5))

sns.histplot(df.price, kde=True, ax=axes[0])
axes[0].set_title('House Price Distribution Plot')

sns.boxplot(x=df.price, ax=axes[1])
axes[1].set_title('House Pricing Spread')

plt.show()

# Numerically sees if there is a skew
print(df.price.describe(percentiles = [0.25, 0.50, 0.75, 0.85, 0.9, 0, 1]))



