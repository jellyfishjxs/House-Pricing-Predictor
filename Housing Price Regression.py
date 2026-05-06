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

# Categorize the data by object + graph them
categorical_list = [x for x in df.columns if df[x].dtype == 'object']

for x in categorical_list: print(x)

# Plots by room

plt.figure(figsize = (15, 5))

plt.subplot(1, 3, 1)
plt1 = df['mainroad'].value_counts().plot(kind= 'bar')
plt.title('Mainroad Histogram')
plt1.set(xlabel= 'mainroad', ylabel= 'Frequency of mainroad')

plt.subplot(1, 3, 2)
plt1 = df['guestroom'].value_counts().plot(kind= 'bar')
plt.title('Guestroom Histogram')
plt1.set(xlabel= 'guestroom', ylabel= 'Frequency of guestroom')

plt.subplot(1, 3, 3)
plt1 = df['basement'].value_counts().plot(kind= 'bar')
plt.title('Basement Histogram')
plt1.set(xlabel= 'basement', ylabel= 'Frequency of basement')

plt.show()

# Plots by facilities

plt.figure(figsize = (15, 5))

plt.subplot(1, 3, 1)
plt1 = df['hotwaterheating'].value_counts().plot(kind= 'bar')
plt.title('Hotwaterheating Histogram')
plt1.set(xlabel= 'hotwaterheating', ylabel= 'Frequency of hotwaterheating')

plt.subplot(1, 3, 2)
plt1 = df['airconditioning'].value_counts().plot(kind= 'bar')
plt.title('Airconditioning Histogram')
plt1.set(xlabel= 'airconditioning', ylabel= 'Frequency of airconditioning')

plt.subplot(1, 3, 3)
plt1 = df['prefarea'].value_counts().plot(kind= 'bar')
plt.title('Prefarea Histogram')
plt1.set(xlabel= 'prefarea', ylabel= 'Frequency of prefarea')

plt.subplot(2, 3, 3)
plt1 = df['furnishingstatus'].value_counts().plot(kind= 'bar')
plt.title('Furnishingstatus Histogram')
plt1.set(xlabel= 'furnishingstatus', ylabel= 'Frequency of furnishingstatus')

plt.show()




