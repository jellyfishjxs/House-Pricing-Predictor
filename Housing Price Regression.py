import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

# Categorize the categorical data by object + graph them
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

plt.show()

plt.figure(figsize = (15, 5))

plt.subplot(1, 3, 1)
plt1 = df['furnishingstatus'].value_counts().plot(kind= 'bar')
plt.title('Furnishingstatus Histogram')
plt1.set(xlabel= 'furnishingstatus', ylabel= 'Frequency of furnishingstatus')

plt.show()

# Compares the attributes to price
plt.figure(figsize = (10, 5))

plt.subplot(1, 2, 1)
plt.title('Mainroad vs. Price')
sns.boxplot(x = df.mainroad, y = df.price)

plt.subplot(1, 2, 2)
plt.title('Guestroom vs. Price')
sns.boxplot(x = df.guestroom, y = df.price)

plt.show()

plt.figure(figsize = (10, 5))

plt.subplot(1, 2, 1)
plt.title('Basement vs. Price')
sns.boxplot(x = df.basement, y = df.price)

plt.subplot(1, 2, 2)
plt.title('Hotwaterheating vs. Price')
sns.boxplot(x = df.hotwaterheating, y = df.price)

plt.show()

plt.figure(figsize = (10, 5))

plt.subplot(1, 3, 1)
plt.title('Airconditioning vs. Price')
sns.boxplot(x = df.airconditioning, y = df.price)

plt.subplot(1, 3, 2)
plt.title('Prefarea vs. Price')
sns.boxplot(x = df.prefarea, y = df.price)

plt.subplot(1, 3, 3)
plt.title('Furnishingstatus vs. Price')
sns.boxplot(x = df.furnishingstatus, y = df.price)
plt.xticks(rotation = 45)

plt.show()

# Compare the numerical data
numerical_list = [x for x in df.columns if df[x].dtype in ('int64','float64')]

print(numerical_list)

# Function that creates a scatterplot
def scatter(x, fig):
    plt.subplot(5, 2, fig)
    plt.scatter(df[x], df['price'])
    plt.title(x + ' vs. Price')
    plt.ylabel('Price')
    plt.xlabel(x)

plt.figure(figsize = (10, 20))

scatter('area', 1)
scatter('bedrooms', 2)
scatter('bathrooms', 3)
scatter('stories', 4)
scatter('parking', 5)

plt.tight_layout()
sns.pairplot(df)

plt.show()

# Correlation mapping
cor_matrix = df[numerical_list].corr()

plt.figure(figsize = (12, 8))

sns.heatmap(cor_matrix, annot= True, cmap= 'coolwarm')

plt.show()