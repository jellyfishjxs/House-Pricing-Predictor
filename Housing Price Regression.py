import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor

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

#plt.show()

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

#plt.show()

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

#plt.show()

plt.figure(figsize = (15, 5))

plt.subplot(1, 3, 1)
plt1 = df['furnishingstatus'].value_counts().plot(kind= 'bar')
plt.title('Furnishingstatus Histogram')
plt1.set(xlabel= 'furnishingstatus', ylabel= 'Frequency of furnishingstatus')

#plt.show()

# Compares the attributes to price
plt.figure(figsize = (10, 5))

plt.subplot(1, 2, 1)
plt.title('Mainroad vs. Price')
sns.boxplot(x = df.mainroad, y = df.price)

plt.subplot(1, 2, 2)
plt.title('Guestroom vs. Price')
sns.boxplot(x = df.guestroom, y = df.price)

#plt.show()

plt.figure(figsize = (10, 5))

plt.subplot(1, 2, 1)
plt.title('Basement vs. Price')
sns.boxplot(x = df.basement, y = df.price)

plt.subplot(1, 2, 2)
plt.title('Hotwaterheating vs. Price')
sns.boxplot(x = df.hotwaterheating, y = df.price)

#plt.show()

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

#plt.show()

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

#plt.show()

# Correlation mapping
cor_matrix = df[numerical_list].corr()

plt.figure(figsize = (12, 8))

sns.heatmap(cor_matrix, annot = True, cmap = 'coolwarm', linewidths = 0.5)
plt.title('Correlation Heatmap')

#plt.show()

# Function that creates dummy variables
def dummies(x, df):
    temp = pd.get_dummies(df[x], drop_first = True).astype(int)
    df = pd.concat([df, temp], axis = 1)
    df.drop([x], axis = 1, inplace = True)
    return df

df = dummies('mainroad', df)
df = dummies('guestroom', df)
df = dummies('hotwaterheating', df)
df = dummies('basement', df)
df = dummies('airconditioning', df)
df = dummies('prefarea', df)
df = dummies('furnishingstatus', df)

df.shape

# Creates a model

np.random.seed(0)

df_train, df_test = train_test_split(df, train_size = 0.75, test_size = 0.25, random_state = 100)
scalar = MinMaxScaler()

df_train[numerical_list] = scalar.fit_transform(df_train[numerical_list])

df_train.head()

y_train = df_train.pop('price')
x_train = df_train

rfe = RFE(estimator = LinearRegression(), n_features_to_select = 10)
rfe = rfe.fit(x_train, y_train)

list(zip(x_train.columns, rfe.support_, rfe.ranking_))

x_train.columns[rfe.support_]

x_train_rfe = x_train[x_train.columns[rfe.support_]]
x_train_rfe.head()

def build_model(x, y):
    x = sm.add_constant(x)
    lm = sm.OLS(y, x).fit()
    print(lm.summary())
    return x

def checkVIF(x):
    vif = pd.DataFrame()
    vif['Features'] = x.columns
    vif['VIF'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = 'VIF', ascending = False)
    return(vif)

# MODEL 1

x_train_new = build_model(x_train_rfe, y_train)
x_train_new = x_train_new.drop(["bedrooms"], axis = 1)

# Model 2

x_train_new = build_model(x_train_new, y_train)
checkVIF(x_train_new)

x_train_new = x_train_new.drop(["yes"], axis = 1)
checkVIF(x_train_new)


lm = sm.OLS(y_train, x_train_new).fit()

y_train_price = lm.predict(x_train_new)

fig = plt.figure()
sns.displot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)
plt.xlabel('Errors', fontsize = 18)

df_test[numerical_list] = scalar.fit_transform(df_test[numerical_list])

y_test = df_test.pop('price')
x_test = df_test

x_train_new = x_train_new.drop('const', axis = 1)
x_test_new = x_test[x_train_new.columns]

x_test_new = sm.add_constant(x_test_new)

y_pred = lm.predict(x_test_new)

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

fig = plt.figure()
plt.scatter(y_test, y_pred)
fig.suptitle('y_test vs y_pred', fontsize = 20)
plt.xlabel('y_test', fontsize = 18)
plt.ylabel('y_pred', fontsize = 16)

print(lm.summary())