import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import seaborn as sns

raw_df = pd.read_csv('data/weatherAUS.csv')

# Remove lines with missing RainToday or RainTomorrow
raw_df.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)

# # Plot data amount each year
# plt.title('asd')
# sns.countplot(x=pd.to_datetime(raw_df.Date).dt.year)
# plt.show()

# Use data before 2015 for training, 2015 for validation and after 2015 for test
year = pd.to_datetime(raw_df.Date).dt.year
train_df = raw_df[year < 2015]
val_df = raw_df[year == 2015]
test_df = raw_df[year > 2015]

# Define input columns and target columns
input_cols = list(train_df.columns)[1:-1]
target_col = 'RainTomorrow'

# Create inputs and targets for each: train, validation and test data
train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()

val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy()

test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()

# Separate numerical columns and categorical columns
numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes(include=np.object_).columns.tolist()

print(train_inputs[numeric_cols].describe())
print(len(train_inputs))
# print(train_inputs[categorical_cols].nunique())

imputer = SimpleImputer(strategy ='mean')

print(train_inputs[numeric_cols].isna().sum() / len(train_inputs) * 100)
