import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

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

# print(train_inputs[numeric_cols].describe())
# print(len(train_inputs))
# print(train_inputs[categorical_cols].nunique())
# print(train_inputs[numeric_cols].isna().sum() / len(train_inputs) * 100)

# Replace numeric NaN values with mean
imputer = SimpleImputer(strategy ='mean')
imputer.fit(raw_df[numeric_cols])

train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])

# Scale all numeric values to [-1, 1]
scaler = MinMaxScaler()
scaler.fit(raw_df[numeric_cols])

train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

# Encode categoricals data
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(raw_df[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

# Create new encoded columns and concatenate with original dataframe
train_encoded = pd.DataFrame(encoder.transform(train_inputs[categorical_cols]),
                             columns=encoded_cols,
                             index=train_inputs.index
)
train_inputs = pd.concat([train_inputs[numeric_cols], train_encoded], axis=1)

val_encoded = pd.DataFrame(encoder.transform(val_inputs[categorical_cols]),
                             columns=encoded_cols,
                             index=val_inputs.index
)
val_inputs = pd.concat([val_inputs[numeric_cols], val_encoded], axis=1)

test_encoded = pd.DataFrame(encoder.transform(test_inputs[categorical_cols]),
                             columns=encoded_cols,
                             index=test_inputs.index
)
test_inputs = pd.concat([test_inputs[numeric_cols], test_encoded], axis=1)

# Create Logistic Regression model
model = LogisticRegression(solver='liblinear')

model.fit(train_inputs, train_targets)

feature_names = numeric_cols + encoded_cols
coef_df = pd.DataFrame({
    "feature": feature_names,
    "weight": model.coef_[0]
})

# Print out coeffs and weights
# coef_df = coef_df.sort_values(by="weight", ascending=False)
# pd.set_option('display.max_rows', None)
# print(coef_df)

# sns.barplot(data=coef_df.tail(20), x='weight', y='feature')
# plt.show()

# # Print distribution of No and Yes prediction base on train_inputs data
# train_preds = pd.Series(model.predict(train_inputs))
# print("\nYes/No distribution:\n",train_preds.value_counts(normalize=True))

# # Print predictions confidence levels for each input row
# train_probs = model.predict_proba(train_inputs)
# print("\nconfidence levels:\n", train_probs)

def predict_and_plot(inputs, targets, name=''):
    preds = model.predict(inputs)

    # Calculate accuracy of prediction compared to target values
    accuracy = accuracy_score(targets, preds)
    print("Accuracy = {:.2f}%".format(accuracy * 100))

    # Visualize errors with confusion matrix
    cf = confusion_matrix(targets, preds, normalize='true')
    plt.figure()
    sns.heatmap(cf, annot=True)
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title('{} Confusion Matrix'.format(name))
    plt.show()
    return preds

# train_preds = predict_and_plot(train_inputs, train_targets, 'Training')
# val_preds = predict_and_plot(val_inputs, val_targets, 'Validation')
test_preds = predict_and_plot(test_inputs, test_targets, 'Pred')

def random_guess(inputs):
    return np.random.choice(["No", "Yes"], len(inputs))

def all_no(inputs):
    return np.full(len(inputs), "No")

print("\nRandom generated model accuracy =\n", accuracy_score(test_targets, random_guess(test_inputs)))
print("\nAll 'No' model accuracy =\n", accuracy_score(test_targets, all_no(test_inputs)))
