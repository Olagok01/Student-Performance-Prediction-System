# train_model.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Load and preprocess data
df = pd.read_csv('student-data.csv')


# Mapping function (from notebook)
def numerical_data(df):
    mapping_dict = {
        'school': {'GP': 0, 'MS': 1},
        'sex': {'M': 0, 'F': 1},
        'address': {'U': 0, 'R': 1},
        'famsize': {'LE3': 0, 'GT3': 1},
        'Pstatus': {'T': 0, 'A': 1},
        'Mjob': {'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4},
        'Fjob': {'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4},
        'reason': {'home': 0, 'reputation': 1, 'course': 2, 'other': 3},
        'guardian': {'mother': 0, 'father': 1, 'other': 2},
        'schoolsup': {'no': 0, 'yes': 1},
        'famsup': {'no': 0, 'yes': 1},
        'paid': {'no': 0, 'yes': 1},
        'activities': {'no': 0, 'yes': 1},
        'nursery': {'no': 0, 'yes': 1},
        'higher': {'no': 0, 'yes': 1},
        'internet': {'no': 0, 'yes': 1},
        'romantic': {'no': 0, 'yes': 1},
        'passed': {'no': 0, 'yes': 1}
    }

    for col, mapping in mapping_dict.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df, mapping_dict


df_processed, mappings = numerical_data(df.copy())


# Feature scaling function (from notebook)
def feature_scaling(df):
    scaling_params = {}
    for i in df.columns:
        if i == 'passed':  # Skip target
            continue
        col = df[i]

        if np.issubdtype(col.dtype, np.number):  # only numeric
            if np.max(col) > 6:
                Max = np.max(col)
                mean = np.mean(col)
                df[i] = (col - mean) / Max
                scaling_params[i] = {'method': 'mean_max', 'mean': mean, 'max': Max}
            else:
                Min = np.min(col)
                Max = np.max(col)
                df[i] = (col - Min) / (Max - Min)
                scaling_params[i] = {'method': 'min_max', 'min': Min, 'max': Max}
    return df, scaling_params


df_processed, scaling_params = feature_scaling(df_processed)

# Prepare data for training
X = df_processed.drop('passed', axis=1)
y = df_processed['passed']
features = list(X.columns)

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Save the model and preprocessing information
import os

os.makedirs('model', exist_ok=True)

with open('model/model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

with open('model/features.pkl', 'wb') as f:
    pickle.dump(features, f)

with open('model/mappings.pkl', 'wb') as f:
    pickle.dump(mappings, f)

with open('model/scaling_params.pkl', 'wb') as f:
    pickle.dump(scaling_params, f)

print("Model and preprocessing information saved successfully!")