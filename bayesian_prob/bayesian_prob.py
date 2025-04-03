# Author: Danny Lillard
# Date: 2025.04.01
# Desc: Here I will create a basic Bayesian probability calculator.

import pandas as pd # just importing pandas to get dataframes, make life easier.
import numpy
from collections import defaultdict # using this so I can have 2d dictionaries.


# Functions
def gaussian_pdf(value, mean, var):
    # Calculate the Gaussian PDF using the formula:
    numerator = numpy.exp(-((value - mean)**2) / (2 * var))
    denominator = numpy.sqrt(2 * numpy.pi * var)

    return numerator / denominator

def predict(test_data, type_priors, means, variances):
    predictions = []

    for i,row in test_data.iterrows():
        type_probs = defaultdict(dict)

        for type in type_priors:
            prior = type_priors[type]
            likelihood = 1  # Start with 1, as we'll multiply PDFs

            numeric_fields = ['sepal_length','sepal_width','petal_length','petal_width']
            for field in numeric_fields:
                value = row[field]
                mean = means[type][field]
                variance = variances[type][field]
                
                # Compute Gaussian PDF
                pdf = gaussian_pdf(value, mean, variance)
                likelihood *= pdf  # Multiply all likelihoods

            type_probs[type] = numpy.log(likelihood * prior)  # Multiply by prior
        #best_class = class with highest probability
        best_pred = max(type_probs,key=type_probs.get)
        #append best_class to predictions
        predictions.append([row['species'], best_pred])
    return predictions

def fit(train:pd.DataFrame):
    # initialize empty dictionaries for type_priors, means, and variances
    type_priors = defaultdict(dict)
    means = defaultdict(dict)
    variances = defaultdict(dict)

    types = train['species'].unique()

    
    for type in types:
        # subset = dataframe where target_column == class
        subset = train.loc[train['species'] == type]
        type_priors[type] = len(subset)/len(train)
        numeric_fields = ['sepal_length','sepal_width','petal_length','petal_width']
        for field in numeric_fields:
            means[type][field] = subset[field].mean()
            variances[type][field] = subset[field].var()

    return type_priors, means, variances



# Main here and below.
user_file_name = 'iris.csv'

df = pd.read_csv(user_file_name)

print(df.tail())

# defining the training and test split.
train=df.sample(frac=0.6,random_state=200)
test=df.drop(train.index)

type_priors, means, variances = fit(train)
preds = predict(test,type_priors,means,variances)

print(preds)