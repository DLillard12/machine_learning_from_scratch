# Author: Daniel Lillard
# Date: 2025.04.03
# Desc: This is a python script where I am implementing linear regression by hand.

import pandas as pd
from sklearn import preprocessing

# Functions

# loss

# gradient descent

# Predict

# fit

# Main
user_file = 'automobile.csv'

numeric_fields = []
categorical_fields = []
unusable_fields = []

raw_data = pd.read_csv(user_file)

print(raw_data.tail())

numeric_fields = ['cylinders','displacement','horsepower','weight','acceleration','model_year']
categorical_fields = ['origin']
unusable_fields = ['name']

# if there are rows with NA drop them
raw_data = raw_data.dropna()

# Split the data into train and test
train=raw_data.sample(frac=0.7,random_state=201)
test=raw_data.drop(train.index)

train_features = train.drop(['mpg'],axis=1)
train_target = train['mpg']

test_features = test.drop(['mpg'],axis=1)
test_target = test['mpg']

# Going to do some cleaning up here, we must: 
# normalize the numeric data
# Encode the categorical
normalizer = preprocessing.Normalizer()

normalized_train_features = normalizer.fit_transform(train_features[numeric_fields])
normalized_train_features = pd.DataFrame(normalized_train_features,columns=numeric_fields,index=train_features[numeric_fields].index)

normalized_test_features = normalizer.transform(test_features[numeric_fields])
normalized_test_features = pd.DataFrame(normalized_test_features,columns=numeric_fields)

print('Now just normalized----------------\n',normalized_train_features)


# now do one hot encoding on the categorical data.
encoded_train_features = pd.get_dummies(train_features[categorical_fields])
encoded_test_features = pd.get_dummies(test_features[categorical_fields])

print(normalized_train_features.index.equals(train_features.index))
# the above proves that the normalized train and the train_features do not have the same indicies. 

# now bring the three back together
train_features = pd.concat([train_features['name'], normalized_train_features, encoded_train_features], axis=1)
test_features = pd.concat([test_features['name'], normalized_test_features, encoded_test_features], axis=0)

print('Now encoded, and normalized----------------\n',train_features.tail())

train_features.to_csv('test_output.csv')

# cool, so the data is now ready to be fit to a model.