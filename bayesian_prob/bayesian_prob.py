# Author: Danny Lillard
# Date: 2025.04.01
# Desc: Here I will create a basic Bayesian probability calculator.

import pandas as pd # just importing pandas to get dataframes, make life easier.
import numpy

# Functions
def gaussian_pdf(series: pd.Series):
    var = series.var()
    mean = series.mean()
    # Calculate the Gaussian PDF using the formula:
    numerator = numpy.exp(-((series - mean)**2) / (2 * var))
    denominator = numpy.sqrt(2 * numpy.pi * var)

    return numerator / denominator

def get_bayesian_prob(train, test, target) -> pd.DataFrame:
    # first, calculate prior from train
    prior = train[target].value_counts(normalize=True)
    print(prior)
    # need to use Gaussian Probability Density Function to
    # use the numerical fields.
    numeric_vals = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def',
       'Speed']
    # need to apply this for each class.
    groups = list(train['Type 1'].unique())
    print(groups)
    # now apply PDF function to each group
    for group in groups:
        train.loc[train['Type 1'] == group, numeric_vals] = train.loc[train['Type 1'] == group, numeric_vals].apply(gaussian_pdf)
    print(train.tail())



# Main here and below.
user_file_name = "Pokemon.csv"
target = 'Type 1'

df = pd.read_csv(user_file_name)

df=df.drop(['Name','Type 2','Generation','Legendary'],axis=1)
print(df.tail())

# defining the training and test split.
train=df.sample(frac=0.6,random_state=200)
test=df.drop(train.index)

df_with_calc = get_bayesian_prob(train, test, target)