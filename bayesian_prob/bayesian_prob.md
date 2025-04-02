The first thing we covered in my class on machine learning was Bayesian probability. I believe this is because it is a basic version of machine learning, Bayesian probability starts with an initial belief, then updates as things are change.
I will be implementing a naive bayes classifier on some pokemon data.
# File Structure
- Bayesian_prob.md: You are here! This will contain my notes, thoughts, and meta info.
- bayesian_prob.py: Here I will create a script to calculate bayesian probability and output the stats to a csv.
- Pokemon.csv: A stripped down version of the dataset from this page: https://www.kaggle.com/datasets/abcsds/pokemon It originally held generations 1-6 but I cut this down to just generation 3.
# Process
1. Load data
2. Define the train test split
3. Pass train data, test data, and output variable to a function that will return the test data with predictions.
4. Save that to a csv.