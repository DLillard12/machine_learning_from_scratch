The first thing we covered in my class on machine learning was Bayesian probability. I believe this is because it is a basic version of machine learning, Bayesian probability starts with an initial belief, then updates as things are change.
I will be implementing a naive bayes classifier on some pokemon data.
# File Structure
- Bayesian_prob.md: You are here! This will contain my notes, thoughts, and meta info.
- bayesian_prob.py: Here I will create a script to calculate bayesian probability and output the stats to a csv.
- iris.csv: The classic dataset of classifing species of iris.
# Process
1. Load data
2. Define the train test split
3. fit the data. We do this by grabbing the priors, means, and variances for each type of each field.
4. Predict values. We do this by going by each row, then each class, apply the gaussian_pdf to the row and find which class has the best probability
5. print the true class with the predicted class.