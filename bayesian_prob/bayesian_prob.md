The first thing we covered in my machine learning class was Bayesian probability. I believe this is because it is a basic version of machine learning, Bayesian probability starts with an initial belief, then updates as things are change.
I will be implementing a naive bayes classifier on some pokemon data.
# Todo
Make the predict function do the operation by vector not by a loop
# File Structure
- Bayesian_prob.md: You are here! This will contain my notes, thoughts, and meta info.
- bayesian_prob.py: Here I will create a script to calculate bayesian probability and output the stats to a csv.
- iris.csv: The classic dataset of classifing species of iris.
# Process
1. Load data.
2. Define the train test split.
3. fit the data. We do this by grabbing the priors, means, and variances for each type of each field.
4. Predict values. We do this by going by each row, then each class, apply the gaussian_pdf to the row and find which class has the best probability.
5. print the true class with the predicted class.
# Scope
This is a fairly simple project just using the iris dataset, it is not general. 
# What was learned
In the coursework we used a dataset with discrete values of pokemon stats e.g. Name: Pikachu, Att: High, Def: Low, etc. This time I used continous values. With this I had to implement the gaussian probability density function. I found this interesting, at first I thought that the Gaussian PDF had to be applied during the fit phase, after some trial I found that was wrong and I only had to pull the statistics during the fit phase, this changed my understanding of the predict phase. We are putting new values into the model, and it is calculating which class is the most likely.
It is like the fit phase is just learning the lay of the land, trying to approximate the function, then the predict applies that learning.
Interesting video that says neural networks can approximate any function (given infinite time and resources): https://www.youtube.com/watch?v=TkwXa7Cvfr8