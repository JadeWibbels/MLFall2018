# MLFall2018 - Kaggle Competition Plasticc - Astronomy 
https://www.kaggle.com/c/PLAsTiCC-2018


# Objective

The Photometric LSST Astronomical Time-Series Classification Challenge (PLAsTiCC) asks Kagglers to help prepare to classify the data from this new survey. Competitors will classify astronomical sources that vary with time into different classes, scaling from a small training set to a very large test set of the type the LSST will discover.

https://arxiv.org/abs/1810.00001

# Plan

1) Start AWS Server for processing, import data
2) Use a naive Neural Network to achieve baseline
3) Iterate on feature engineering:

Good resource for feature engineering ideas: https://machinelearningmastery.com/an-introduction-to-feature-selection/
https://towardsdatascience.com/why-how-and-when-to-apply-feature-selection-e9c69adfabf2

- Brainstorming or Testing features
- Deciding what features to create
- Creating features
- Checking how the features work with your model
- Improving your features if needed
- Go back to brainstorming/creating more features until the work is done.

Featuring Engineering implamented:
- OneHot passband values 
- check baseline with and without distmod (Alex similarirty check)

4) Iterate on Neural Network hyper parameters
- (initial) 

# Results
Submissions are evaluated using a weighted multi-class logarithmic loss. The overall effect is such that each class is roughly equally important for the final score.

Each object has been labeled with one type. For each object, you must submit a set of predicted probabilities (one for every category). 
