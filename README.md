# MLFall2018 - Kaggle Competition Plasticc - Astronomy 
https://www.kaggle.com/c/PLAsTiCC-2018


# Objective

The Photometric LSST Astronomical Time-Series Classification Challenge (PLAsTiCC) asks Kagglers to help prepare to classify the data from this new survey. Competitors will classify astronomical sources that vary with time into different classes, scaling from a small training set to a very large test set of the type the LSST will discover.

https://arxiv.org/abs/1810.00001

# Plan

1) Start AWS Server for processing, import data
2) Use a naive Neural Network to achieve baseline
3) Iterate on feature extraction:
- OneHot passband values 

4) Iterate on Neural Network hyper parameters
- (initial) 

#Results
Submissions are evaluated using a weighted multi-class logarithmic loss. The overall effect is such that each class is roughly equally important for the final score.

Each object has been labeled with one type. For each object, you must submit a set of predicted probabilities (one for every category). The formula is then

$Log Loss=−⎛⎝∑Mi=1wi⋅∑Nij=1yijNi⋅lnpij∑Mi=1wi⎞⎠$
where N is the number of objects in the class set, M is the number of classes,  ln is the natural logarithm, yij is 1 if observation (i) belongs to class (j) and 0 otherwise, pij is the predicted probability that observation i belongs to class j.

The submitted probabilities for a given object are not required to sum to one because they are rescaled prior to being scored (each row is divided by the row sum). In order to avoid the extremes of the log function, predicted probabilities are replaced with max(min(p,1−10−15),10−15).
