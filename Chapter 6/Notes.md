# Decision Trees

Like SVMs, decision trees are versatile Machine Learning algorithms that can perform both classification and regression (even multi-output). 

These are the fundamentals to the Random Forest algorithms we learn later.

## Making Predictions

It is essentially a long if-then condition decision tree. You start at the root node, see the conditions for certain features, and then move accordingly for that. 

Scikit learn uses CART algorithm, essentially meaning that it only produces binary trees. 

[Decistion Tree](iris_tree.png)

## Estimating Class Probabilities

A DT can also estimate the probability that an instance belongs to a particular class _k_. 

## The CART Training Algorithm

CART (Classification and Regression Tree) algorithm is used by Scikit-Learn to train Decision Trees. It works by:
1. First, split the training set into two subsets using a single feature k and a threshold $t_k$. 
    - It chooses k and $t_k$ by searching for a pair (k, $t_k$) that produces the purest subsets
2. Once split, it then splits each subset again using the same logic recursively, until it can no longer do so

This is a greedy algorithm (will study in CPSC 413); it greedily searches for an optimum split at the top level, then repeats the process at each subsequent level.

## Computational Complexity 

Making predictions requires traversing the Decision Tree from the root to a leaf. DTs tend to be approximately balanced, so traversing them means roughly O($\log_2(m)$) nodes. This is also the prediction complexity seeing as how each node requires a simple check.

The training algorithm compares all features on all samples at each node. Comparing all features on all samples at each node results in a training complexity of 
O($n \cdot m\log_2(m)$). This is slow, and scikit-learn presorts smaller datasets for faster computation. Its in the larger sets where the computation can be bad.

## Gini Impurity or Entropy?

Gini is used by default, but we can set it to use entropy instead by setting the `criterion` hyperparameter. 

The concept of entropy is from thermodynamics as a measure of molecular disorder: entropy approaches zero when molecules are still and well ordered. In ML, entropy is frequently used as an impurity measure: a set's entropy is zero when it contains instances of only one class. 

They usually produce same output. Main differences:
- Gini is faster
- Entropy produces more balanced trees

## Regularization Hyperparameters
Decision trees make very few assumptions about the data itself; hence, if left unconstrained, the tree structure will adapt itself to the training data, overfitting it. Such a model is called nonparametric model, not because it has no parameters but rather because the number of parameters is not determined prior to training.

By constraining with parameters, we can make sure the model doesn't just embody the training set, and we can limit overfitting this way.

To avoid DT overfitting, we have to restrict its freedom during training. This is called regularization. The regularization hyperparameters depend on the algorithm used, but generally you can atleast restrict the maximum depth of the Decision Tree (`max_depth` hyperparameter). Reducing it will regularize the model and this reduce the risk of overfitting. 

A couple of other hyperparameters to keep in note of:
1. `min_samples_splits`: minimum number of samples a node must have before it can be split
2. `min_samples_leaf`: the minimum number of samples a leaf node must have
3. `min_weight_fraction_leaf`: same as 2 but expressed as a fraction of the total number of weighted instances
4. `max_features`: the maximum number of features that are evaluated for splitting at each node

Good rule of thumb: increase `min` hyperparameters and decrease `max` ones for regularization.

## Regression

It will make a similar tree as before, just now it will be predicting values at each node rather than classifying into a class. Instead of minimizing Gini or impurity in the CART algorithm's splits, we instead minimize the MSE (Mean squared error). Check textbook pages 183-184 for better diagrams

Again, just like in classifiers, make sure to regularize well so that we do not overfit.

## Instability
DT seem to be simple to understand, easy to use and interpret, versatile and powerful. However, they have a few limitations:
- DT love orthogonal decision boundaries (all splits are perpendicular to an axis), which makes them sensitive to training set rotation

More generally, the main issue is that DTs are very sensitive to small variations in training data. Random forests help limit this instability by averaging predictions over many trees.

## Exercises
