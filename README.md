# Support-Vector-Machines-SVMs-
## SVMs and Machine Learning ##
Machine learning classifiers can be grouped into many categories. Two popular kinds of classifiers are:

Estimators of posterior probabilities
Direct estimators of decision boundaries
Models that estimate posterior probabilities are for example naive Bayes classifiers, logistic regression or even neural networks. These models attempt to recreate a posterior probability function. This approximate function of the posterior can then be evaluated to determine the probability that a sample belongs to each class, then make a decision.

Direct estimators of the decision boundary, such as the perceptrons and Support Vector Machines (SVMs), do not try to learn a probability function, instead, they learn a “line” or a high dimensional hyperplane, which can be used to determine the class of each sample. If a sample is to one side of the hyperplane it belongs to a class, otherwise, it belongs to the other.

These two approaches are fundamentally different, and they can affect the results of the classifier.