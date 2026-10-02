# Machine-Learning-Feature-Selection-
A practical study of feature selection in Python with scikit-learn, comparing filter, wrapper, and embedded methods across two datasets.

Feature Selection in Machine Learning
A practical, two-part study of feature selection techniques in Python, built with scikit-learn. The project replicates a two-version resource end to end. Version 1 applies a range of selection methods to a single classification problem and compares them on one consistent metric. Version 2 works through the three main families of feature selection methods, with a clean worked example of each.

Overview
Feature selection is the process of choosing a smaller, more informative set of input features for a model. Dropping features that are noisy, redundant, or irrelevant can improve accuracy, cut training time, reduce the risk of overfitting, and make a model far easier to interpret. This project demonstrates how to apply feature selection in practice and how the different methods compare on real data.

The work is organised into two versions, which match the two labelled sections of the notebook.
Version 1: selection techniques on the Wine dataset
Version 1 uses the Wine dataset (178 samples, 13 chemical features, 3 classes). A Gradient Boosting Classifier trained on all 13 features sets the baseline, and every method below is measured against it using the weighted F1 score.
Variance method. An unsupervised filter. Features are scaled into a common range so their variances are comparable, and the lowest variance features are dropped.
Mutual information (SelectKBest). A supervised filter that keeps the features sharing the most information with the target, including non-linear relationships.
Recursive Feature Elimination (RFE). A wrapper that uses the model itself to rank features, dropping the weakest and repeating until the target number remain.
Boruta. An all-relevant wrapper that keeps every feature able to consistently outperform shuffled "shadow" copies of the data.
The version closes with a chart comparing all methods, showing that small, well-chosen subsets can match or beat the full feature set.
Version 2: the three families of feature selection on the Pima Diabetes dataset
Version 2 uses the Pima Indians Diabetes dataset (768 samples, 8 features, binary target) and takes each family of methods in turn.
Filter method (chi-squared). Scores each feature by its statistical dependence on the target and keeps the strongest.
Wrapper method (RFE with logistic regression). The same elimination strategy as Version 1, this time wrapped around a linear model to show how the choice of estimator shapes the result.
Embedded method (Ridge / L2). Selection happens during training, with an L2 penalty shrinking the coefficients of weaker features so coefficient size signals importance.
Techniques explored
Baseline modelling with Gradient Boosting
Variance threshold filtering with feature scaling
Mutual information filtering
Chi-squared filtering
Recursive Feature Elimination (tree-based and linear estimators)
Boruta all-relevant selection
Ridge (L2) regularisation as an embedded method
Evaluation with the weighted F1 score
Repository structure
File	Description
`Egwuogu Ibukun_Machine Learning –  Feature Selection.ipynb`	The full implementation. Version 1 and Version 2 are clearly labelled, with each method in its own section.
`Egwuogu Ibukun_Feature_Selection_Documentation.docx`	A detailed reference explaining every method, how it works, why it matters, and how the two versions connect.
`README.md`	This overview.
Run the notebook top to bottom. Version 1 relies only on scikit-learn's built-in Wine dataset, while Version 2 downloads the Pima Diabetes dataset from a public URL, so an internet connection is needed for that section. Boruta is installed from within the notebook, since it is not part of scikit-learn.
Tech stack
Python, pandas, numpy, scikit-learn, matplotlib, seaborn, Boruta.
Key takeaways
Smaller, well-chosen feature subsets frequently matched or outperformed the full set of features, which points to real redundancy in the data.
Different methods select different features, because each one measures a different property. That disagreement is expected rather than a problem.
Filter methods are fast and model-agnostic, wrapper methods are more accurate but heavier because they retrain the model, and embedded methods fold selection into training itself.
The best method to reach for depends on the dataset, the model, and whether the goal is the smallest possible subset or every feature that is genuinely relevant.
