# StarbucksCapstoneChallenge
[Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) [Capstone Project - Analyze Starbucks Capstone Challenge Dataset](https://medium.com/@mspcvsp/starbucks-capstone-challenge-dataset-customer-offer-success-prediction-916bbcdc5bd5)  

## Project Overview
Customer satisfaction drives business success and data analytics provides insight into what customers think. For example, the phrase "[360-degree customer view](https://searchsalesforce.techtarget.com/definition/360-degree-customer-view)" refers to aggregating data describing a customer's purchases and customer service interactions.
  
The Starbucks [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) Capstone challenge data set is a simulation of customer behavior on the Starbucks rewards mobile application. Periodically, Starbucks sends offers to users that may be an advertisement, discount, or buy one get on free (BOGO). An important characteristic regarding this dataset is that not all users receive the same offer.
  
This data set contains three files. The first file describes the characteristics of each offer, including its duration and the amount  a customer needs to spend to complete it (difficulty). The second file contains customer demographic data including their age, gender, income, and when they created an account on the Starbucks rewards mobile application. The third file describes customer purchases and when they received, viewed, and completed an offer. An offer is only successful when a customer both views an offer and meets or exceeds its difficulty within the offer's duration.
  
## Problem Statement / Metrics 
The problem that I chose to solve is to build a model that predicts whether a customer will respond to an offer. My strategy for solving this problem has four steps. First, I will combine the offer portfolio, customer profile, and transaction data. Each row of this combined dataset will describe an offer's attributes, customer demographic data, and whether the offer was successful. Second, I will assess the [accuracy](https://developers.google.com/machine-learning/crash-course/classification/accuracy) and [F1-score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) of a naive model that assumes all offers were successful. This provides me a baseline for evaluating the performance of models that I construct. Accuracy measures how well a model correctly predicts whether an offer is successful. However, if the percentage of successful or unsuccessful offers is very low, [accuracy is not a good measure of model performance](https://www.manning.com/books/practical-data-science-with-r). For this situation, evaluating a model's [precision and recall](https://towardsdatascience.com/beyond-accuracy-precision-and-recall-3da06bea9f6c) provides better insight to its performance. I chose the F1-score metric because it is "[a weighted average of the precision and recall metrics"](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html). Third, I will compare the performance of [logistic regression](https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc), [random forest](https://towardsdatascience.com/the-random-forest-algorithm-d457d499ffcd), and [gradient boosting](https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/) models. Fourth, I will refine the parameters of the model that has the highest accuracy and F1-score.  

## Results Summary
- Model ranking based on training data [accuracy](https://www.datarobot.com/wiki/accuracy/)  
    1. RandomForestClassifier model accuracy: 0.742
    2. GradientBoostingClassifier model accuracy: 0.736
    3. LogisticRegression model accuracy: 0.722
    4. Naive predictor accuracy: 0.471
- Model ranking based on training data [F1-score](https://en.wikipedia.org/wiki/Precision_and_recall)  
    1. RandomForestClassifier model f1-score: 0.735
    2. GradientBoostingClassifier model f1-score: 0.725
    3. LogisticRegression model f1-score: 0.716
    4. Naive predictor f1-score: 0.640
- Results suggest that the random forest model has the best training data accuracy and F1-score  

[Bias and variance](https://machinelearningmastery.com/gentle-introduction-to-the-bias-variance-trade-off-in-machine-learning/) are two characteristics of a machine learning model. Bias refers to inherent model assumptions regarding the decision boundary between different classes. On the other hand, variance refers a model's sensitivity to changes in its inputs. 
A logistic regression model constructs a [linear decision boundary](https://datascience.stackexchange.com/questions/6048/decision-tree-or-logistic-regression) to separate successful and unsuccessful offers. However, my exploratory analysis of customer demographics for each offer suggests that this decision boundary will be non-linear. Therefore, an [ensemble method](https://datascience.stackexchange.com/questions/6048/decision-tree-or-logistic-regression) like random forest or gradient boosting should perform better.

Both [random forest](http://blog.citizennet.com/blog/2012/11/10/random-forests-ensembles-and-performance-metrics) and gradient boosting models are a combination of multiple decision trees. A random forest classifier randomly samples the training data with replacement to construct a set of decision trees that are combined using [majority voting](http://blog.citizennet.com/blog/2012/11/10/random-forests-ensembles-and-performance-metrics). In contrast, [gradient boosting](http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/) iteratively constructs a set of decision trees with the goal of reducing the number of misclassified training data samples from the previous iteration. A consequence of these [model construction strategies](https://stats.stackexchange.com/questions/173390/gradient-boosting-tree-vs-random-forest) is that the depth of decision trees generated during random forest model training are typically greater than gradient boosting weak learner depth to minimize [model variance](https://stats.stackexchange.com/questions/173390/gradient-boosting-tree-vs-random-forest). Typically, gradient boosting performs better than a random forest classifier. However, gradient boosting may overfit the training data and requires additional effort to tune. A random forest classifier is less prone to overfitting because it constructs decision trees from random training data samples. Also, a random forest classifier's hyperparameters are easier to optimize (1).

The problem that I chose to solve was to build a model that predicts whether a customer will respond to an offer. My strategy for solving this problem has four steps. First, I combined offer portfolio, customer profile, and transaction data. Second, I assessed the accuracy and F1-score of a naive model that assumes all offers were successful. Third, I compared the performance of logistic regression, random forest, and gradient boosting models. This analysis suggests that a random forest model has the best training data accuracy and F1-score. Fourth, I refined random forest model hyperparameters using a grid search. My analysis suggests that the resulting random forest model has an training data accuracy of 0.753 and an F1-score of 0.746. 
The test data set accuracy of 0.736 and F1-score of 0.727 suggests that the random forest model I constructed did not [overfit the training data](https://elitedatascience.com/overfitting-in-machine-learning).  

"[Feature importance](https://towardsdatascience.com/running-random-forests-inspect-the-feature-importances-with-this-code-2b00dd72b92e)" refers to a numerical value that describes a feature's contribution to building a model that maximizes its evaluation metric. A random forest classifier is an example of a model that estimates feature importance during training. My analysis of the Starbucks Capstone Challenge customer offer effectiveness training data suggests that the top five features based on their importance are:  
  
    1. Offer difficulty (how much money a customer must spend to complete an offer)  
    2. Offer duration   
    3. Offer reward  
    4. Customer income  
    5. Whether a customer created an account on the Starbucks rewards mobile application in 2018  
    
Since the top three features are associated with an customer offer, it may be possible to improve the performance of a random forest model by creating features that describe an offer's success rate as a function of offer difficulty, duration, and reward. These additional features should provide a random forest classifier the opportunity to construct a better decision boundary that separates successful and unsuccessful customer offers.

(1) [How can the performance of a Gradient Boosting Machine be worse than Random -Forests](https://www.quora.com/How-can-the-performance-of-a-Gradient-Boosting-Machine-be-worse-than-Random-Forests).  

## Files  
- Starbucks_Capstone_notebook.ipynb  
  - [Jupyter notebook](https://jupyter.org/) that performs three tasks:  
    - Combines offer portfolio, customer demographic, and customer transaction data  
    - Generates training customer demographic data visualizations and computes summary statistics  
    - Generates logistic regression, random forest, & gradient boosting models  
- clean_data.py  
  - Python software that combines offer portfolio, customer demographic, and customer transaction data  
- exploratory_data_analysis.py  
  - Generates training customer demographic data visualizations and computes summary statistics  
- LICENSE  
  - Repository license file  
.gitignore  
  - [Describes](https://git-scm.com/docs/gitignore) files and/or directories that should not be checked into revision control  
- README.md  
  - [Markdown](https://guides.github.com/features/mastering-markdown/) file that summarizes this repository  
	
## Python Libraries Used
-[Python Data Analysis Library](https://pandas.pydata.org/)  
-[Numpy](http://www.numpy.org/)  
-[Matplotlib](https://matplotlib.org/)  
-[seaborn: Statistical Data Visualization](https://seaborn.pydata.org/)  
-[re: Regular expression operations](https://docs.python.org/3/library/re.html)  
-[os — Miscellaneous operating system interfaces](https://docs.python.org/3/library/os.html)  
-[scikit-learn: Machine Learning in Python](https://scikit-learn.org/stable/)  
-[Joblib: running Python functions as pipeline jobs](https://joblib.readthedocs.io/en/latest/)  
  
## References
- [360-degree customer view definition](https://searchsalesforce.techtarget.com/definition/360-degree-customer-view)  
- [Model accuracy definition](https://developers.google.com/machine-learning/crash-course/classification/accuracy)  
- [F1-score definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)  
- [Evaluation of models with unbalanced classes](https://www.manning.com/books/practical-data-science-with-r)  
- [Beyond accuracy precision and recall](https://towardsdatascience.com/beyond-accuracy-precision-and-recall-3da06bea9f6c)  
- [Logistic regression detailed overview](https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc)  
- [Random forest algorithm](https://towardsdatascience.com/the-random-forest-algorithm-d457d499ffcd)  
- [Gradient boosting algorithm](https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/)  
- [Multi label binarizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html#sklearn.preprocessing.MultiLabelBinarizer)  
- [Why one hot encode data in machine learning?](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/)  
- [Using categorical data with one hot encoding](https://www.kaggle.com/dansbecker/using-categorical-data-with-one-hot-encoding)  
- [Is there a rule-of-thumb for how to divide a dataset into training and validation sets?](https://stackoverflow.com/questions/13610074/is-there-a-rule-of-thumb-for-how-to-divide-a-dataset-into-training-and-validatio)  
- [The use of feature scaling in scikit-learn](https://stackoverflow.com/questions/51660001/the-use-of-feature-scaling-in-scikit-learn)  
- [Machine learning evaluate classification model](https://www.ritchieng.com/machine-learning-evaluate-classification-model/)
- [Hyperparameter tuning the random forest in Python using scikit-learn](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74)  
- [Random forest feature importances](https://towardsdatascience.com/running-random-forests-inspect-the-feature-importances-with-this-code-2b00dd72b92e)  
- [Gentle introduction to the bias variance trade-off in machine learning](https://machinelearningmastery.com/gentle-introduction-to-the-bias-variance-trade-off-in-machine-learning/)  
- [Decision tree or logistic regression?](https://datascience.stackexchange.com/questions/6048/decision-tree-or-logistic-regression)  
- [Random forests ensembles and performance metrics](http://blog.citizennet.com/blog/2012/11/10/random-forests-ensembles-and-performance-metrics)  
- [A Kaggle master explains gradient boosting](http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/)  
- [Gradient boosting tree vs random forest](https://stats.stackexchange.com/questions/173390/gradient-boosting-tree-vs-random-forest)  
- [How can the performance of a Gradient Boosting Machine be worse than Random Forests](https://www.quora.com/How-can-the-performance-of-a-Gradient-Boosting-Machine-be-worse-than-Random-Forests)  
- [Overfitting in machine learning](https://elitedatascience.com/overfitting-in-machine-learning)
- [Rotate axis text in Python matplotlib](https://stackoverflow.com/questions/10998621/rotate-axis-text-in-python-matplotlib)
- [Analytic dataset definition](https://github.com/jtleek/datasharing)
- [Set order of columns in pandas DataFrame](https://stackoverflow.com/questions/41968732/set-order-of-columns-in-pandas-dataframe)  
- [Python pandas selecting rows whose column value is null none nan](https://stackoverflow.com/questions/40245507/python-pandas-selecting-rows-whose-column-value-is-null-none-nan)  
- [scikit-learn MultiLabelBinarizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html#sklearn.preprocessing.MultiLabelBinarizer)  
- [Python round up to the nearest ten](https://stackoverflow.com/questions/26454649/python-round-up-to-the-nearest-ten)  
- [datetime strptime in Python](https://stackoverflow.com/questions/44596077/datetime-strptime-in-python)  
- [How to match exact multiple strings in Python](https://stackoverflow.com/questions/4953272/how-to-match-exact-multiple-strings-in-python)
- [How to determine a Python variable's type](https://stackoverflow.com/questions/402504/how-to-determine-a-python-variables-type)
- [Pandas DataFrame settingwithcopywarning a value is trying to be set on a copy](https://stackoverflow.com/questions/49728421/pandas-dataframe-settingwithcopywarning-a-value-is-trying-to-be-set-on-a-copy)  
- [Should binary features be one hot encoded](https://stackoverflow.com/questions/43515877/should-binary-features-be-one-hot-encoded)  
- [Select pandas DataFrame rows and columns using iloc, loc and ix](https://www.shanelynn.ie/select-pandas-dataframe-rows-and-columns-using-iloc-loc-and-ix/)  
- [How to merge two dictionaries in a single expression](https://stackoverflow.com/questions/38987/how-to-merge-two-dictionaries-in-a-single-expression)  
- [Python to print out status bar and percentage](https://stackoverflow.com/questions/3002085/python-to-print-out-status-bar-and-percentage)  
- [Progress bar introduction](https://progressbar-2.readthedocs.io/en/latest/index.html#introduction)  
- [Progress bar documentation](https://progressbar-2.readthedocs.io/en/latest/progressbar.bar.html)  
- [Reversing one hot encoding in pandas](https://stackoverflow.com/questions/38334296/reversing-one-hot-encoding-in-pandas)  
- [If else in a list comprehension](https://stackoverflow.com/questions/4406389/if-else-in-a-list-comprehension)  
- [Pandas DataFrame groupby two columns and get counts](https://stackoverflow.com/questions/17679089/pandas-dataframe-groupby-two-columns-and-get-counts)  
- [Converting a pandas groupby object to DataFrame](https://stackoverflow.com/questions/10373660/converting-a-pandas-groupby-object-to-dataframe)  
- [Change order of columns in stacked bar plot](https://stackoverflow.com/questions/32015669/change-order-of-columns-in-stacked-bar-plot)  
- [Print Python version](https://stackoverflow.com/questions/1252163/printing-python-version-in-output)  
