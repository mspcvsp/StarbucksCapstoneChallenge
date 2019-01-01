# StarbucksCapstoneChallenge
[Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) [Capstone Project - Analyze Starbucks Capstone Challenge Dataset](https://medium.com/@mspcvsp/starbucks-capstone-challenge-dataset-customer-offer-success-prediction-916bbcdc5bd5)  

# Project Overview
Customer satisfaction drives business success and data analytics provides insight into what customers think. For example, the phrase "[360-degree customer view](https://searchsalesforce.techtarget.com/definition/360-degree-customer-view)" refers to aggregating data describing a customer's purchases and customer service interactions.
  
The Starbucks [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) Capstone challenge data set is a simulation of customer behavior on the Starbucks rewards mobile application. Periodically, Starbucks sends offers to users that may be an advertisement, discount, or buy one get on free (BOGO). An important characteristic regarding this dataset is that not all users receive the same offer.
  
This data set contains three files. The first file describes the characteristics of each offer, including its duration and the amount  a customer needs to spend to complete it (difficulty). The second file contains customer demographic data including their age, gender, income, and when they created an account on the Starbucks rewards mobile application. The third file describes customer purchases and when they received, viewed, and completed an offer. An offer is only successful when a customer both views an offer and meets or exceeds its difficulty within the offer's duration.
  
## Problem Statement / Metrics 
The problem that I chose to solve is to build a model that predicts whether a customer will respond to an offer. My strategy for solving this problem has four steps. First, I will combine the offer portfolio, customer profile, and transaction data. Each row of this combined dataset will describe an offer's attributes, customer demographic data, and whether the offer was successful. Second, I will assess the [accuracy](https://developers.google.com/machine-learning/crash-course/classification/accuracy) and [F1-score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) of a naive model that assumes all offers were successful. This provides me a baseline for evaluating the performance of models that I construct. Accuracy measures how well a model correctly predicts whether an offer is successful. However, if the percentage of successful or unsuccessful offers is very low, [accuracy is not a good measure of model performance](https://www.manning.com/books/practical-data-science-with-r). For this situation, evaluating a model's [precision and recall](https://towardsdatascience.com/beyond-accuracy-precision-and-recall-3da06bea9f6c) provides better insight to its performance. I chose the F1-score metric because it is "[a weighted average of the precision and recall metrics"](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html). Third, I will compare the performance of [logistic regression](https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc), [random forest](https://towardsdatascience.com/the-random-forest-algorithm-d457d499ffcd), and [gradient boosting](https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/) models. Fourth, I will refine the parameters of the model that has the highest accuracy and F1-score.  
  
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
