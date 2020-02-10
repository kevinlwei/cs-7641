from sklearn.svm import SVC
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
import data
import math
import analysis
import numpy as np
import matplotlib.pyplot as plt

x_train_2016_processed = preprocessing.scale(data.x_train_2016)
x_test_2016_processed = preprocessing.scale(data.x_test_2016)
x_train_twitter_processed = preprocessing.scale(data.x_train_tweets)
x_test_twitter_processed = preprocessing.scale(data.x_test_tweets)

def svm_hyperparameter_tuning(x_train, y_train, scoring='f1_micro'):
    #svm = model_selection.RandomizedSearchCV(estimator = SVC(), param_distributions={'C':[0.01,0.05,0.1,0.25,0.5,0.75,0.9,1.0,2.0,3.0,4.0,5.0,7.5,10.0,20.0],'kernel':['linear','poly','rbf','sigmoid'],'degree':[2,3,4,5],'gamma':['scale','auto']},n_iter=200,n_jobs=-1,scoring=scoring)
    svm = model_selection.GridSearchCV(estimator = SVC(), param_grid={'C':[0.01,0.05,0.1,0.25,0.5,0.75,0.9,1.0,2.0,3.0,4.0,5.0,7.5,10.0,20.0],'kernel':['linear','poly','rbf','sigmoid'],'degree':[2,3,4,5],'gamma':['scale','auto']},n_jobs=-1,scoring=scoring)
    svm.fit(x_train, y_train)
    print(svm.best_params_)

    return svm.best_params_

#hp_2016 = svm_hyperparameter_tuning(x_train_2016_processed, data.y_train_2016)
#hp_2016 = {'kernel': 'poly', 'gamma': 'scale', 'degree': 2, 'C': 0.25}
#no timeout {'kernel': 'rbf', 'gamma': 'scale', 'degree': 4, 'C': 3.0}
hp_2016 = {'C': 3.0, 'degree': 2, 'gamma': 'scale', 'kernel': 'rbf'} #from grid search, no timeout

#hp_tweets = svm_hyperparameter_tuning(x_train_twitter_processed, data.y_train_tweets)
#hp_tweets = {'kernel': 'rbf', 'gamma': 'auto', 'degree': 5, 'C': 3.0}
#no timeout {'kernel': 'rbf', 'gamma': 'auto', 'degree': 2, 'C': 10.0}
hp_tweets= {'C': 10.0, 'degree': 2, 'gamma': 'scale', 'kernel': 'rbf'} #from grid search, no timeout

svm_2016 = SVC(C=hp_2016['C'],kernel=hp_2016['kernel'],gamma=hp_2016['gamma'],degree=hp_2016['degree'])
analysis.plot_learning_curve(svm_2016, "Learning Curves for SVMs on 2016 Election Data", data.counties, data.dem, cv=[[np.asarray(data.train_2016_indices), np.asarray(data.test_2016_indices)]]).show()
analysis.plot_learning_curve(svm_2016, "Learning Curves for SVMs on 2016 Election Data (Cross-Validation)", data.counties, data.dem).show()
svm_2016.fit(x_train_2016_processed,data.y_train_2016)
analysis.plot_cm(svm_2016, x_test_2016_processed, data.y_test_2016, normalize='true')
#print("2016 data: ")
#analysis.print_metrics(svm_2016,x_train_2016_processed, data.y_train_2016, x_test_2016_processed, data.y_test_2016)

svm_tweets = SVC(C=hp_tweets['C'],kernel=hp_tweets['kernel'],gamma=hp_tweets['gamma'],degree=hp_tweets['degree'])
analysis.plot_learning_curve(svm_tweets, "Learning Curves for SVMs on Twitter Data", data.tweets, data.bot, cv=[[np.asarray(data.train_tweets_indices), np.asarray(data.test_tweets_indices)]]).show()
analysis.plot_learning_curve(svm_tweets, "Learning Curves for SVMs on Twitter Data (Cross-Validation)", data.tweets, data.bot).show()
svm_tweets.fit(x_train_twitter_processed,data.y_train_tweets)
analysis.plot_cm(svm_tweets, x_test_twitter_processed, data.y_test_tweets, normalize='true')
#print("Tweet data: ")
#analysis.print_metrics(svm_tweets,x_train_twitter_processed, data.y_train_tweets, x_test_twitter_processed, data.y_test_tweets)
