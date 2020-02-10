from sklearn.ensemble import AdaBoostClassifier as ada
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import model_selection
import data
import math
import analysis
import numpy as np
import matplotlib.pyplot as plt

def ada_hyperparameter_tuning(x_train, y_train, scoring='f1_micro'):
    ada_model = model_selection.RandomizedSearchCV(estimator = ada(random_state=1), param_distributions={'base_estimator':[DecisionTreeClassifier(max_depth=1),DecisionTreeClassifier(max_depth=2),DecisionTreeClassifier(max_depth=3),DecisionTreeClassifier(max_depth=4),DecisionTreeClassifier(max_depth=5)],'n_estimators':[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,200,250],'learning_rate':[0.01,0.05,0.1,0.25,0.5,0.75,0.9,1.0,1.5,2.0,2.5,3.0,5.0,7.0,10.0]},n_iter=100,n_jobs=-1,scoring=scoring)
    ada_model.fit(x_train, y_train)
    print(ada_model.best_params_)

    return ada_model.best_params_

hp_2016 = ada_hyperparameter_tuning(data.x_train_2016, data.y_train_2016)
hp_tweets = ada_hyperparameter_tuning(data.x_train_tweets, data.y_train_tweets)

#hp_2016 = {'n_estimators': 80, 'learning_rate': 0.5, 'base_estimator': DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=1, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='best')}
#hp_tweets = {'n_estimators': 30, 'learning_rate': 0.1, 'base_estimator': DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=4, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='best')}

ada_2016 = ada(base_estimator=hp_2016['base_estimator'],n_estimators=hp_2016['n_estimators'],learning_rate=hp_2016['learning_rate'],random_state=1)
analysis.plot_learning_curve(ada_2016, "Learning Curves for Boosted Classifier on 2016 Election Data", data.counties, data.dem, cv=[[np.asarray(data.train_2016_indices), np.asarray(data.test_2016_indices)]]).show()
analysis.plot_learning_curve(ada_2016, "Learning Curves for Boosted Classifier on 2016 Election Data (Cross-Validation)", data.counties, data.dem).show()
ada_2016.fit(data.x_train_2016,data.y_train_2016)
analysis.plot_cm(ada_2016, data.x_test_2016, data.y_test_2016, normalize='true')
print("2016 data: ")
analysis.print_metrics(ada_2016,data.x_train_2016, data.y_train_2016, data.x_test_2016, data.y_test_2016)

ada_tweets = ada(base_estimator=hp_tweets['base_estimator'],n_estimators=hp_tweets['n_estimators'],learning_rate=hp_tweets['learning_rate'],random_state=1)
analysis.plot_learning_curve(ada_tweets, "Learning Curves for Boosted Classifier on Twitter Data", data.tweets, data.bot, cv=[[np.asarray(data.train_tweets_indices), np.asarray(data.test_tweets_indices)]]).show()
analysis.plot_learning_curve(ada_tweets, "Learning Curves for Boosted Classifier on Twitter Data (Cross-Validation)", data.tweets, data.bot).show()
ada_tweets.fit(data.x_train_tweets,data.y_train_tweets)
analysis.plot_cm(ada_tweets, data.x_test_tweets, data.y_test_tweets, normalize='true')
print("Tweet data: ")
analysis.print_metrics(ada_tweets,data.x_train_tweets, data.y_train_tweets, data.x_test_tweets, data.y_test_tweets)
