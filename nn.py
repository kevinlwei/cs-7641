from sklearn.neural_network import MLPClassifier as mlp
from sklearn import metrics
from sklearn import model_selection
import data
import math
import analysis
import numpy as np
import matplotlib.pyplot as plt
import itertools

def mlp_hyperparameter_tuning(x_train, y_train, scoring='f1_micro'):
    layers_to_test = [x for x in itertools.product((10,20,30,40,50,60,70,80,90,100,110,120,130,140,150),repeat=1)]
    layers_to_test.extend([x for x in itertools.product((10,20,30,40,50,60,70,80,90,100,110,120,130,140,150),repeat=2)])
    layers_to_test.extend([x for x in itertools.product((10,20,30,40,50,60,70,80,90,100,110,120,130,140,150),repeat=3)])

    ann = model_selection.RandomizedSearchCV(estimator = mlp(solver='sgd',random_state=1), param_distributions={'activation':['identity', 'logistic', 'tanh', 'relu'], 'hidden_layer_sizes':layers_to_test, 'alpha': [0.001,0.005,0.010,0.025,0.050,0.075,0.01,0.25,0.5]}, n_iter=200, n_jobs=-1, cv=5,scoring=scoring)
    ann.fit(x_train, y_train)
    print(ann.best_params_)

    return ann.best_params_

#hp_2016 = mlp_hyperparameter_tuning(data.x_train_2016, data.y_train_2016)
#hp_2016 = {'hidden_layer_sizes': (100, 60, 150), 'activation': 'tanh'} #original hyperparameter tuning
hp_2016 = {'hidden_layer_sizes': (70, 30, 150), 'alpha': 0.01, 'activation': 'tanh'} #hyperparameter tuning w/ L2

#hp_tweets = mlp_hyperparameter_tuning(data.x_train_tweets, data.y_train_tweets)
#hp_tweets = {'hidden_layer_sizes': (100, 110, 60), 'activation': 'relu'} #original hyperparameter tuning
hp_tweets = {'hidden_layer_sizes': (100, 90, 60), 'alpha': 0.025, 'activation': 'relu'} #hyperparameter tuning w/ L2

nn_2016 = mlp(hidden_layer_sizes=hp_2016['hidden_layer_sizes'],alpha=hp_2016['alpha'],activation=hp_2016['activation'],solver='sgd',random_state=1)
#analysis.plot_learning_curve(nn_2016, "Learning Curves for Neural Net on 2016 Election Data", data.counties, data.dem, cv=[[np.asarray(data.train_2016_indices), np.asarray(data.test_2016_indices)]]).show()
#analysis.plot_learning_curve(nn_2016, "Learning Curves for Neural Net on 2016 Election Data (Cross-Validation)", data.counties, data.dem).show()
nn_2016.fit(data.x_train_2016,data.y_train_2016)
#analysis.plot_cm(nn_2016, data.x_test_2016, data.y_test_2016, normalize='true')
print("2016 data: ")
analysis.print_metrics(nn_2016,data.x_train_2016, data.y_train_2016, data.x_test_2016, data.y_test_2016)

nn_tweets = mlp(hidden_layer_sizes=hp_tweets['hidden_layer_sizes'],alpha=hp_tweets['alpha'],activation=hp_tweets['activation'],solver='sgd',random_state=1)
#analysis.plot_learning_curve(nn_tweets, "Learning Curves for Neural Net on Twitter Data", data.tweets, data.bot, cv=[[np.asarray(data.train_tweets_indices), np.asarray(data.test_tweets_indices)]]).show()
#analysis.plot_learning_curve(nn_tweets, "Learning Curves for Neural Net on Twitter Data (Cross-Validation)", data.tweets, data.bot).show()
nn_tweets.fit(data.x_train_tweets,data.y_train_tweets)
#analysis.plot_cm(nn_tweets, data.x_test_tweets, data.y_test_tweets, normalize='true')
#print("Tweet data: ")
analysis.print_metrics(nn_tweets,data.x_train_tweets, data.y_train_tweets, data.x_test_tweets, data.y_test_tweets)
