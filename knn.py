from sklearn.neighbors import KNeighborsClassifier as k
from sklearn import metrics
from sklearn import model_selection
import data
import math
import analysis
import numpy as np
import matplotlib.pyplot as plt

def knn_hyperparameter_tuning(x_train, y_train, scoring='f1_micro'):
    knn = model_selection.GridSearchCV(estimator = k(), param_grid={'n_neighbors':list(range(1,100))}, cv=max(2,int(math.floor(len(x_train.index) / 1000))), n_jobs=-1, scoring=scoring)
    knn.fit(x_train, y_train)
    print(knn.best_params_)

    return knn.best_params_

def knn_f1_vs_hyperparameters(x_train, y_train, x_test, y_test, max_neighbor, plot_title='F1 Score vs Number of Neighbors in KNN'):
    testing_f1s = []
    training_f1s = []

    for i in range(1,max_neighbor):
        #random_state included for reproducibility
        clf = k(n_neighbors=i)
        clf.fit(x_train,y_train)
        y_test_hat = clf.predict(x_test)
        testing_f1s.append(metrics.f1_score(y_test,y_test_hat))

        y_train_hat = clf.predict(x_train)
        training_f1s.append(metrics.f1_score(y_train,y_train_hat))

    plt.plot(range(1,max_neighbor), testing_f1s, 'd-', c='m', label='F1 Scores on Testing Data')
    plt.plot(range(1,max_neighbor), training_f1s, 'd-', c='c', label='F1 Scores on Training Data')
    plt.title(plot_title)
    plt.ylabel('F1 Scores on the KNN Model')
    plt.xlabel('Number of Neighbors in KNN Model')

    plt.legend(loc='best',fontsize='small')
    plt.tight_layout()
    plt.show()


hp_2016 = knn_hyperparameter_tuning(data.x_train_2016, data.y_train_2016)
hp_tweets = knn_hyperparameter_tuning(data.x_train_tweets, data.y_train_tweets)

knn_2016 = k(n_neighbors=hp_2016['n_neighbors'])
knn_f1_vs_hyperparameters(data.x_train_2016, data.y_train_2016, data.x_test_2016, data.y_test_2016, 100, "F1 Scores vs Neighbors on 2016 Election Data")
analysis.plot_learning_curve(knn_2016, "Learning Curves for KNN on 2016 Election Data", data.counties, data.dem, cv=[[np.asarray(data.train_2016_indices), np.asarray(data.test_2016_indices)]]).show()
analysis.plot_learning_curve(knn_2016, "Learning Curves for KNN on 2016 Election Data (Cross-Validation)", data.counties, data.dem).show()
knn_2016.fit(data.x_train_2016,data.y_train_2016)
analysis.plot_cm(knn_2016, data.x_test_2016, data.y_test_2016, normalize='true')
print("2016 data: ")
analysis.print_metrics(knn_2016,data.x_train_2016, data.y_train_2016, data.x_test_2016, data.y_test_2016)

knn_tweets = k(n_neighbors=hp_tweets['n_neighbors'])
knn_f1_vs_hyperparameters(data.x_train_tweets, data.y_train_tweets, data.x_test_tweets, data.y_test_tweets, 100, "F1 Scores vs Neighbors on Twitter Bot Data")
analysis.plot_learning_curve(knn_tweets, "Learning Curves for KNN on Twitter Data", data.tweets, data.bot, cv=[[np.asarray(data.train_tweets_indices), np.asarray(data.test_tweets_indices)]]).show()
analysis.plot_learning_curve(knn_tweets, "Learning Curves for KNN on Twitter Data (Cross-Validation)", data.tweets, data.bot).show()
knn_tweets.fit(data.x_train_tweets,data.y_train_tweets)
analysis.plot_cm(knn_tweets, data.x_test_tweets, data.y_test_tweets, normalize='true')
print("Tweet data: ")
analysis.print_metrics(knn_tweets,data.x_train_tweets, data.y_train_tweets, data.x_test_tweets, data.y_test_tweets)

for i in ['accuracy', 'average_precision', 'precision', 'recall', 'roc_auc']:
    print("*****************************************************************")
    print("SCORING MECHANISM")
    print(i)
    hp_2016 = knn_hyperparameter_tuning(data.x_train_2016, data.y_train_2016, scoring=i)
    hp_tweets = knn_hyperparameter_tuning(data.x_train_tweets, data.y_train_tweets, scoring=i)

    knn_2016 = k(n_neighbors=hp_2016['n_neighbors'])
    knn_tweets = k(n_neighbors=hp_tweets['n_neighbors'])

    print("2016 data")
    print("K value: ")
    print(hp_2016['n_neighbors'])
    analysis.print_metrics(knn_2016,data.x_train_2016, data.y_train_2016, data.x_test_2016, data.y_test_2016)

    print("Tweets data")
    print("K value: ")
    print(hp_tweets['n_neighbors'])
    analysis.print_metrics(knn_tweets,data.x_train_tweets, data.y_train_tweets, data.x_test_tweets, data.y_test_tweets)
