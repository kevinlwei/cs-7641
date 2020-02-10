from sklearn import tree
from sklearn import metrics
from sklearn import model_selection
import matplotlib.pyplot as plt
import math
import data
import numpy as np
from analysis import plot_learning_curve
from analysis import print_metrics

def dt_hyperparameter_tuning(x_train, y_train, scoring='f1_micro'):
    best_tree = model_selection.GridSearchCV(estimator = tree.DecisionTreeClassifier(), param_grid={'min_samples_leaf':list(range(1,int(math.floor(len(x_train.index) / 20)))), 'max_depth':list(range(1,len(x_train.columns)))}, cv=max(2,int(math.floor(len(x_train.index) / 1000))),scoring=scoring)
    best_tree.fit(x_train, y_train)
    print(best_params_)

    return best_tree.best_params_

def decision_tree_f1_vs_depth(x_train, y_train, x_test, y_test, depth=50, pruning=10):
    testing_f1s = []
    training_f1s = []

    for i in range(1,depth):
        #random_state included for reproducibility
        clf = tree.DecisionTreeClassifier(criterion='entropy',splitter='best',max_depth=i, min_samples_leaf=pruning,random_state=1)
        clf.fit(x_train,y_train)
        y_test_hat = clf.predict(x_test)
        testing_f1s.append(metrics.f1_score(y_test,y_test_hat))

        y_train_hat = clf.predict(x_train)
        training_f1s.append(metrics.f1_score(y_train,y_train_hat))

    print("F1")
    print(max(testing_f1s))
    print(testing_f1s.index(max(testing_f1s)))
    plt.plot(range(1,depth), testing_f1s, 'd-', c='m', label='F1 Scores on Testing Data')
    plt.plot(range(1,depth), training_f1s, 'd-', c='c', label='F1 Scores on Training Data')
    plt.ylabel('F1 Scores on the Decision Tree Model')
    plt.xlabel('Max Tree Depth')

    plt.legend(loc='best',fontsize='small')
    plt.show()

def decision_tree_precision_vs_depth(x_train, y_train, x_test, y_test, depth=50, pruning=10):
    testing_f1s = []
    training_f1s = []

    for i in range(1,depth):
        #random_state included for reproducibility
        clf = tree.DecisionTreeClassifier(criterion='entropy',splitter='best',max_depth=i, min_samples_leaf=pruning,random_state=1)
        clf.fit(x_train,y_train)
        y_test_hat = clf.predict(x_test)
        testing_f1s.append(metrics.precision_score(y_test,y_test_hat))

        y_train_hat = clf.predict(x_train)
        training_f1s.append(metrics.precision_score(y_train,y_train_hat))

    print("Precision")
    print(max(testing_f1s))
    print(testing_f1s.index(max(testing_f1s)))
    plt.plot(range(1,depth), testing_f1s, 'd-', c='m', label='Precision Scores on Testing Data')
    plt.plot(range(1,depth), training_f1s, 'd-', c='c', label='Precision Scores on Training Data')
    plt.ylabel('Precision Scores on the Decision Tree Model')
    plt.xlabel('Max Tree Depth')

    plt.legend(loc='best',fontsize='small')
    plt.show()

#adapted from https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py
def decision_tree_alpha_pruning(x_train, y_train, x_test, y_test):
    clf = tree.DecisionTreeClassifier(random_state=1)
    path = clf.cost_complexity_pruning_path(x_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")

    clfs = []
    for i in ccp_alphas:
        clf = tree.DecisionTreeClassifier(random_state=1, ccp_alpha=i)
        clf.fit(x_train, y_train)
        clfs.append(clf)
    print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
          clfs[-1].tree_.node_count, ccp_alphas[-1]))
    clfs = clfs[:-1]

    ccp_alphas = ccp_alphas[:-1]

    node_counts = [clf.tree_.node_count for clf in clfs]
    depth = [clf.tree_.max_depth for clf in clfs]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs alpha")
    ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title("Depth vs alpha")
    fig.tight_layout()
    fig.show()
    ax.show()

#best_params_2016 = {'max_depth': 9, 'min_samples_leaf': 6} #Best parameters per tuning on f1
best_params_2016 = dt_hyperparameter_tuning(data.x_train_2016, data.y_train_2016)
#retune hyperparams on precision
#best_params_2016 = dt_hyperparameter_tuning(data.x_train_2016, data.y_train_2016, scoring='precision')
decision_tree_f1_vs_depth(data.x_train_2016, data.y_train_2016, data.x_test_2016, data.y_test_2016, pruning=best_params_2016['min_samples_leaf'])
decision_tree_precision_vs_depth(data.x_train_2016, data.y_train_2016, data.x_test_2016, data.y_test_2016, pruning=best_params_2016['min_samples_leaf'])
best_tree = tree.DecisionTreeClassifier(criterion='entropy',splitter='best',max_depth=best_params_2016['max_depth'], min_samples_leaf=best_params_2016['max_depth'],random_state=1)
plot_learning_curve(best_tree, "Learning Curves for Decision Tree on 2016 Election Data", data.counties, data.dem, cv=[[np.asarray(data.train_2016_indices), np.asarray(data.test_2016_indices)]]).show()
#Cross-validation with 5 subsets
plot_learning_curve(best_tree, "Learning Curves for Decision Tree on 2016 Election Data (Cross-Validation)", data.counties, data.dem).show()
best_tree.fit(data.x_train_2016,data.y_train_2016)
tree.plot_tree(best_tree)
plt.show()
metrics.plot_confusion_matrix(best_tree, data.x_test_2016, data.y_test_2016, normalize='true').confusion_matrix
plt.show()
print(tree.export_text(best_tree, feature_names=list(data.x_test_2016.columns.values)))
print_metrics(best_tree,data.x_train_2016, data.y_train_2016, data.x_test_2016, data.y_test_2016)

#Best parameters per tuning: {'max_depth': 5, 'min_samples_leaf': 51}
best_params_tweets = dt_hyperparameter_tuning(data.x_train_tweets, data.y_train_tweets)
#print(best_params_tweets)
best_params_tweets = {'max_depth': 5, 'min_samples_leaf': 51}
decision_tree_f1_vs_depth(data.x_train_tweets, data.y_train_tweets, data.x_test_tweets, data.y_test_tweets, depth=len(data.x_train_tweets.columns), pruning=best_params_tweets['min_samples_leaf'])
best_tree = tree.DecisionTreeClassifier(criterion='entropy',splitter='best',max_depth=best_params_tweets['max_depth'], min_samples_leaf=best_params_tweets['max_depth'],random_state=1)
plot_learning_curve(best_tree, "Learning Curves for Decision Tree on Twitter Bot Data", data.tweets, data.bot, cv=[[np.asarray(data.train_tweets_indices), np.asarray(data.test_tweets_indices)]]).show()
Cross-validation with 5 subsets
plot_learning_curve(best_tree, "Learning Curves for Decision Tree on Twitter Bot Data (Cross-Validation)", data.tweets, data.bot).show()
best_tree.fit(data.x_train_tweets,data.y_train_tweets)
tree.plot_tree(best_tree, filled=True)
plt.show()
metrics.plot_confusion_matrix(best_tree, data.x_test_tweets, data.y_test_tweets, normalize='true').confusion_matrix
plt.show()
print_metrics(best_tree,data.x_train_tweets, data.y_train_tweets, data.x_test_tweets, data.y_test_tweets)

print(tree.export_text(best_tree, feature_names=list(data.x_test_tweets.columns.values)))
