
from sklearn.ensemble import RandomForest
from sklearn.cross_validation import cross_val_score
import numpy as np

clf = RandomForest() #Initialize with whatever parameters you want to

clf = RandomForestClassifier(max_depth=2, random_state=0)

clf.fit(X, y)


print(tf.feature_importances_)


print(clf.predict([[0, 0, 0, 0]]))

np.mean(cross_val_score(clf, X_train, y_train, cv=10))

param_grid = {
                 'n_estimators': [5, 10, 15, 20],
                 'max_depth': [2, 5, 7, 9]
             }


from sklearn.grid_search import GridSearchCV

grid_clf = GridSearchCV(clf, param_grid, cv=10)
grid_clf.fit(X_train, y_train)

print(grid_clf.best_estimator_)

print(grid_clf.best_params_)

print(grid_clf.grid_scores_)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
iris = datasets.load_iris()


