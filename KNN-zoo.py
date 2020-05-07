# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:06:34 2020

@author: patel
"""


import pandas as pd
import numpy as np
import math
import operator
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

data = pd.read_csv("C:\\Users\\patel\\Downloads\\zoo.csv",index_col='animal name')

print(data)

colnames=data.columns
colnames

target=colnames[8]
y=data[target]
y

predictors=colnames[0:8]
x =data[predictors]
x

x.shape,y.shape


from datetime import datetime

def evaluate_model_cv(model, X=x, y=y):
    start = datetime.now()
    kfold = KFold(n_splits=10, random_state=42)
    results = cross_val_score(model, X, y, cv=kfold,
                              scoring='accuracy', verbose=1)
    end = datetime.now()
    elapsed = int((end - start).total_seconds() * 1000)
    score = results.mean() * 100
    stddev = results.std() * 100
    print(model, '\nCross-Validation Score: %.2f (+/- %.2f) [%5s ms]' % \
          (score, stddev, elapsed))
    return score, stddev, elapsed

    
def fine_tune_model(model, params, X=x, y=y):
  print('\nFine Tuning Model:')
  print(model, "\nparams:", params)
  kfold = KFold(n_splits=10, random_state=42)
  grid = GridSearchCV(estimator=model, param_grid=params,
                      scoring='accuracy', cv=kfold, verbose=1)
  grid.fit(X, y)
  print('\nGrid Score: %.2f %%' % (grid.best_score_ * 100))
  print('Best Params:', grid.best_params_)
  return grid

#K-Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)

evaluate_model_cv(model)

params = {'n_neighbors':[1, 3, 5, 7, 9]}
fine_tune_model(model,params)

models = []

models.append(('KNN', KNeighborsClassifier(n_neighbors=3)))

results = []

names = []

scores = []

stddevs = []

times = []

for name, model in models:
    score, stddev, elapsed = evaluate_model_cv(model, X=x, y=y)
    results.append((score, stddev))
    names.append(name)
    scores.append(score)
    stddevs.append(stddev)
    times.append(elapsed)
    
results_df = pd.DataFrame({
'Model': names,
'Score': scores,
'Std Dev': stddevs,
'Time (ms)': times})

results_df.sort_values(by='Score', ascending=False)