# -*- coding: utf-8 -*-
"""
Created on Mon May  4 11:54:11 2020

@author: patel
"""

# Importing libraries
import pandas as pd
import numpy as np
import math
import operator
import matplotlib.pyplot as plt


data = pd.read_csv("C:\\Users\\patel\\Downloads\\glass.csv",index_col='RI')
print(data)

colnames=data.columns
colnames
###
target=colnames[8]
y=data[target]
y

predictors=colnames[0:8]
x =data[predictors]
x

x.shape,y.shape

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x_scaled=scaler.fit_transform(x)
x=pd.DataFrame(x_scaled, columns=colnames[0:8])
x

#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
x_train
y_train


######
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import f1_score
clf=KNN(n_neighbors=1)

clf.fit(x_train,y_train)

#predict
pred=clf.predict(x_test)
pred

k=f1_score(y_test,pred,average='micro')
k


##elbow curve
def Elbow(K):
    #initilizing empty list
    test_error=[]
    
    #traing model for every value of k
    for i in K:
        #Instance on KNN
        clf=KNN(n_neighbors = i)
        clf.fit(x_train,y_train)
        #appending F1 scores to empty lisy calc. using pred
        tmp=clf.predict(x_test)
        tmp=f1_score(tmp,y_test,average='weighted')
        error=1-tmp
        test_error.append(error)
    return test_error 

k=range(1,10,2)

test=Elbow(k) 

#plot
plt.plot(k,test)
plt.xlabel('K neigbhors')
plt.ylabel('Teset error')
plt.title('Elbow curve')
# elbow is at 3

clf=KNN(n_neighbors=3)

clf.fit(x_train,y_train)

#predict
pred=clf.predict(x_test)
pred

k=f1_score(y_test,pred,average='micro')
k ## 0.697