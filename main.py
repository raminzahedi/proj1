# -*- coding: utf-8 -*-
"""
Student name: Ramin Zahedi Darshoori
Banner ID: 800606858
NMSU
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Perceptron import Perceptron
from Adaline import Adaline
from SGD import SGD

#Assigning the Model from user input
if sys.argv[1]=='perceptron':
    model=Perceptron()
elif sys.argv[1]=='adaline':
    model=Adaline()
elif sys.argv[1]=='SGD':
    model=SGD()
else:
    print("invalid model name")
    exit()

#Reading the data from user input
try:
    df = pd.read_csv(sys.argv[2], header=None)
except:
    print("invalid dataset")
    exit()

#extracting features and class labels from the dataset
if sys.argv[2]=='iris.csv':
    X = df.iloc[:, 0:4].values
    y = df.iloc[:, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    
elif sys.argv[2]=='BreastCancer.csv':
    X = df.iloc[:, 0:30].values
    y = df.iloc[:, 30].values
    y = np.where(y == 'M', -1, 1)
else:
    print("invalid datase")
    exit()
    

#data normalization
X_std = np.copy(X)
if sys.argv[2]=='iris.csv':
    for i in range(4):
        X_std[:,i] = (X[:,i] - X[:,i].mean()) / X[:,i].std()
elif sys.argv[2]=='BreastCancer.csv':
    for i in range(30):
        X_std[:,i] = (X[:,i] - X[:,i].mean()) / X[:,i].std()


if sys.argv[1]=='perceptron':
    model.fit(X_std, y)
    plt.plot(range(1, len(model.errors_) + 1), model.errors_, marker='o')
    print("number of errors is: " + " " + str(model.errors_[-1]))
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.show()
    exit()
    
    
elif sys.argv[1]=='adaline':
    if sys.argv[2]=='iris.csv':
        model=Adaline(eta=0.0001)
    elif sys.argv[2]=='BreastCancer.csv':
        model=Adaline(eta=0.0000000001)
    model.fit(X_std, y)
    plt.plot(range(1, len(model.cost_) + 1), model.cost_, marker='o')
    print("the final sum squared error is: " + " " + str(model.cost_[-1]))
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')
    plt.show()
    exit()
    
    
elif sys.argv[1]=='SGD':
    if sys.argv[2]=='iris.csv':
        model=SGD(eta=0.0001)
    elif sys.argv[2]=='BreastCancer.csv':
        model=SGD(eta=0.0000000001)
    model.fit(X_std, y)
    plt.plot(range(1, len(model.cost_) + 1), model.cost_, marker='o')
    print("the final Average Cost is: " + " " + str(model.cost_[-1]))
    plt.xlabel('Epochs')
    plt.ylabel('Average Cost')
    plt.show()
    exit()

