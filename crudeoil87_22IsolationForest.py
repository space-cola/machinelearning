# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 16:09:24 2022

@author: space-cola
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

dataSheet = pd.read_csv("EuropeBSP.csv") #read the datasheet from .csv
dataSheet = dataSheet.to_numpy() #cnovert the datasheet to a numpy array for processing with ML

def machineTrain():
    global features #make the features a global variable as they will be used to plot
    features = dataSheet[:,-1] #extract the features (the price of oil per barrel) from the datasheet
    mlModel = IsolationForest(random_state=0, contamination = 0.02) #use of IsolationForest and a low contamination value to reduce anomaly sensitivity
    features = features.reshape(-1,1) #one feature means a reshape into a 2D array
    mlModel.fit(features) #use of fit to train the model (an unsupervised ML algorithm) with the features
    global modelPredictions #declare predictions as global as they will be used for plot too
    modelPredictions = mlModel.predict(features) #extract the model's anomaly predictions and place them in a variable

machineTrain()

def plot():
    years = np.linspace(2022, 1987, np.shape(features)[0]) #create the data for the plot graph, flatten the features to 1D again
    plt.plot(years,features) #plot the initial graph
    anomalies=np.where(modelPredictions == -1) #use of np.where to find the anomalies which are listed as -1 by the model
    plt.plot(years[anomalies],features[anomalies],marker='x', linestyle = '') #plot the anomalies on the graph

plot()


