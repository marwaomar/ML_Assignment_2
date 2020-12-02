#Loading Data
import csv
import pandas as pd
import numpy as np

#Uncomment following line for multivariate data:
# data=np.loadtxt('multivariateData.dat',delimiter=',')

#Uncomment following line for univariate data:
data=np.loadtxt('univariateData.dat',delimiter=',')

def matrixMulti(a, X, n):
    h = np.ones((X.shape[0],1))
    theta = a.reshape(1,n+1)
    for i in range(0,X.shape[0]):
        h[i] = float(np.matmul(a, X[i]))
    h = h.reshape(X.shape[0])
    return h
def computeCost(X,y,h):
    cost = (1/X.shape[0]) * 0.5 * sum(np.square(h - y))
    return cost
def gradientDescent(a, alpha, num_iters, h, X, y, n):
    cost = np.ones(num_iters)
    for i in range(0,num_iters):
        a[0] = a[0] - (alpha/X.shape[0]) * sum(h - y)
        for j in range(1,n+1):
            a[j] = a[j] - (alpha/X.shape[0]) * sum((h-y) * X.transpose()[j])
        h = matrixMulti(a, X, n)
        cost[i] = computeCost(X,y,h)
    theta = a.reshape(1,n+1)
    return a, cost
def fit(X, y, alpha=0.0001, num_iters=100000):
    n = X.shape[1]
    one_column = np.ones((X.shape[0],1))
    X = np.concatenate((one_column, X), axis = 1)
    # initializing the parameter vector...
    a = np.zeros(n+1)
    # matrixMulti calculation....
    h = matrixMulti(a, X, n)
    # returning the optimized parameters by Gradient Descent...
    a, cost = gradientDescent(a,alpha,num_iters,h,X,y,n)
    return a

#Splitting Data
from sklearn.model_selection import train_test_split
X=data[:,:-1]
y=data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def scaling(dataToScale):
    mean = np.ones(dataToScale.shape[1])
    std = np.ones(dataToScale.shape[1])
    for i in range(0, dataToScale.shape[1]):
        mean[i] = np.mean(dataToScale.transpose()[i])
        std[i] = np.std(dataToScale.transpose()[i])
        for j in range(0, dataToScale.shape[0]):
            dataToScale[j][i] = (dataToScale[j][i] - mean[i])/std[i]

    return dataToScale
X_train=scaling(X_train)
a = fit(X_train, y_train)
print('a values ',a)

def vectorization(data):
    data=scaling(data)
    data = np.concatenate((np.ones((data.shape[0],1)), data),axis = 1)
    return data
def predict(predictiondata):
    predictiondata = vectorization(predictiondata)  
    predictions = matrixMulti(a,predictiondata, predictiondata.shape[1] - 1)
    return predictions

y_predict=predict(X_test)
print('Actual Data:',y_test)
print('Predicted Data:',y_predict)

def EvaluatePerformance(prediction,actual):
#Evaluating preformance by using R^2 method
    sst = np.sum((actual-actual.mean())**2)
    ssr = np.sum((prediction-actual)**2)
    r2 = 1-(ssr/sst)
    return(r2)

Accuracy=EvaluatePerformance(y_predict,y_test)
print('Accuracy:',Accuracy)

# Comparing between this method and the original model from sklearn:
from sklearn.linear_model import LinearRegression
clf = LinearRegression()
reg=clf.fit(X_train, y_train)

pred=reg.predict(X_test)
print('Implementation of Code',y_predict)
print('Library Model Results',pred)
print('Accuracy:',EvaluatePerformance(pred,y_test))