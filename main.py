import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import time

# sigmoid function
def sigmoid(x):
  return 1/ (1+np.exp(-x))


# cost function
def cost(predicted_y, actual_y, q):
  costs = 1/q *np.sum(-actual_y * np.log(predicted_y) - (1-actual_y)* np.log(1-predicted_y))
  return costs

# Note: dont really need cost list if not going to use
def logisticRegression(trainFeatures, trainLabels, learning_rate, epochs):
  q, x = trainFeatures.shape  # x is the number of features and q is sample quantity
  weights = np.zeros(x)  # initialize weights to 0
  cost_list=[]  # cost list
  for i in range(epochs):
    predicted_y = np.dot(trainFeatures.toarray(), weights) # predicted y or y_hat
    sig = sigmoid(predicted_y)  # sigmoid of predicted y
    delta = sig - trainLabels  # prediction error
    gradient = 1/q * np.dot(trainFeatures.toarray().T, delta)  # partial derivative of cost function
    #intercept_gradient = np.mean(delta)  # calculates mean of prediction error
    weights -=learning_rate * gradient  # update weights
    costs = cost(sig,trainLabels, q)  # calculate cost
    cost_list.append(costs)  # add cost to list
  return weights


# train set
dataset_train = pd.read_csv('dataset/train_amazon.csv')
dataset_train = dataset_train.sample(frac=0.05, random_state=45)
features_train = dataset_train['text']
trainLabels = dataset_train['label']
trainLabels = np.array(trainLabels, dtype=np.float64)

# test set
dataset_test = pd.read_csv('dataset/test_amazon.csv')
dataset_test = dataset_test.sample(frac=0.05, random_state=45)
features_test = dataset_test['text']
testLabels = dataset_test['label']
testLabels = np.array(dataset_test['label'], dtype=np.float64)

# bag of words
bag = CountVectorizer()
trainFeatures = bag.fit_transform(features_train)
testFeatures = bag.transform(features_test)

# training
start = time.time()
weights = logisticRegression(trainFeatures, trainLabels, 0.01, 25)
duration = time.time() - start
# testing
y_hat = np.round(sigmoid(np.dot(testFeatures.toarray(), weights)))

# metrics
accuracy = np.mean(y_hat==testLabels)*100
print(f"The efficiency of this model is based on the training speed.\nThe training speed is {duration:.2f} seconds")
print(f"The accuracy of the Logisitc Regression model is {accuracy:.2f}%")
