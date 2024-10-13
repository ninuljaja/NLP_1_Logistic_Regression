import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report
import time
from tqdm import tqdm

# sigmoid function
def sigmoid(x):
  return 1/ (1+np.exp(-x))


# Note: dont really need cost list if not going to use
def logisticRegression(trainFeatures, trainLabels, learning_rate, epochs):
  q, x = trainFeatures.shape  # x is the number of features and q is sample quantity
  weights = np.zeros(x)  # initialize weights to 0
  for i in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
    predicted_y = trainFeatures.dot(weights)  # Predicting y_hat (sparse matrix dot product)
    sig = sigmoid(predicted_y)  # sigmoid of predicted y
    delta = sig - trainLabels  # prediction error
    gradient = (1/q) * trainFeatures.T.dot(delta)   # partial derivative of cost function (sparse dot product)
    weights -=learning_rate * gradient  # update weights
  return weights


def printResult(duration, accuracy, conf_matrix, classif_report):
  print(f"The efficiency of this model is based on the training speed.\nThe training speed is {duration:.2f} seconds")
  print(f"The accuracy of the Logisitc Regression model is {accuracy:.2f}%")
  print("\nConfusion Matrix:")
  print(conf_matrix)
  print("\nClassification Report:")
  print(classif_report)


# train set
dataset_train = pd.read_csv('dataset/train_amazon.csv')
features_train = dataset_train['text']
trainLabels = dataset_train['label'].values.astype(np.float64)

# test set
dataset_test = pd.read_csv('dataset/test_amazon.csv')
features_test = dataset_test['text']
testLabels = dataset_test['label'].values.astype(np.float64)

#############################################################
# Model that uses the whole training dataset and            #
# CountVectorizer for tokenization (counts word frequencies)#
# learning rate = 0.01, epochs = 60                         #
#############################################################


# bag of words
bag = CountVectorizer()
trainFeatures = bag.fit_transform(features_train)
testFeatures = bag.transform(features_test)

# training
start = time.time()
weights = logisticRegression(trainFeatures, trainLabels, 0.01, 60)
duration = time.time() - start

# testing
predictLabel = np.round(sigmoid(testFeatures.dot(weights)))

# metrics
accuracy = np.mean(predictLabel==testLabels)*100

# Confusion Matrix
conf_matrix = confusion_matrix(testLabels, predictLabel)

# Label the confusion matrix using pandas
conf_matrix_with_labels = pd.DataFrame(conf_matrix, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])


# Classification Report
classif_report = classification_report(testLabels, predictLabel)


print("Model that uses the whole training dataset and CountVectorizer for tokenization (counts word frequencies)\nLearning rate = 0.01, epochs = 60")
printResult(duration, accuracy, conf_matrix_with_labels, classif_report)


#############################################################
# Model that uses the 80% of the training dataset and       #
# CountVectorizer for tokenization (counts word frequencies)#
# learning rate = 0.01, epochs = 60                         #
#############################################################

dataset_train = dataset_train.sample(frac=0.80, random_state=50)
features_train = dataset_train['text']
trainLabels = dataset_train['label'].values.astype(np.float64)

# bag of words
bag = CountVectorizer()
trainFeatures = bag.fit_transform(features_train)
testFeatures = bag.transform(features_test)

# training
start = time.time()
weights = logisticRegression(trainFeatures, trainLabels, 0.01, 60)
duration = time.time() - start

# testing
predictLabel = np.round(sigmoid(testFeatures.dot(weights)))

# metrics
accuracy = np.mean(predictLabel==testLabels)*100

# Confusion Matrix
conf_matrix = confusion_matrix(testLabels, predictLabel)

# Label the confusion matrix using pandas
conf_matrix_with_labels = pd.DataFrame(conf_matrix, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])


# Classification Report
classif_report = classification_report(testLabels, predictLabel)


print("Model that uses 80% of the training dataset and CountVectorizer for tokenization (counts word frequencies)\nLearning rate = 0.01, epochs = 60")
printResult(duration, accuracy, conf_matrix_with_labels, classif_report)



###################################################################
# Model that uses the 80% of the training dataset and             #
# TfidfVectorizer for feature extraction with max features = 7000 #
# (weighs word counts by their relative importance)               #
# learning rate = 0.01, epochs = 60                               #
###################################################################


# extracting features
features = TfidfVectorizer(max_features=7000)
trainFeatures = features.fit_transform(features_train)
testFeatures = features.transform(features_test)

# training
start = time.time()
weights = logisticRegression(trainFeatures, trainLabels, 0.01, 60)
duration = time.time() - start
# testing
predictLabel = np.round(sigmoid(testFeatures.dot(weights)))

# metrics
accuracy = np.mean(predictLabel==testLabels)*100


# Confusion Matrix
conf_matrix = confusion_matrix(testLabels, predictLabel)

# Label the confusion matrix using pandas
conf_matrix_with_labels = pd.DataFrame(conf_matrix, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])

# Classification Report
classif_report = classification_report(testLabels, predictLabel)

print("\nModel that uses the 80% of the training dataset and and "
      "TfidfVectorizer for feature extraction with max features = 7000 (weighs word counts by their relative importance)."
      "\nLearning rate = 0.01, epochs = 60")
printResult(duration, accuracy, conf_matrix_with_labels, classif_report)


###################################################################
# Model that uses the 80% of the training dataset and             #
# TfidfVectorizer for feature extraction with max features = 2100 #
# (weighs word counts by their relative importance)               #
# learning rate = 0.01, epochs = 60                               #
###################################################################


# extracting features
features = TfidfVectorizer(max_features=2100)
trainFeatures = features.fit_transform(features_train)
testFeatures = features.transform(features_test)

# training
start = time.time()
weights = logisticRegression(trainFeatures, trainLabels, 0.01, 60)
duration = time.time() - start

# testing
predictLabel = np.round(sigmoid(testFeatures.dot(weights)))

# metrics
accuracy = np.mean(predictLabel==testLabels)*100

# Confusion Matrix
conf_matrix = confusion_matrix(testLabels, predictLabel)

# Label the confusion matrix using pandas
conf_matrix_with_labels = pd.DataFrame(conf_matrix, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])

# Classification Report
classif_report = classification_report(testLabels, predictLabel)

print("\nModel that uses the 80% of the training dataset and and "
      "TfidfVectorizer for feature extraction with max features = 2100 (weighs word counts by their relative importance)."
      "\nLearning rate = 0.01, epochs = 60")
printResult(duration, accuracy, conf_matrix_with_labels, classif_report)



###################################################################
# Model that uses the 80% of the training dataset and             #
# TfidfVectorizer for feature extraction with max features = 2100 #
# (weighs word counts by their relative importance)               #
# learning rate = 0.1, epochs = 240                               #
###################################################################


# training
start = time.time()
weights = logisticRegression(trainFeatures, trainLabels, 0.1, 240)
duration = time.time() - start

# testing
predictLabel = np.round(sigmoid(testFeatures.dot(weights)))

# metrics
accuracy = np.mean(predictLabel==testLabels)*100

# Confusion Matrix
conf_matrix = confusion_matrix(testLabels, predictLabel)

# Label the confusion matrix using pandas
conf_matrix_with_labels = pd.DataFrame(conf_matrix, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])

# Classification Report
classif_report = classification_report(testLabels, predictLabel)

print("\nModel that uses the 80% of the training dataset and and "
      "TfidfVectorizer for feature extraction with max features = 2100 (weighs word counts by their relative importance)."
      "\nLearning rate = 0.1, epochs = 240")
printResult(duration, accuracy, conf_matrix_with_labels, classif_report)

