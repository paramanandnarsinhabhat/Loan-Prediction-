# Steps to build a Neural Network using Keras
#Load the dataset
#Import the required libraries
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


# check version on sklearn
print('Version of sklearn:', sklearn.__version__)

# loading the pre-processed dataset
data = pd.read_csv('/Users/paramanandbhat/Downloads/Loan Prediction pre-processing 2/loan_prediction_data.csv')

# looking at the first five rows of the dataset
data.head()

print(data.head())

# checking missing values
data.isnull().sum()

print(data.isnull().sum())

# checking the data type
print(data.dtypes)

# removing the loan_ID since these are just the unique values
data = data.drop('Loan_ID', axis=1)

print('Dropping loan_id column')
print(data.dtypes)

# looking at the shape of the data
print(data.shape)

# separating the independent and dependent variables
# storing all the independent variables as X
X = data.drop('Loan_Status', axis=1)

# storing the dependent variable as y
y = data['Loan_Status']

# shape of independent and dependent variables
X.shape, y.shape

print('Independent variables' , X.shape)
print('Dependent variables' , y.shape)

#Create training andvalidation set
# stratify will make sure that the distribution of classes in train and validation set it similar
# random state to regenerate the same train and validation set
# test size 0.2 will keep 20% data in validation and remaining 80% in train set

X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=data['Loan_Status'],random_state=10,test_size=0.2)

# shape of training and validation set
(X_train.shape, y_train.shape), (X_test.shape, y_test.shape)

#Training sets
#Indpenedent vaiable
print('X_train Shape',X_train.shape)
#Dependent variable
print('y_train Shape',y_train.shape)

#Validation Sets
#Dependent variable
print('X_test shape',X_test.shape)
print('y_test shape',y_test.shape)

#Defining the arhitecture of the model
# checking the version of keras

print(keras.__version__)



