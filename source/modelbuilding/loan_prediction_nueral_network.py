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

import tensorflow as tf
print(tf.__version__)

### a. Create a model

# importing the sequential model
from keras.models import Sequential

### b. Defining different layers
# importing different layers from keras
from keras.layers import InputLayer, Dense 
# number of input neurons
X_train.shape

print('No of input nuerons',X_train.shape)

# number of features in the data
X_train.shape[1]

print('no of features', X_train.shape[1])

# defining input neurons
input_neurons = X_train.shape[1]

print(input_neurons)

# number of output neurons

# since loan prediction is a binary classification problem, we will have single neuron in the output layer 

# define number of output neurons
output_neurons = 1

# number of hidden layers and hidden neurons
# It is a hyperparameter and we can pick the 
#hidden layers and hidden neurons 
#on our own
number_of_hidden_layers = 2
neuron_hidden_layer_1 = 10
neuron_hidden_layer_2 = 5

# activation function of different layers

# for now I have picked relu as an activation function for hidden layers, you can change it as well
# since it is a binary classification problem, I have used sigmoid activation function in the final layer

# defining the architecture of the model
model = Sequential()
model.add(InputLayer(input_shape=(input_neurons,)))
model.add(Dense(units=neuron_hidden_layer_1, activation='relu'))
model.add(Dense(units=neuron_hidden_layer_2, activation='relu'))
model.add(Dense(units=output_neurons, activation='sigmoid'))

# summary of the model
model.summary()

print(model.summary())


# number of parameters between input and first hidden layer
p_ifhl = input_neurons*neuron_hidden_layer_1
noofnuerons = input_neurons
hiddenlayer1nueron = neuron_hidden_layer_1

print('No of nuerons',noofnuerons)
print('hiddenlayer1neron',neuron_hidden_layer_1)
print('number of parameters between input and first hidden layer',p_ifhl)

# number of parameters between input and first hidden layer

# adding the bias for each neuron of first hidden layer

input_neurons*neuron_hidden_layer_1 + 10

#Bias is 1 for 1 layer  and for 10 hidden layers, it will be 10 * 1
bias = 10
print('no of parameters between ip and first hidden layer and bias',input_neurons*neuron_hidden_layer_1 + bias)

# number of parameters between first and second hidden layer
hiddenlayer1_nueron = neuron_hidden_layer_1
hiddenlayer2_nueron = neuron_hidden_layer_2

print('hiddenlayer1_nueron',hiddenlayer1_nueron)
print('hiddenlayer2_nueron',hiddenlayer2_nueron)
#Bias is 1 for 1 layer and for 5 hidden layers , it will be 5 *1 

neuron_hidden_layer_1*neuron_hidden_layer_2 + 5
print('no of parameters between first  and second hidden layer and bias',neuron_hidden_layer_1*neuron_hidden_layer_2 + 5)

#Number of parameters between second hidden and output layer

print('hiddenlayer2_nueron',neuron_hidden_layer_2)
print('output_neurons',output_neurons)

#Bias is 1 and for 1 layer, it will be 1 *1

print('no of parameters between second hiddena nd op layer',neuron_hidden_layer_2 * output_neurons + 1)

## Compiling the model (defining loss function, optimizer)
model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])

print(model.summary())
# Print the compilation details
print("Model Compilation Details:")
print(f"Loss Function: {model.loss}")
print(f"Optimizer: {model.optimizer}")
print(f"Metrics: {model.metrics}")

#Training the model
model_history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50)

## 6. Evaluating model performance on validation set
# getting predictions for the validation set
prediction = model.predict(X_test)

# Convert probabilities to class labels
prediction = (prediction > 0.5).astype("int32")

# calculating the accuracy on validation set
accscore = accuracy_score(y_test, prediction)
print('Accuracy Score' , accscore)

### Visualizing the model performance

# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# summarize history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
