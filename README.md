

# Loan Prediction Neural Network

This repository hosts a machine learning project that predicts loan approval status using a neural network built with Keras.

## Project Structure

The project is organized as follows:

- `data/`: Contains the dataset files used for training and validation.
- `myenv/`: A virtual environment directory for the project.
- `notebook/`: Jupyter notebooks with the preprocessing and model building code.
- `source/`: Python scripts for model building and data preprocessing.
- `requirements.txt`: A list of Python packages required for the project.

## Getting Started

To set up this project, ensure you have Python installed on your machine, then follow these steps:

1. Clone this repository to your local machine.
2. Navigate to the cloned directory.
3. Create a virtual environment:

   ```
   python -m venv myenv
   ```

4. Activate the virtual environment:

   - On Unix or MacOS, run:

     ```
     source myenv/bin/activate
     ```

   - On Windows, run:

     ```
     myenv\Scripts\activate
     ```

5. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

## Data Preprocessing

The data preprocessing steps include:

1. Importing necessary libraries like pandas and numpy.
2. Reading the loan prediction data.
3. Filling missing values with mode for categorical data and mean for continuous data.
4. Converting categorical features to numerical format.
5. Normalizing the features to a range of 0 to 1.
6. Saving the processed data for model training.

You can run the preprocessing notebook `loan_prediction_preprocessing.ipynb` to prepare the data.

## Neural Network Model

The steps for building and training the neural network using Keras are:

1. Importing required libraries such as pandas, numpy, scikit-learn, matplotlib, tensorflow, and keras.
2. Loading the pre-processed dataset.
3. Dropping unnecessary columns like `Loan_ID`.
4. Splitting the data into training and validation sets.
5. Defining the neural network architecture with input, hidden, and output layers.
6. Compiling the model with an appropriate loss function and optimizer.
7. Training the model on the training data and evaluating it on the validation set.
8. Visualizing the model's training performance.

Run the model building notebook `loan_prediction_neural_network.ipynb` to train and evaluate the model.

## Visualization

After training the model, you can visualize the performance by plotting the accuracy and loss of the model over epochs, which helps in understanding how well the model has learned from the training process.

## Requirements

The project depends on the following Python packages:

- pandas
- numpy
- scikit-learn
- matplotlib
- tensorflow
- keras

