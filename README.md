# Loan Prediction Neural Network Model

This repository contains a machine learning project for predicting loan approval status. The project uses a neural network model built with Keras and TensorFlow, and processes the data using pandas, NumPy, and scikit-learn.

## Project Structure

- `data/`
  - `loan_data.csv` - Initial loan dataset.
  - `loan_prediction_data.csv` - Pre-processed loan dataset ready for model training.
- `source/`
  - `modelbuilding/`
    - `loan_prediction_neural_network.py` - Script for building and training the neural network model.
  - `preprocessing/`
    - `loan_prediction_preprocessing.py` - Script for data preprocessing.
- `myenv/` - Python virtual environment directory.
- `.gitignore` - Specifies intentionally untracked files to ignore.
- `LICENSE` - The license file.
- `README.md` - The file you are currently reading.

## Environment Setup

To install the necessary libraries, use the following command:

```
pip install pandas numpy scikit-learn matplotlib tensorflow keras
```

Alternatively, you can use the `requirements.txt` file if provided.

## Data Preprocessing

The preprocessing steps include:

1. Filling missing values with mode for categorical data and mean for continuous data.
2. Converting categorical variables to numerical values.
3. Normalizing the data to bring all variables into the range 0 to 1.

## Neural Network Model

The neural network model steps include:

1. Loading the pre-processed dataset.
2. Dropping the `Loan_ID` column.
3. Splitting the dataset into training and validation sets.
4. Defining the architecture of the neural network with two hidden layers.
5. Compiling the model with the Adam optimizer and binary crossentropy as the loss function.
6. Training the model for 50 epochs.
7. Evaluating the model's performance on the validation set.
8. Visualizing the model's accuracy and loss over the epochs.

## Visualization of Model Training

The training process can be visualized by plotting the model's accuracy and loss, which helps in assessing the model's performance over time.

## Usage

To run the preprocessing and model training scripts, navigate to the respective directories and execute the Python scripts:

```
python loan_prediction_preprocessing.py
python loan_prediction_neural_network.py
```

Ensure that you are in the correct virtual environment before running these commands.

## Model Performance

The model's performance is evaluated using accuracy as the metric. The accuracy score is printed at the end of the model training process, and the training history is plotted using matplotlib.

