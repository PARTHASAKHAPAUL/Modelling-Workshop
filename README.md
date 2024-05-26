# Modelling-Workshop
## Forecasting Daily Stock Market using Artificial Neural Network(Compairing the results by FFNN,RNN,LSTM)

## Overview
This project demonstrates the process of predicting stock market returns using FeedForward Neural Network(FFNN), Recurrent Neural Network(RNN), Long-Short Term Memory Neural Network. The steps include loading and preparing the data, normalizing it, creating sequences, splitting the data into training and testing sets, building the FFNN,RNN,LSTM models, and training the model. The goal is to predict the direction of stock returns based on historical data.

## Steps

### Step 1: Load the Data
The data is loaded from an Excel file into a pandas DataFrame. The 'Date' column is set as the index, and a preliminary inspection of the data is done by printing the first few rows.

### Step 2: Data Preprocessing
#### a. Modify Return Directions
In the dataset, the 'Returns direction' column contains binary values indicating the direction of stock returns. We replace -1 with 0 to facilitate binary classification.

#### b. Normalize the Data
To standardize the features, the StandardScaler from sklearn is used. This ensures that each feature has a mean of 0 and a standard deviation of 1, making the training process more efficient and stable.

### Step 3: Create Sequences
RNNs require sequential data as input. A function is defined to create sequences of a specified length (`n_steps`) from the dataset. Each sequence consists of a set of feature values for the past `n_steps` days, and the target value is the return direction on the next day.

### Step 4: Split the Data
The dataset is split into training and testing sets with a specified ratio (e.g., 80-20 split). This helps in evaluating the model's performance on unseen data.

### Step 5: One-Hot Encode Labels
The return direction labels are one-hot encoded to convert them into a format suitable for classification tasks. This process transforms the binary labels into vectors.

### Step 6: Build the Model
Flexible FFNN is implemented from scratch using only Numpy. The model consists of multiple hidden layers with configurable units and activation functions, followed by a dense output layer with a specified activation function. Different optimizers can be selected for training the model, with the option to configure learning rate, momentum, and other hyperparameters.

Flexible RNN and LSTM models are constructed using TensorFlow's Keras API. These models consist of multiple RNN layers with configurable units and activation functions, followed by a dense output layer with a specified activation function. Different optimizers can be selected for training the model, with the option to configure learning rate, momentum, and other hyperparameters.

### Step 7: Compile and Summarize the Model
The models are compiled with a chosen optimizer and loss function. The summary of the models are printed to provide an overview of its architecture, including the number of parameters and the structure of each layer.

### Step 8: Evaluation
#### a. Predicting the Next Step
The RNN and LSTM models predict the next step using the latest `n_steps` data points. It prints the predicted class based on the model's output.

#### b. Mean Squared Error Calculation
The mean squared error between y_test and predicted_class is calculated to evaluate the model's performance.

#### c. Confusion Matrix and Classification Report
A confusion matrix, classification report, and model accuracy are generated to assess the model's performance.

#### d. Visualization
A comparison of actual and predicted classes is visualized using a line plot.

## Key Points
- **Data Normalization**: Essential for stable and efficient training.
- **Sequence Creation**: Transforms the data into a suitable format for RNNs.
- **One-Hot Encoding**: Converts categorical labels into a numerical format for classification.
- **Model Flexibility**: The architectures and hyperparameters of the FFNN,RNN,LSTM models are configurable, allowing experimentation with different setups.
- **Evaluation**: The models are trained on the training set and evaluated on the test set to assess its performance.

## Requirements
- Python
- numpy
- pandas
- scikit-learn
- TensorFlow

## Conclusion
This project provides a comprehensive approach to stock market prediction using `LSTMs`. By following the outlined steps, one can preprocess data, build, and train a LSTM model to predict stock market returns. The flexibility in model configuration allows for experimentation and optimization to achieve better predictive performance.

