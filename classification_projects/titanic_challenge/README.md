# Titanic Survival Prediction with Machine Learning

## Objective
The objective of this project is to develop and train a Machine Learning model capable of predicting weather a passenger survives the tragedy of the Titanic or not.
This is achieved by training a classification algorithm with a dataset which contains a label column ('Survived') and feature columns that describe each passenger.
These features contain information such as the age, the fare paid for the ticket, the name of the passenger, the cabin and the sex.

## Process
The development of the solution for this project began with an EDA (Exploratory Data Analysis). During this analysis it is of essence to explore and understand the 
data we are given tot rain the model. Based on this analysis we may determine some features useless, derive new features from existing ones and decide how to encode non-numeric ones.

After the EDA I moved forth with preprocessing as a set of functions that can be grouped in a single pipeline. This preprocessing functions have two objectives:
1. To clean up data to make it ready for feature engineering
2. To fill in incomplete data

Moving to the feature engineering process. It is mainly about:
1. Encoding non-numeric features
2. Grouping and simplifying numeric features that contain wide ranges
3. Creating new features based on the currently available ones

Moving to model training. You will notice that in the notebook I train several models. That is due to the fact that I didn't know which classification algorithm would yield
the best results. In the pipeline and pipeline functions for Machine Learning I make use only of the Decision Tree algorithm as it is the most efficient (yielded highest accuracy consuming the least amount
of computational resources/the least complex algorithm). This pipeline focuses on:
1. Training a Decision Tree model
2. Testing its accuracy against the training data
3. Returning the trained model

## Model Deployment & API Creation
In order to turn this model into a RESTful API the following process can be taken as an example
1. Load the model into the application using the pickle Python package.
2. Create an endpoint to receive the data to perform a single prediction. The object to be recieved can be a simple dictionary.
3. Optionally, the data can be preprocessed and features can be engineered using very similar functions to the ones described above.
4. Send this data into the `model.predict()` function. (See implementation in the ML pipeline)
5. Return the prediction.

To create the API please refer to [this guide.](https://medium.com/analytics-vidhya/serve-a-machine-learning-model-using-sklearn-fastapi-and-docker-85aabf96729b) 
