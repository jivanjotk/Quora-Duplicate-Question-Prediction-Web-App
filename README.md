# Quora Duplicate Question Prediction Using NLP

The **Quora Duplicate Question Prediction Web App** is a machine learning-powered web application designed to predict whether two questions asked on Quora are duplicates. The app leverages a trained Random Forest model and TF-IDF vectorization to classify pairs of questions as either duplicates or not. It provides an interactive user interface using **Streamlit**, making it easy for users to test the model by inputting their own questions.

This project aims to help users determine whether a question already exists on Quora, avoiding redundant content.

## UI Interface


![Screenshot 2025-01-20 213600](https://github.com/user-attachments/assets/9e6562c1-8930-4cba-aa03-86e346abc90d)

## App 
👉 [Click here to access the deployed web app](https://quora-duplicate-question-prediction-web-app.streamlit.app/)  


## Features

- **Text Input for Questions**: The user can input two questions to check if they are duplicates.
- **Model Prediction**: After entering the questions, the app predicts whether they are duplicates or not using a trained Random Forest model.
- **Interactive Web Interface**: The app is built with Streamlit for a clean and user-friendly interface.
- **Model Explanation**: The app uses TF-IDF vectorization to convert questions into numerical features, which are fed into the Random Forest model to make predictions.

## Technologies Used

- **Python**: The backend language for the app and model.
- **Streamlit**: Framework to deploy the interactive web app.
- **scikit-learn**: For model training (Random Forest Classifier and TF-IDF vectorizer).
- **Pickle**: Used to serialize the trained model for later use in prediction.


