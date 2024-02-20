# ML Projects
# Internet Usage Prediction App

This is the README file for a Python application that predicts internet usage based on the year using a linear regression model.

## Features:

- Predicts internet usage for any year: Simply enter a year in the text field and click the "Predict Usage" button. The app will use the trained model to estimate the internet usage for that year.
- Easy to use: The interface is simple and intuitive, making it accessible to users without any technical knowledge.
- Built with PyQt5: The app is developed using the PyQt5 library, ensuring a native look and feel on different operating systems.

## Requirements:
- Python 3.6 or later
- PyQt5
- NumPy
- scikit-learn

## Instructions:

- Make sure you have the required libraries installed (instructions can be found online).
Run the main script main.py.
- Enter a year in the text field and click "Predict Usage".
- The predicted internet usage for that year will be displayed.

## Data Source:
The internet usage data used in this application is assumed to be stored in the Dataset.internet module. Please ensure this module exists and contains the necessary data arrays years and usage.

## Model Training:

The linear regression model is trained on the provided data during the initialization of the application. You can modify the training process within the MainWindow class if needed.

## Conclusion:

This application demonstrates a basic example of using machine learning to predict values based on historical data. It leverages Python libraries and Qt for a user-friendly interface. Feel free to explore and customize the code further to fit your specific needs!


# Crop Yield Prediction App

## Description:

This application provides a user-friendly interface to predict crop yield based on weather conditions using a trained linear regression model. The user can input values for rainfall, sunshine hours, and average temperature, and the app will return the predicted yield in tons per hectare. Additionally, the app offers visualization functionality that allows users to compare the predicted yield with the actual data used to train the model.

## Features:

- User-friendly graphical interface
- Data input fields for rainfall, sunshine hours, and average temperature
- A "Predict Crop Yield" button that triggers the prediction
- A "Visualize Predicted Data" button that displays a scatter plot comparing the predicted yield with the actual data
- Output display for the predicted yield
- Automatically generated result text
- Error handling for invalid numerical inputs

## Installation:

To run this application, ensure you have the following installed:

- Python 3 or later
- NumPy
- Scikit-learn
- PyQt5

To install PyQt5 using pip, open a command prompt or terminal and run the following command:

```pip install PyQt5```

## Usage:

- Download the application files and save them in the same directory.
- Open a command prompt or terminal and navigate to the directory where the files are located.
- Run the following command to launch the application:

## Notes:

- The linear regression model has been trained on a limited dataset and should be used for demonstration purposes only.
- The actual crop yield may vary depending on various other factors not considered in this model.
- This app is intended to provide an interactive way to explore the relationship between weather conditions and crop yield and to demonstrate the use of linear regression for predictive modeling.
