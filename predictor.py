import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout
import numpy as np
from Dataset.internet import years
from Dataset.internet import usage
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Create two 1D arrays
year = np.array(years)
usage = np.array(usage)

# Transpose the combined array to have features as columns
combined_array = np.vstack((year, usage)).T

# Split the data into features (X) and target variable (y)
X = combined_array[:, 0].reshape(-1, 1)  # Reshape X to have 2D shape expected by LinearRegression
y = combined_array[:, 1]

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set window title and size
        self.setWindowTitle("Internet Usage Prediction")
        self.resize(400, 200)

        # Create widgets
        self.year_label = QLabel("Enter a year:")
        self.year_input = QLineEdit()
        self.predict_button = QPushButton("Predict Usage")
        self.prediction_label = QLabel("Predicted usage:")

        # Layout widgets using a layout manager (e.g., QVBoxLayout)
        layout = QVBoxLayout()
        layout.addWidget(self.year_label)
        layout.addWidget(self.year_input)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.prediction_label)

        # Add buttons for visualization options
        self.visualize_linear_fit = QPushButton("Visualize Linear Fit")
        self.visualize_prediction = QPushButton("Visualize Prediction")
        layout.addWidget(self.visualize_linear_fit)
        layout.addWidget(self.visualize_prediction)

        # Create a central widget and set the layout
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Connect button signals to their respective functions
        self.predict_button.clicked.connect(self.predict_usage)
        self.visualize_linear_fit.clicked.connect(self.visualize_linear_fit_data)
        self.visualize_prediction.clicked.connect(self.visualize_prediction_data)

    def predict_usage(self):
        # Get the year from the input field
        new_year = int(self.year_input.text())

        # Run your model prediction code here
        predicted_usage = model.predict([[new_year]])[0]  # Assuming you have the model trained

        # Display the prediction
        self.prediction_label.setText(f"Predicted Internet usage for {new_year} is: {round(predicted_usage, 2)} Billions")

    def visualize_linear_fit_data(self):
        # Plot the original data and the linear fit
        plt.figure()
        plt.scatter(years, usage)
        plt.plot(years, model.predict(X), color='red')
        plt.xlabel("Year")
        plt.ylabel("Internet Usage (Billions)")
        plt.title("Linear Fit for Internet Usage Data")
        plt.show()

    def visualize_prediction_data(self):
        # Get the year and predicted usage
        new_year = int(self.year_input.text())
        predicted_usage = model.predict([[new_year]])[0]

        # Plot the data points and the predicted value as a bar
        plt.figure()
        plt.bar(years, usage, label="Actual Usage")
        plt.bar([new_year], predicted_usage, label="Predicted Usage", color='blue')
        plt.xlabel("Year")
        plt.ylabel("Internet Usage (Billions)")
        plt.title(f"Predicted Internet Usage for {new_year}")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)  # Create the application instance
    window = MainWindow()  # Create the main window instance
    window.show()  # Show the main window
    sys.exit(app.exec_())  # Start the event loop and exit properly
