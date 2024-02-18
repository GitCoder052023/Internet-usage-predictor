import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout
import numpy as np
from Dataset.internet import years
from Dataset.internet import usage
from sklearn.linear_model import LinearRegression


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

        # Create a central widget and set the layout
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Connect the button's clicked signal to the prediction function
        self.predict_button.clicked.connect(self.predict_usage)

    def predict_usage(self):
        # Get the year from the input field
        new_year = int(self.year_input.text())

        # Run your model prediction code here
        predicted_usage = model.predict([[new_year]])[0]  # Assuming you have the model trained

        # Display the prediction
        self.prediction_label.setText(f"Predicted Internet usage for {new_year} is: {round(predicted_usage, 2)} Billions")

if __name__ == "__main__":
    app = QApplication(sys.argv)  # Create the application instance
    window = MainWindow()  # Create the main window instance
    window.show()  # Show the main window
    sys.exit(app.exec_())  # Start the event loop and exit properly