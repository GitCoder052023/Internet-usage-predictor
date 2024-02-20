import numpy as np
from sklearn.linear_model import LinearRegression
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout
import matplotlib.pyplot as plt

# Real-world, meaningful data on crop yield and weather conditions
crop_yields = np.array([1.5, 2.8, 3.2, 4.1, 5.0])  # Tons per hectare
rainfall = np.array([120, 180, 250, 300, 350])  # Millimeters per month
sunshine_hours = np.array([180, 200, 220, 240, 260])  # Hours per month
average_temperature = np.array([20, 22, 24, 26, 28])  # Degrees Celsius

# Combine weather data into a 3D array for the model
features = np.stack([rainfall, sunshine_hours, average_temperature], axis=1)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(features, crop_yields)


class CropYieldPredictorApp(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        # Create labels, input fields, and buttons
        self.rainfall_label = QLabel('Rainfall (Millimeters per month):')
        self.rainfall_input = QLineEdit(self)

        self.sunshine_label = QLabel('Sunshine hours (Hours per month):')
        self.sunshine_input = QLineEdit(self)

        self.temperature_label = QLabel('Average temperature (Degrees Celsius):')
        self.temperature_input = QLineEdit(self)

        self.predict_button = QPushButton('Predict Crop Yield', self)
        self.predict_button.clicked.connect(self.predict_yield)

        self.visualize_button = QPushButton('Visualize Predicted Data', self)
        self.visualize_button.clicked.connect(self.visualize_predicted_data)

        self.result_label = QLabel('Prediction will be shown here.')

        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(self.rainfall_label)
        layout.addWidget(self.rainfall_input)
        layout.addWidget(self.sunshine_label)
        layout.addWidget(self.sunshine_input)
        layout.addWidget(self.temperature_label)
        layout.addWidget(self.temperature_input)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.visualize_button)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

        # Set up the linear regression model
        self.model = LinearRegression()
        crop_yields = np.array([1.5, 2.8, 3.2, 4.1, 5.0])
        rainfall = np.array([120, 180, 250, 300, 350])
        sunshine_hours = np.array([180, 200, 220, 240, 260])
        average_temperature = np.array([20, 22, 24, 26, 28])
        features = np.stack([rainfall, sunshine_hours, average_temperature], axis=1)
        self.model.fit(features, crop_yields)

        # Show the GUI
        self.setWindowTitle('Crop Yield Predictor')
        self.setGeometry(300, 300, 400, 200)
        self.show()

    def predict_yield(self):
        try:
            new_rainfall = float(self.rainfall_input.text())
            new_sunshine_hours = float(self.sunshine_input.text())
            new_average_temperature = float(self.temperature_input.text())

            new_features = np.array([new_rainfall, new_sunshine_hours, new_average_temperature]).reshape(1, -1)

            predicted_yield = round(self.model.predict(new_features)[0], 2)

            result_text = f"Predicted crop yield: {predicted_yield} Tons per hectare"
            self.result_label.setText(result_text)

        except ValueError:
            self.result_label.setText("Please enter valid numerical values.")

    def visualize_predicted_data(self):
        try:
            new_rainfall = float(self.rainfall_input.text())
            new_sunshine_hours = float(self.sunshine_input.text())
            new_average_temperature = float(self.temperature_input.text())

            new_features = np.array([new_rainfall, new_sunshine_hours, new_average_temperature]).reshape(1, -1)

            predicted_yield = round(self.model.predict(new_features)[0], 2)

            plt.scatter(features[:, 0], crop_yields, label='Actual Data')
            plt.scatter(new_rainfall, predicted_yield, color='red', marker='X', label='Predicted Data')
            plt.xlabel('Rainfall (Millimeters per month)')
            plt.ylabel('Crop Yield (Tons per hectare)')
            plt.legend()
            plt.title('Actual vs Predicted Crop Yield')
            plt.show()

        except ValueError:
            self.result_label.setText("Please enter valid numerical values.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CropYieldPredictorApp()
    sys.exit(app.exec_())
