import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QLineEdit, QCheckBox, \
    QHBoxLayout
from PyQt5.QtCore import Qt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures



class MICP:
    def __init__(self):
        # Read CSV file into a DataFrame
        self.df = pd.read_csv('Dataset/medical.csv')

    def BaseModel(self, test=True):
        age = self.df['age'].values.tolist()
        sex = self.df['sex'].apply(lambda x: 1 if x == 'female' else 0).values.tolist()
        bmi = self.df['bmi'].values.tolist()
        children = self.df['children'].values.tolist()
        smoker = self.df['smoker'].apply(lambda x: 1 if x == 'yes' else 0).values.tolist()
        region = self.df['region'].apply(lambda x: 1 if x == 'southeast' else 0).values.tolist()
        charges = self.df['charges'].values.tolist()

        features = np.stack([age, sex, bmi, children, smoker, region], axis=1)

        if test:
            # Split the data into 75% training and 25% testing
            X_train, X_test, y_train, y_test = train_test_split(features, charges, test_size=0.25, random_state=42)

            # TRAINING LINEAR REGRESSION MODEL
            model = LinearRegression()
            model.fit(X_train, y_train)

            # PREDICT ON TEST SET
            y_pred = model.predict(X_test)

            # CALCULATE MEAN SQUARED ERROR
            mse = mean_squared_error(y_test, y_pred)
            print(f'Mean Squared Error: {mse}')

            # Plotting the original charges with a red line
            plt.plot(y_test, color='red', label='Original Charges')

            # Plotting the predicted charges with a blue line
            plt.plot(y_pred, color='blue', label='Predicted Charges')

            plt.xlabel("Data Points")
            plt.ylabel("Charges")
            plt.title("Original Charges vs Predicted Charges")
            plt.legend()
            plt.show()
        else:
            # If test is False, train on 100% data
            model = LinearRegression()
            model.fit(features, charges)
            return model  # Returning the trained model for future predictions

    def AlphaModel(self, test=True):
        # DATA PREPROCESSING
        age = self.df['age'].values.tolist()
        sex = self.df['sex'].apply(lambda x: 1 if x == 'female' else 0).values.tolist()
        bmi = self.df['bmi'].values.tolist()
        children = self.df['children'].values.tolist()
        smoker = self.df['smoker'].apply(lambda x: 1 if x == 'yes' else 0).values.tolist()
        region = self.df['region'].apply(lambda x: 1 if x == 'southeast' else 0).values.tolist()
        charges = self.df['charges'].values.tolist()

        features = np.stack([age, sex, bmi, children, smoker, region], axis=1)

        if test:
            # Feature Scaling
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # Polynomial Features
            poly = PolynomialFeatures(degree=2)
            features_poly = poly.fit_transform(features_scaled)

            # Split the data into 75% training and 25% testing
            X_train, X_test, y_train, y_test = train_test_split(features_poly, charges, test_size=0.25, random_state=42)

            # Ridge Regression with Hyperparameter Tuning
            param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
            ridge_model = Ridge()
            grid_search = GridSearchCV(ridge_model, param_grid, cv=5)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)

            # CALCULATE MEAN SQUARED ERROR
            mse = mean_squared_error(y_test, y_pred)
            print(f'Mean Squared Error: {mse}')

            # Plotting the original charges with a red line
            plt.plot(y_test, color='red', label='Original Charges')

            # Plotting the predicted charges with a blue line
            plt.plot(y_pred, color='blue', label='Predicted Charges')

            plt.xlabel("Data Points")
            plt.ylabel("Charges")
            plt.title("Original Charges vs Predicted Charges")
            plt.legend()
            plt.show()
        else:
            # If test is False, train on 100% data
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            poly = PolynomialFeatures(degree=2)
            features_poly = poly.fit_transform(features_scaled)

            # Ridge Regression with Hyperparameter Tuning
            param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
            ridge_model = Ridge()
            grid_search = GridSearchCV(ridge_model, param_grid, cv=5)
            grid_search.fit(features_poly, charges)
            best_model = grid_search.best_estimator_
            return best_model  # Returning the trained model for future predictions

class MICPGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        # Create widgets
        self.model_label = QLabel('Choose Model:')
        self.model_combobox = QComboBox()
        self.model_combobox.addItem('BaseModel')
        self.model_combobox.addItem('AlphaModel')

        self.developer_label = QLabel('Are you a developer?')
        self.developer_checkbox = QCheckBox()
        self.developer_checkbox.stateChanged.connect(self.on_developer_checkbox_changed)

        self.age_label = QLabel('Age:')
        self.age_input = QLineEdit()

        self.sex_label = QLabel('Sex (female/male):')
        self.sex_input = QLineEdit()

        self.bmi_label = QLabel('BMI:')
        self.bmi_input = QLineEdit()

        self.children_label = QLabel('Number of Children:')
        self.children_input = QLineEdit()

        self.smoker_label = QLabel('Smoker? (yes/no):')
        self.smoker_input = QLineEdit()

        self.region_label = QLabel('Region (southeast/other):')
        self.region_input = QLineEdit()

        self.result_label = QLabel('Predicted Charges:')

        self.submit_button = QPushButton('Submit')
        self.submit_button.clicked.connect(self.on_submit)

        # Layout
        layout = QVBoxLayout()

        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.model_combobox)

        developer_layout = QHBoxLayout()
        developer_layout.addWidget(self.developer_label)
        developer_layout.addWidget(self.developer_checkbox)

        form_layout = QVBoxLayout()
        form_layout.addLayout(model_layout)
        form_layout.addLayout(developer_layout)
        form_layout.addWidget(self.age_label)
        form_layout.addWidget(self.age_input)
        form_layout.addWidget(self.sex_label)
        form_layout.addWidget(self.sex_input)
        form_layout.addWidget(self.bmi_label)
        form_layout.addWidget(self.bmi_input)
        form_layout.addWidget(self.children_label)
        form_layout.addWidget(self.children_input)
        form_layout.addWidget(self.smoker_label)
        form_layout.addWidget(self.smoker_input)
        form_layout.addWidget(self.region_label)
        form_layout.addWidget(self.region_input)

        layout.addLayout(form_layout)
        layout.addWidget(self.submit_button)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

        self.setWindowTitle('MICP GUI')
        self.setGeometry(100, 100, 400, 400)
        self.show()

    def on_developer_checkbox_changed(self, state):
        # Disable/enable input fields based on the state of the developer checkbox
        is_enabled = state != Qt.Checked
        for widget in [self.age_input, self.sex_input, self.bmi_input, self.children_input,
                       self.smoker_input, self.region_input]:
            widget.setEnabled(is_enabled)

    def on_submit(self):
        chosen_model = self.model_combobox.currentText()
        is_developer = self.developer_checkbox.isChecked()

        micp = MICP()

        if is_developer:
            if chosen_model == 'BaseModel':
                model = micp.BaseModel(test=True)
            elif chosen_model == 'AlphaModel':
                model = micp.AlphaModel(test=True)
            self.result_label.setText('Model called in developer mode. Check the console for details.')
        else:
            # Take necessary details for prediction
            age = float(self.age_input.text())
            sex = 1 if self.sex_input.text().lower() == 'female' else 0
            bmi = float(self.bmi_input.text())
            children = int(self.children_input.text())
            smoker = 1 if self.smoker_input.text().lower() == 'yes' else 0
            region = 1 if self.region_input.text().lower() == 'southeast' else 0

            # Pre-process data
            input_data = np.array([[age, sex, bmi, children, smoker, region]])
            if chosen_model == 'AlphaModel':
                scaler = StandardScaler()
                input_data = scaler.fit_transform(input_data)

                poly = PolynomialFeatures(degree=2)
                input_data = poly.fit_transform(input_data)

            # Make predictions
            if chosen_model == 'BaseModel':
                model = micp.BaseModel(test=False)
            elif chosen_model == 'AlphaModel':
                model = micp.AlphaModel(test=False)

            prediction = model.predict(input_data)
            rounded_prediction = round(prediction[0], 2)  # Round to two decimal places
            self.result_label.setText(f'Predicted Annual Charges: ${rounded_prediction}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    micp_gui = MICPGUI()
    sys.exit(app.exec_())
