# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load your dataset (assuming it's in a CSV file, adjust accordingly)
data = pd.read_csv('your_dataset.csv')

# Assuming your dataset has 'temperature', 'current', and 'magnetic_field' columns
# Features (X) and target variable (y)
X = data[['temperature', 'current']]
y = data['magnetic_field']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print('Mean Squared Error:', mse)

# Example usage: Predict magnetic field variation for a new data point
new_data_point = [[25, 0.5]]  # Example temperature: 25°C, current: 0.5A
predicted_magnetic_field = model.predict(new_data_point)
print('Predicted Magnetic Field Variation:', predicted_magnetic_field)


