# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
# Replace 'path_to_dataset.csv' with the actual path to your dataset
data = pd.read_csv('weather.csv')

# Check for missing values and handle them (e.g., by dropping or imputing)
data.dropna(inplace=True)

# Define feature columns (X) and the target column (y)
X = data[['temperature', 'humidity', 'wind', 'rain']]
y = data['fire_risk']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("\nClassification Report:\n", report)

# Example prediction
sample_data = [[30, 40, 10, 0]]  # temperature, humidity, wind, rain
predicted_risk = model.predict(sample_data)

print("Predicted Fire Risk (1=High, 0=Low):", predicted_risk[0])
