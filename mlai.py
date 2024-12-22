from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np

# Load and preprocess data
data = pd.read_csv('data1.csv')  # Replace with your dataset path
X = data.drop('Cv', axis=1)
y = data['Cv']

# Handle possible outliers in the target
y = np.clip(y, y.quantile(0.05), y.quantile(0.95))  # Limit to 5th-95th percentile

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

# Reshape X for CNN (reshape to [samples, time_steps, features] for 1D convolution)
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the CNN model
model = Sequential()

# 1D Convolutional Layer
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))

# Additional Convolutional and MaxPooling layers
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# Flatten the output to feed into fully connected layers
model.add(Flatten())

# Dense layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))  # Dropout for regularization
model.add(Dense(64, activation='relu'))
model.add(Dense(1))  # Output layer (for regression)

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mae', metrics=['mse'])

# Train the model
history = model.fit(
    X_train_reshaped, y_train,
    epochs=300,
    batch_size=32,
    validation_data=(X_test_reshaped, y_test),
    verbose=1
)

# Evaluate the model
predictions = model.predict(X_test_reshaped).flatten()

# Calculate performance metrics
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Test MAE: {mae}, R2 Score: {r2}")

# Combine predictions with actual values for comparison
results = np.concatenate((predictions.reshape(-1, 1), y_test.values.reshape(-1, 1)), axis=1)
results_df = pd.DataFrame(results, columns=['Predicted Value', 'Actual Value'])

# Calculate custom accuracy (e.g., within ±10% tolerance)
tolerance = 0.1  # 10% tolerance
correct_predictions = np.abs(predictions - y_test) <= tolerance * y_test
accuracy = np.sum(correct_predictions) / len(y_test) * 100

# Display predictions and accuracy
print(f"Custom Accuracy (within ±10% tolerance): {accuracy:.2f}%")
print(results_df.head(10))  # Display the first 10 rows
