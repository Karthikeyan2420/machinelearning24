# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# Load and Prepare the Data
data = pd.read_csv('data.csv')  # Load the dataset
X = data.drop('cv', axis=1)  # Features
y = data['cv']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Target Scaling
scaler_y = MinMaxScaler()
y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test = scaler_y.transform(y_test.values.reshape(-1, 1))

# Define the Neural Network Model
model = Sequential()
model.add(Dense(512, input_shape=(X_train.shape[1],), activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(1))  # Output layer for regression

# Compile the Model with Adam Optimizer
model.compile(optimizer=Adam(learning_rate=0.0005),
              loss='mean_squared_error',
              metrics=['mean_squared_error'])

# Early Stopping Callback to Avoid Overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Train the Model
history = model.fit(X_train, y_train,
                    epochs=300,
                    batch_size=8,
                    validation_split=0.2,
                    callbacks=[early_stop],
                    verbose=1)

# Evaluate the Model
mse = model.evaluate(X_test, y_test, verbose=0)[1]
print(f'Mean Squared Error on test set: {mse:.2f}')

# Make Predictions
y_pred = model.predict(X_test)

# Denormalize Predictions and Actual Values
y_pred_original = scaler_y.inverse_transform(y_pred)
y_test_original = scaler_y.inverse_transform(y_test)

# Display Predictions vs Actual
results = pd.DataFrame({'Actual': y_test_original.flatten(), 'Predicted': y_pred_original.flatten()})
print("\nPredictions vs Actual:")
print(results.head())

# Visualize Predictions vs Actual Values
plt.figure(figsize=(10, 6))
plt.plot(y_test_original, label='Actual')
plt.plot(y_pred_original, label='Predicted', linestyle='--')
plt.title('Actual vs Predicted Values')
plt.xlabel('Samples')
plt.ylabel('cv')
plt.legend()
plt.show()
