from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
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

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

# Scale the target variable
scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# Build the FFNN model with improvements
model = Sequential()
model.add(Dense(256, input_dim=X_train.shape[1], kernel_regularizer=l2(0.01)))  # L2 regularization
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())  # Batch Normalization
model.add(Dropout(0.4))  # Increased dropout for better regularization

# Additional hidden layers
model.add(Dense(128, kernel_regularizer=l2(0.01)))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(64, kernel_regularizer=l2(0.01)))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(Dropout(0.4))

# Output layer
model.add(Dense(1))  # Output layer

# Compile the model with an advanced optimizer
optimizer = Nadam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mae', metrics=['mse'])

# Callbacks: Early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=300,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate the model
loss, mse = model.evaluate(X_test, y_test)
predictions = model.predict(X_test).flatten()

# Reverse scaling to original range for predictions and actual values
predictions_original = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Calculate R2 Score and Mean Absolute Error
mae = mean_absolute_error(y_test_original, predictions_original)
r2 = r2_score(y_test_original, predictions_original)

print(f"Test MAE: {mae}, R2 Score: {r2}")

# Combine predictions with actual values
results = np.concatenate(
    (predictions_original.reshape(-1, 1), y_test_original.reshape(-1, 1)),
    axis=1
)
results_df = pd.DataFrame(results, columns=['Predicted Value', 'Actual Value'])

# Calculate custom accuracy (e.g., within ±10% tolerance)
tolerance = 0.1  # 10% tolerance
correct_predictions = np.abs(predictions_original - y_test_original) <= tolerance * y_test_original
accuracy = np.sum(correct_predictions) / len(y_test_original) * 100

# Display predictions and accuracy
print(f"Custom Accuracy (within ±10% tolerance): {accuracy:.2f}%")
print(results_df.head(10))  # Display the first 10 rows
