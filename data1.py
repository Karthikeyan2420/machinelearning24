import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber

# Load data
data = pd.read_csv('data.csv')
X = data.drop('cv', axis=1)
y = data['cv']

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

# Target Scaling (MinMax)
scaler_y = MinMaxScaler()
y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test = scaler_y.transform(y_test.values.reshape(-1, 1))

# Model Architecture
model = Sequential()

# Input Layer + Batch Normalization
model.add(Dense(512, input_shape=(X_train.shape[1],), activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Hidden Layers
model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Output Layer
model.add(Dense(1))  # Regression output

# Compile the Model
optimizer = Adam(learning_rate=0.001)
loss_fn = Huber(delta=1.0)  # Huber Loss for regression tasks
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['mean_squared_error'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)

# Train the Model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=500,
    batch_size=16,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Evaluate on Test Set
test_mse = model.evaluate(X_test, y_test, verbose=0)[1]
print(f"\nFinal Mean Squared Error on Test Set: {test_mse:.4f}")

# Predictions
y_pred = model.predict(X_test)
y_pred_original = scaler_y.inverse_transform(y_pred)
y_test_original = scaler_y.inverse_transform(y_test)

# Display Results
results = pd.DataFrame({'Actual': y_test_original.flatten(), 'Predicted': y_pred_original.flatten()})
print("\nPredictions vs Actual:")
print(results.head(10))

# Plot Actual vs Predicted Values
plt.figure(figsize=(10, 6))
plt.plot(y_test_original, label='Actual', marker='o')
plt.plot(y_pred_original, label='Predicted', linestyle='--', marker='x')
plt.title('Actual vs Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('cv')
plt.legend()
plt.grid()
plt.show()
