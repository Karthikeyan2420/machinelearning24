""" import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
data = pd.read_csv('adult.csv')
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 
                       'relationship', 'race', 'sex', 'nativeCountry']
data = pd.get_dummies(data, columns=categorical_columns)
data['income'] = data['income'].apply(lambda x: 1 if x == '>50K' else 0)
X = data.drop('income', axis=1)
y = data['income']
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
 """


""" 
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset
data = pd.read_csv('data.csv')
X = data.drop('cv', axis=1)
y = data['cv']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Decision Tree Regressor
tree = DecisionTreeRegressor(max_depth=5, random_state=42)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

# Calculate MSE
mse_tree = mean_squared_error(y_test, y_pred)
print(f'Decision Tree MSE: {mse_tree:.2f}')

# Baseline Model: Mean Predictor
y_baseline_pred = np.full(shape=y_test.shape, fill_value=y_train.mean())
mse_baseline = mean_squared_error(y_test, y_baseline_pred)
print(f'Baseline MSE: {mse_baseline:.2f}')

# Interpretation
if mse_tree < mse_baseline:
    print("The Decision Tree model performs better than the baseline.")
else:
    print("The Decision Tree model does not outperform the baseline.")
 """
""" 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load the dataset
data = pd.read_csv('data.csv')

# Separate features and target variable
X = data.drop('cv', axis=1)
y = data['cv']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the data for better performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the Feedforward Neural Network
model = Sequential()

# Input layer + Hidden layer 1
model.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))

# Hidden layer 2
model.add(Dense(32, activation='relu'))

# Output layer (no activation for regression)
model.add(Dense(1))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_squared_error'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2)

# Evaluate the model
mse = model.evaluate(X_test, y_test, verbose=0)[1]
print(f'Mean Squared Error on test set: {mse:.2f}') """

""" 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load the dataset
data = pd.read_csv('data.csv')

# Separate features and target variable
X = data.drop('cv', axis=1)
y = data['cv']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the data for better performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the Multilayer Perceptron model
model = Sequential()

# Input layer + Hidden layer 1
model.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))

# Hidden layer 2
model.add(Dense(32, activation='relu'))

# Hidden layer 3
model.add(Dense(16, activation='relu'))

# Output layer (no activation for regression)
model.add(Dense(1))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_squared_error'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2)

# Evaluate the model
mse = model.evaluate(X_test, y_test, verbose=0)[1]
print(f'Mean Squared Error on test set: {mse:.2f}')

 """
""" 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load and preprocess data
data = pd.read_csv('data.csv')
X = data.drop('cv', axis=1)  # Features
y = data['cv']               # Target variable

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the Deep Neural Network
model = Sequential()

# Add multiple hidden layers to make it deep
model.add(Dense(128, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

# Output layer for regression (1 neuron, no activation)
model.add(Dense(1))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_squared_error'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model
mse = model.evaluate(X_test, y_test, verbose=0)[1]
print(f'Mean Squared Error on test set: {mse:.2f}')
 """
""" 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

# Load dataset
data = pd.read_csv('data.csv')
X = data.drop('target_column', axis=1)  # Features
y = data['target_column']               # Target variable

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train a gradient boosting model
model = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, max_depth=3)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
 """
""" 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import BaggingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from xgboost import plot_importance

# Step 1: Load the dataset
data = pd.DataFrame({
    "G": [2.7, 2.7, 2.7, 2.7, 2.65, 2.65, 2.65, 2.58, 2.58, 2.58, 2.7, 2.7, 2.7, 2.7],
    "wL": [37, 37, 37, 37, 39, 39, 39, 73.4, 73.4, 73.4, 73.5, 73.5, 73.5, 73.5],
    "wp": [18, 18, 18, 18, 29.5, 29.5, 29.5, 51.9, 51.9, 51.9, 37.9, 37.9, 37.9, 37.9],
    "IP": [19, 19, 19, 19, 9.5, 9.5, 9.5, 21.5, 21.5, 21.5, 35.6, 35.6, 35.6, 35.6],
    "% Clay": [26, 26, 26, 26, 5, 5, 5, 27.5, 27.5, 27.5, 37.9, 37.9, 37.9, 37.9],
    "% Silt": [38.5, 38.5, 38.5, 38.5, 58.5, 58.5, 58.5, 51.5, 51.5, 51.5, 51.5, 51.5, 51.5, 51.5],
    "% Sand": [35.5, 35.5, 35.5, 35.5, 36.5, 36.5, 36.5, 21, 21, 21, 35.5, 35.5, 35.5, 35.5],
    "Pressure": [25, 50, 100, 200, 25, 50, 100, 25, 50, 100, 25, 50, 100, 200],
    "cv": [15.11, 19.3, 11.52, 8.2, 11.54, 5.3, 2.67, 2.07, 1.07, 0.7, 1.07, 0.5, 0.2, 0.07]
})

# Step 2: Split the dataset into features and target
X = data.drop("cv", axis=1)
y = data["cv"]

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: XGBoost Regressor
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# Step 5: Evaluate XGBoost
y_pred = xgb_model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"XGBoost RMSE: {rmse:.4f}")

# Step 6: Cross-Validation
cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring="neg_root_mean_squared_error")
print(f"Cross-Validation RMSE: {-cv_scores.mean():.4f}")

# Step 7: Feature Importance Sensitivity Analysis
print("\nFeature Sensitivity Analysis:")
for feature in X.columns:
    X_copy = X_test.copy()
    X_copy[feature] += np.random.normal(0, 0.1, size=X_copy[feature].shape)
    y_perturbed_pred = xgb_model.predict(X_copy)
    sensitivity = np.mean(np.abs(y_perturbed_pred - y_pred))
    print(f"Sensitivity for {feature}: {sensitivity:.4f}")


# Step 8: Bagging Regressor
bagging_model = BaggingRegressor(estimator=XGBRegressor(), n_estimators=10, random_state=42)
bagging_model.fit(X_train, y_train)
y_bagging_pred = bagging_model.predict(X_test)
bagging_rmse = mean_squared_error(y_test, y_bagging_pred, squared=False)
print(f"\nBagging RMSE: {bagging_rmse:.4f}")


# Step 9: Boosting with Hyperparameters
boosting_model = XGBRegressor(
    n_estimators=200, 
    learning_rate=0.05, 
    max_depth=4, 
    subsample=0.8, 
    colsample_bytree=0.8, 
    random_state=42
)
boosting_model.fit(X_train, y_train)
y_boosting_pred = boosting_model.predict(X_test)
boosting_rmse = mean_squared_error(y_test, y_boosting_pred, squared=False)
print(f"Boosting RMSE: {boosting_rmse:.4f}")

# Step 10: Plot Feature Importance
plt.figure(figsize=(10, 8))
plot_importance(xgb_model, importance_type="weight")
plt.title("Feature Importance")
plt.show()

 """

""" 
import pandas as pd
import numpy as np
df = pd.read_csv("data.csv")
def perturb_data(df, noise_level=0.1):
    df_perturbed = df.copy()
    for column in df.select_dtypes(include=np.number).columns:
        noise = np.random.normal(0, noise_level, df[column].shape)
        df_perturbed[column] = df[column] + noise
    return df_perturbed
additional_rows = 1000 
expanded_df = df.copy()
for _ in range(additional_rows // len(df)): 
    perturbed_df = perturb_data(df)
    expanded_df = pd.concat([expanded_df, perturbed_df], ignore_index=True)
expanded_df.to_csv("large_data.csv", index=False)
 """
""" 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# Generate synthetic data
X, y = make_regression(n_samples=1000, n_features=5, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network
model = Sequential([
    Dense(10, input_dim=X.shape[1], activation='relu'),
    Dense(1, activation='linear')
])

# Compile and train the model
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)


def garson_feature_importance(model):
    
    weights = [layer.get_weights()[0] for layer in model.layers if len(layer.get_weights()) > 0]
    
    # Extract input-to-hidden and hidden-to-output weights
    input_hidden_weights = weights[0]
    hidden_output_weights = weights[1]
    
    # Compute absolute contributions
    abs_input_hidden = np.abs(input_hidden_weights)
    abs_hidden_output = np.abs(hidden_output_weights)
    
    # Contribution of each input feature
    total_contribution = np.sum(abs_input_hidden, axis=0) * np.sum(abs_hidden_output, axis=1)
    feature_importance = total_contribution / np.sum(total_contribution)
    
    return feature_importance

# Get feature importances
importances = garson_feature_importance(model)
print("Feature Importances (Garson):", importances)


def connection_weight_importance(model):
  
    weights = [layer.get_weights()[0] for layer in model.layers if len(layer.get_weights()) > 0]
    
    # Extract input-to-hidden and hidden-to-output weights
    input_hidden_weights = weights[0]
    hidden_output_weights = weights[1]
    
    # Compute product of weights for connection weights
    connection_weights = input_hidden_weights @ hidden_output_weights
    feature_importance = np.sum(np.abs(connection_weights), axis=1)
    feature_importance /= np.sum(feature_importance)  # Normalize
    
    return feature_importance

# Get feature importances
importances_cw = connection_weight_importance(model)
print("Feature Importances (Connection Weight):", importances_cw)


"""











import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# One-hot encode the target labels (3 classes)
y_one_hot = np.zeros((y.size, y.max() + 1))
y_one_hot[np.arange(y.size), y] = 1

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define a simple neural network with one hidden layer
input_layer_neurons = X_train.shape[1]  # Number of features (4 for Iris dataset)
hidden_layer_neurons = 5               # Number of neurons in the hidden layer
output_layer_neurons = y_train.shape[1] # Number of classes (3 classes)

# Initialize weights and biases
hidden_weights = np.random.randn(input_layer_neurons, hidden_layer_neurons)
hidden_bias = np.random.randn(1, hidden_layer_neurons)
output_weights = np.random.randn(hidden_layer_neurons, output_layer_neurons)
output_bias = np.random.randn(1, output_layer_neurons)

# Set learning rate and number of epochs
learning_rate = 0.01
epochs = 10000

# Training the neural network using backpropagation
for epoch in range(epochs):
    # Feedforward
    hidden_layer_input = np.dot(X_train, hidden_weights) + hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_output, output_weights) + output_bias
    predicted_output = sigmoid(output_layer_input)
    
    # Calculate the error (difference between predicted and actual output)
    error = y_train - predicted_output
    
    # Backpropagation
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # Update weights and biases
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    hidden_weights += X_train.T.dot(d_hidden_layer) * learning_rate
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate
    
    # Print error every 1000 iterations
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Error: {np.mean(np.abs(error))}")

# Final output after training
final_output = predicted_output

# Convert final output to predicted class labels
predicted_class = np.argmax(final_output, axis=1)
true_class = np.argmax(y_train, axis=1)

# Calculate accuracy
accuracy = accuracy_score(true_class, predicted_class)
print("\nTraining Accuracy: ", accuracy)

# Test the model on the test set
hidden_layer_input_test = np.dot(X_test, hidden_weights) + hidden_bias
hidden_layer_output_test = sigmoid(hidden_layer_input_test)

output_layer_input_test = np.dot(hidden_layer_output_test, output_weights) + output_bias
predicted_output_test = sigmoid(output_layer_input_test)

# Convert final output to predicted class labels
predicted_class_test = np.argmax(predicted_output_test, axis=1)
true_class_test = np.argmax(y_test, axis=1)

# Calculate accuracy on the test set
test_accuracy = accuracy_score(true_class_test, predicted_class_test)
print("Test Accuracy: ", test_accuracy)
