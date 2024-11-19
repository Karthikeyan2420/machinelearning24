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
