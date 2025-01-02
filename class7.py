"""Logistic Regression is used to predict the categorical dependent variable 
using a given set of independent variables."""
""" 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# Load data from CSV
data = pd.read_csv('adult.csv')

# Assuming the last column is the target variable and the rest are features
X = data.drop('age', axis=1)  # Features
y = data['age']  #
X = pd.get_dummies(X)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Model evaluation
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')
plt.xlabel('Actual Age')
plt.ylabel('Predicted Age')
plt.title('Actual vs. Predicted Age (Test Set)')
plt.show()  """



""" 
#With Plot
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load data from CSV
data = pd.read_csv('adult.csv')

# Assuming 'age' is the target variable and the rest are features
X = data.drop('age', axis=1)  # Features
y = data['age']  # Target variable

# One-hot encode categorical variables
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Model evaluation
train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)

print("Training RMSE:", train_rmse)
print("Testing RMSE:", test_rmse)

# Plot predictions vs. actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')
plt.xlabel('Actual Age')
plt.ylabel('Predicted Age')
plt.title('Actual vs. Predicted Age (Test Set)')
plt.show()  """
 



"""Linear regression is used to predict the continuous dependent variable using a given set of independent variables. """
 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load data from CSV
data = pd.read_csv('adult.csv')

# Assuming 'age' is the target variable and the rest are features
X = data.drop('age', axis=1)  # Features
y = data['age']  # Target variable

# One-hot encode categorical variables
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Model evaluation
train_rmse = mean_squared_error(y_train, y_pred_train)
test_rmse = mean_squared_error(y_test, y_pred_test)

print("Training RMSE:", train_rmse)
print("Testing RMSE:", test_rmse) 
 

""" 
 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load data from CSV
data = pd.read_csv('adult.csv')

# Assuming 'y' is the target variable and 'X1' is the only feature
X_simple = data[['workclass']]  # Feature for simple linear regression
y_simple = data['age']  # Target variable
X_simple = pd.get_dummies(X_simple)
# Split the data for simple linear regression
X_simple_train, X_simple_test, y_simple_train, y_simple_test = train_test_split(X_simple, y_simple, test_size=0.2, random_state=42)

# Create and train simple linear regression model
model_simple = LinearRegression()
model_simple.fit(X_simple_train, y_simple_train)

# Predictions for simple linear regression
y_simple_pred_train = model_simple.predict(X_simple_train)
y_simple_pred_test = model_simple.predict(X_simple_test)

# Model evaluation for simple linear regression
train_rmse_simple = mean_squared_error(y_simple_train, y_simple_pred_train, squared=False)
test_rmse_simple = mean_squared_error(y_simple_test, y_simple_pred_test, squared=False)

print("Simple Linear Regression - Training RMSE:", train_rmse_simple)
print("Simple Linear Regression - Testing RMSE:", test_rmse_simple)
 """