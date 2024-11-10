""" #random forest
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
data = pd.read_csv('adult.csv')
print(data)
data = data.drop(columns=['fnlwgt', 'nativeCountry', 'education', 'relationship', 'marital-status'])
data_encoded = pd.get_dummies(data)
X = data_encoded.drop(columns=['income_<=50K', 'income_>50K'])
y = data_encoded['income_>50K']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Plot feature importances
plt.figure(figsize=(10, 6))
importances = rf_classifier.feature_importances_
indices = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=90)
plt.title("Feature Importances")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()  """







""" 

#poission regression
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Generate some example data
np.random.seed(0)
n = 1000
X = np.random.rand(n, 2)  # Two independent variables
# Simulate counts
rate = np.exp(2 + 0.5*X[:,0] + 0.8*X[:,1])  # True Poisson rate
y = np.random.poisson(rate)

# Create a DataFrame
data = pd.DataFrame({'y': y, 'x1': X[:,0], 'x2': X[:,1]})
print(data)
# Fit Poisson regression model
model = sm.GLM(data['y'], sm.add_constant(data[['x1', 'x2']]), family=sm.families.Poisson()).fit()

# Print model summary
print(model.summary()) """

 
#Least Absolute Shrinkage and Selection Operator (LASSO)
""" 
import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generating some random data
X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lasso regression
lasso_reg = Lasso(alpha=0.1)  # You can adjust the alpha parameter for strength of regularization
lasso_reg.fit(X_train, y_train)
lasso_pred = lasso_reg.predict(X_test)
lasso_mse = mean_squared_error(y_test, lasso_pred)
print("Lasso MSE:", lasso_mse)

# Ridge regression
ridge_reg = Ridge(alpha=0.1)  # You can adjust the alpha parameter for strength of regularization
ridge_reg.fit(X_train, y_train)
ridge_pred = ridge_reg.predict(X_test)
ridge_mse = mean_squared_error(y_test, ridge_pred)
print("Ridge MSE:", ridge_mse) """



""" import numpy as np
import statsmodels.api as sm

# Generating some example data
np.random.seed(42)
n = 1000
x = np.random.normal(size=n)
z = np.random.binomial(1, 0.2, size=n)  # Binary process
y_count = np.random.poisson(np.exp(x))  # Count process
y = np.where(z == 1, 0, y_count)  # Combining zero-inflation and count process

# Creating design matrices
X = sm.add_constant(x)
Z = sm.add_constant(z)
print(X,Z)
# Fitting the zero-inflated negative binomial model
model = sm.ZeroInflatedNegativeBinomialP(y, X, Z)
results = model.fit()
print(results.summary()) """

"""Negative binomial regression is used to test for connections between confounding and predictor 
variables on a count outcome variable."""

import numpy as np
import statsmodels.api as sm

# Generate sample data
np.random.seed(123)
n_obs = 100
X = np.column_stack((np.ones(n_obs), np.random.normal(0, 1, n_obs)))
true_coefficients = np.array([1.5, 0.5])  # True coefficients
prob = 1 / (1 + np.exp(-np.dot(X, true_coefficients)))  # Probability of success
y = np.random.negative_binomial(1, prob)  # Generate negative binomial distributed response variable

# Fit the negative binomial regression model
negbin_model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
negbin_results = negbin_model.fit()

# Print the summary of the model
print(negbin_results.summary()) 
