
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder

# Load the CSV data
data = pd.read_csv("Tennisdataset1.2.csv")

# Preprocess the features data
feature_cols = ['Outlook', 'Temperature', 'Humidity', 'Wind']
X = data[feature_cols]
X_encoded = X.apply(LabelEncoder().fit_transform)

# Preprocess the target variable
y = data['enjoysport']
y_encoded = LabelEncoder().fit_transform(y)

# Initialize the Naive Bayes classifier
clf = CategoricalNB()

# Train the classifier
clf.fit(X_encoded, y_encoded)

""" # Example data for prediction
new_data = pd.DataFrame({
    'Outlook': ['Sunny'],
    'Temperature': ['Hot'],
    'Humidity': ['High'],
    'Wind': ['Weak']
})

# Preprocess new data
new_data_encoded = new_data.apply(LabelEncoder().fit_transform)

# Make predictions
predictions = clf.predict(new_data_encoded)
print(predictions) """

""" 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

# Load your dataset (replace 'your_dataset.csv' with the path to your dataset)
data = pd.read_csv('tennisdataset1.1.csv')

# Preprocess your data if needed

# Split your dataset into features (X) and target variable (y)
X = data.drop('enjoysport', axis=1)  # Features
y = data['enjoysport']  # Target variable
X = pd.get_dummies(X)
print(X)
# Split your dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Choose a Naive Bayes classifier (e.g., Gaussian Naive Bayes for continuous features)
model = GaussianNB()

# Train your classifier
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate your classifier
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred, average='weighted'))
print("Recall:", metrics.recall_score(y_test, y_pred, average='weighted'))
print("F1 Score:", metrics.f1_score(y_test, y_pred, average='weighted'))



 """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Create a larger synthetic dataset
data = {
    'age': np.random.randint(20, 60, 100),  # Random ages between 20 and 60
    'education_level': np.random.choice(
        ['High School', 'Bachelor\'s', 'Master\'s', 'PhD'], 100
    ),
    'hours_per_week': np.random.randint(20, 80, 100),  # Random hours between 20 and 80
    'experience_years': np.random.randint(0, 40, 100),  # Random experience between 0 and 40 years
    'income': np.random.randint(25000, 120000, 100)  # Random income between 25k and 120k
}

# Convert to a pandas dataframe
df = pd.DataFrame(data)

# Print the first few rows of the dataset
print("Sample Data:\n", df.head())

# Separate features and target variable
X = df.drop('income', axis=1)  # Features
y = df['income']               # Target

# Identify categorical and numerical columns
categorical_cols = ['education_level']
numerical_cols = ['age', 'hours_per_week', 'experience_years']

# Preprocessor: One-hot encode categorical features, scale numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),  # Standardize numerical features
        ('cat', OneHotEncoder(), categorical_cols)  # One-hot encode categorical features
    ])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create SVR model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),      # Preprocessing step
    ('svr', SVR(kernel='linear'))        # SVR model with a linear kernel
])

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
