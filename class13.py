""" import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset (assuming it's a CSV file)
df = pd.read_csv('adult.csv')

# Drop any missing values
df.dropna(inplace=True)

# Features (X) and target (y)
X = df.drop('income', axis=1)  # Drop the target column
y = df['income']  # Target column

# Encode the target variable (<=50K -> 0, >50K -> 1)
y = y.map({'<=50K': 0, '>50K': 1})

# Preprocessing for numerical and categorical data
numerical_features = ['age', 'fnlwgt', 'education_num', 'capitalGain', 'capitalLoss', 'hoursPerWeek']
categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'nativeCountry']

# Create transformers for preprocessing
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a column transformer that applies the appropriate transformation
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# To determine the correct input dimension after encoding, we need to apply the transformations
X_processed = preprocessor.fit_transform(X)

# Convert X_processed to float32 to avoid type issues
X_processed = X_processed.astype('float32')

# Get the number of features after preprocessing
input_dim = X_processed.shape[1]

# Build the ANN model
model = Sequential([
    Dense(64, input_dim=input_dim, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Create a pipeline that first applies preprocessing, then trains the ANN model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert y_train and y_test to the correct type (int)
y_train = y_train.astype('int')
y_test = y_test.astype('int')

# Fit the pipeline, passing epochs to the model's fit method using `classifier__epochs`
pipeline.fit(X_train, y_train, classifier__epochs=10, classifier__batch_size=32)

# Ensure X_test is also in float32 format
X_test_processed = preprocessor.transform(X_test)
X_test_processed = X_test_processed.astype('float32')

# Ensure y_test is in int32 format
y_test = y_test.astype('int')

# Evaluate the Keras model directly since `score` is not available in the pipeline
test_loss, test_accuracy = model.evaluate(X_test_processed, y_test)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
 """
 
 
""" a=125
r=0

while(a>0):
    c=a%10
    r=(r*10)+c
    a=a//10
print(r) """
