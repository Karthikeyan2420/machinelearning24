import pandas as pd
import numpy as np
adult_df = pd.read_csv('adult.csv')

#print(adult_df.head(15))
#print(adult_df)
#print(adult_df.tail(15)) 
# print(type(adult_df.age) )
#print(adult_df.age) 
# print(type(adult_df))
#print(adult_df.loc[0].index)
#print(adult_df.loc[1])
#print(adult_df.age.index)
#print(adult_df.set_index(np.arange(10000,42561),inplace=True))
#print(adult_df.set_index(np.arange(10000,42561)))



# print(adult_df.iloc[0].loc['fnlwgt'])
# print(adult_df.education.loc[5])
# print(adult_df['education'].iloc[2])
# print(adult_df.at[2,'education'])  

# row_series = adult_df.loc[2]
# print(row_series.loc['education'])
# print(row_series.iloc[3])
# print(row_series['education'])
# print(row_series.education)

# columns_series = adult_df.education
# print(columns_series.loc[2])
# print(columns_series.iloc[2])
# print(columns_series[2])


""" # print(row_series.2)  #This will give syntax error!
my_array = np.array([[2, 3, 5, 7], [11, 13, 17, 19],
                     [23, 29, 31, 37], [41, 43, 47, 49]])

print("Original Array:")
print(my_array)

# Indexing and Slicing
print("Value at row 1, column 1:", my_array[1, 1])
print("Values in the second row:", my_array[1, :])
print("Values in the second column:", my_array[:, 1])
print("Values from row 1 to 2, columns 0 to 1:")
print(my_array[0:3, 1:3]) """


""" # Selecting columns using loc
selected_columns = adult_df.loc[0:10, 'education':'occupation']
print("Selected Columns using loc:")
print(selected_columns)  """

# Sorting and selecting rows
# sorted_df = adult_df.sort_values('education_num').reset_index()
# selected_rows = sorted_df.iloc[1:32561:4000]
# print("Selected Rows after Sorting:")
# print(selected_rows)

""" preschool=adult_df.education=='Preschool'
print(preschool)
mopa=np.mean(adult_df[preschool].age)
print(adult_df[preschool].age)
print("Mean value of preschool age : ",mopa) """

""" education_gt=adult_df['education_num']>10
education_lt=adult_df['education_num']<10

print("This is eduction_gt",education_gt)
print("This is education_lt",education_lt)

gtmean=np.mean(adult_df[education_gt].capitalGain)
ltmean=np.mean(adult_df[education_lt].capitalGain)


print("Greater than of 10 Mean value of Capital Gain :",gtmean)
print("Less than of 10 Mean value of Capital Gain :",ltmean) """


""" relationunique=adult_df.relationship.unique()
print(relationunique)

relationcount=adult_df.relationship.value_counts()
print(relationcount) """


import matplotlib.pyplot as plt

# Data Visualization
""" adult_df.age.plot.hist()
plt.show()
print("Unique values in 'relationship':", adult_df.relationship.unique())
print("Value counts for 'relationship':")
print(adult_df.relationship.value_counts())
adult_df.relationship.value_counts().plot.bar()
plt.show() """
# Applying functions to columns
""" def MultiplyBy2(n):
    return n*2

print(adult_df.age.apply(MultiplyBy2)) """

""" total_fnlwgt = adult_df.fnlwgt.sum()

def CalculatePercentage(v):
    return v/total_fnlwgt*100
total_fnlwgt = adult_df.fnlwgt.sum()

adult_df.fnlwgt = adult_df.fnlwgt.apply(lambda v: v/total_fnlwgt*100) """




#print(adult_df.shape)
#print(adult_df.describe())
#print(adult_df.info())
#print(adult_df.nunique())
#print(adult_df.isnull().sum())


import seaborn as sns
import matplotlib.pyplot as plt
 
 
# sns.boxplot( x="age", y='workclass', data=adult_df, )
# plt.show()

""" import seaborn as sns
import matplotlib.pyplot as plt 

 
# sns.pairplot(adult_df, hue='age', height=2)
# plt.show()
from scipy.stats import skew
from scipy.stats import kurtosis
# Assuming adult_df is your DataFrame
skewness = skew(adult_df["age"], axis=0, bias=True)
#kurtosis1=kurtosis(adult_df, axis=0, bias=True)
# Display skewness value
print("Skewness:", skewness)

# Create a histogram
plt.hist(adult_df["age"], bins=50, color='blue', alpha=0.7)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()



from scipy.stats import kurtosis
import matplotlib.pyplot as plt

# Assuming adult_df is your DataFrame
kurtosis1 = kurtosis(adult_df["age"], bias=True)
print("Kurtosis:", kurtosis1)

# Create a histogram
plt.hist(adult_df["age"], bins=50, color='blue', alpha=0.7)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()  """

# Show skewness information on the plot



""" 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset from CSV file
data = pd.read_csv('Company_Data.csv')

# Assuming the last column is the target variable and the rest are features
X = data.iloc[:, :6]  # Features
y = data.iloc[:, 1]   # Target variable
print(X,y)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the decision tree classifier
clf = DecisionTreeClassifier()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)  """


""" def gini_coefficient(values):
    sorted_values = sorted(values)
    n = len(values)
    cumulative_sum = sum(sorted_values)
    gini_num = 0

    for i, value in enumerate(sorted_values):
        gini_num += (i + 1) * value
    gini_den = n * cumulative_sum
    gini_coefficient = (2 * gini_num) / gini_den - (n + 1) / n

    return gini_coefficient

# Example usage
data = adult_df.iloc[:, 0]  # Example data
gini = gini_coefficient(data)
print("Gini Coefficient:", gini) """


'''The Gini index has a maximum impurity 
is 0.5 and maximum purity is 0, whereas Entropy has a maximum impurity of 1 and maximum purity is 0. '''


""" 
import math

def entropy(labels):
    # Count occurrences of each label
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    # Calculate entropy
    total_instances = len(labels)
    entropy_val = 0
    for count in label_counts.values():
        probability = count / total_instances
        entropy_val -= probability * math.log2(probability)

    return entropy_val

# Example usage
labels = ['A', 'B', 'A', 'C', 'B', 'B', 'A', 'C']
entropy_value = entropy(labels)
print("Entropy:", entropy_value) """



""" 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load your dataset
adult_df = pd.read_csv("adult.csv")

# Step 2: Preprocess your data
# Assuming "adult_df" contains both numerical and categorical columns
# Separate numerical and categorical columns
numerical_cols = adult_df.select_dtypes(include=['int', 'float']).columns
categorical_cols = adult_df.select_dtypes(include=['object']).columns

# One-hot encode categorical columns
encoder = OneHotEncoder()
encoded_categorical_cols = encoder.fit_transform(adult_df[categorical_cols])
encoded_categorical_cols = pd.DataFrame(encoded_categorical_cols, columns=encoder.get_feature_names_out(categorical_cols))

# Concatenate numerical and encoded categorical columns
X = pd.concat([adult_df[numerical_cols], encoded_categorical_cols], axis=1)
y = adult_df["age"]

# Step 3: Split your dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Instantiate and train the KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

 """
# Step 6: Tune hyperparameters (optional)
# Example: Tune hyperparameters using grid search or randomized search """


#Decision Tree
""" 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data=pd.read_csv('Company_Data.csv')

X=data.iloc[:,:6]
y=data.iloc[:,5]
#print(X,y)
X_tra,X_tes,Y_tra,y_tes=train_test_split(X,y,test_size=0.2,random_state=42)
clf=DecisionTreeClassifier()
clf.fit(X_tra,Y_tra)
pred=clf.predict(X_tes)
acc=accuracy_score(y_tes,pred)
print(acc) """



""" #GINI

#max impurity is 0.5 maxi purity is 0

def gini_coefficient(val):

    sorval=sorted(val)
    n=len(val)

    cum=sum(sorval)

    gn=0
    for i, val in enumerate(sorval):
        gn+=(i+1)*val
    giniden=n*cum

    gini_coefficient=(2*gn)/giniden-(n+1)/n

    print(gini_coefficient)

data=adult_df.iloc[:,0]
gini_coefficient(data) """


""" # Filtering and calculating statistics
preschool_filter = adult_df.education == 'Preschool'
print('Mean age for Preschool education: {}'.format(np.mean(adult_df[preschool_filter].age)))


education_gt_10 = adult_df['education_num'] > 10
education_lt_10 = adult_df['education_num'] < 10
print(education_gt_10,education_lt_10)
print('Mean capital gain for more than 10 years of education: {}'.format(np.mean(adult_df[education_gt_10].capitalGain)))
print('Mean capital gain for less than 10 years of education: {}'.format(np.mean(adult_df[education_lt_10].capitalGain))) """
"""
# Data Visualization
adult_df.age.plot.hist()
print("Unique values in 'relationship':", adult_df.relationship.unique())
print("Value counts for 'relationship':")
print(adult_df.relationship.value_counts())
adult_df.relationship.value_counts().plot.bar()

# Applying functions to columns
def MultiplyBy2(n):
    return n*2

adult_df.age.apply(MultiplyBy2)

total_fnlwgt = adult_df.fnlwgt.sum()

def CalculatePercentage(v):
    return v/total_fnlwgt*100
total_fnlwgt = adult_df.fnlwgt.sum()

adult_df.fnlwgt = adult_df.fnlwgt.apply(lambda v: v/total_fnlwgt*100)
adult_df
def CalcLifeNoEd(row):
    return row.age - row.education_num

adult_df.apply(CalcLifeNoEd,axis=1)
 """







""" 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset from CSV file
data = pd.read_csv('Company_Data.csv')

# Assuming the last column is the target variable and the rest are features
X = data.iloc[:, :6]  # Features
y = data.iloc[:, 1]   # Target variable
print(X,y)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the decision tree classifier
clf = DecisionTreeClassifier()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)  """


""" def gini_coefficient(values):
    
    # Sort values
    sorted_values = sorted(values)

    # Number of values
    n = len(values)

    # Cumulative sum of values
    cumulative_sum = sum(sorted_values)

    # Gini numerator
    gini_num = 0

    for i, value in enumerate(sorted_values):
        gini_num += (i + 1) * value

    # Gini denominator
    gini_den = n * cumulative_sum

    # Gini coefficient
    gini_coefficient = (2 * gini_num) / gini_den - (n + 1) / n

    return gini_coefficient

# Example usage
data = adult_df.iloc[:, 0]  # Example data
gini = gini_coefficient(data)
print("Gini Coefficient:", gini) """


'''The Gini index has a maximum impurity 
is 0.5 and maximum purity is 0, whereas Entropy has a maximum impurity of 1 and maximum purity is 0. '''


""" 
import math

def entropy(labels):
    # Count occurrences of each label
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    # Calculate entropy
    total_instances = len(labels)
    entropy_val = 0
    for count in label_counts.values():
        probability = count / total_instances
        entropy_val -= probability * math.log2(probability)

    return entropy_val

# Example usage
labels = ['A', 'B', 'A', 'C', 'B', 'B', 'A', 'C']
entropy_value = entropy(labels)
print("Entropy:", entropy_value) """



""" 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load your dataset
# Example: df = pd.read_csv("your_dataset.csv")

# Step 2: Preprocess your data
# Assuming "df" contains both numerical and categorical columns
# Separate numerical and categorical columns
numerical_cols = adult_df.select_dtypes(include=['int', 'float']).columns
categorical_cols = adult_df.select_dtypes(include=['object']).columns

# One-hot encode categorical columns
encoder = OneHotEncoder(sparse=False)
encoded_categorical_cols = encoder.fit_transform(adult_df[categorical_cols])
encoded_categorical_cols = pd.DataFrame(encoded_categorical_cols, columns=encoder.get_feature_names_out(categorical_cols))

# Concatenate numerical and encoded categorical columns
X = pd.concat([adult_df[numerical_cols], encoded_categorical_cols], axis=1)
y = adult_df["age"]

# Step 3: Split your dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Instantiate and train the KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy) """

# Step 6: Tune hyperparameters (optional)
# Example: Tune hyperparameters using grid search or randomized search






""" import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('Tennisdataset1.1.csv')

# Separate features and target variable
X = data.drop(columns=['enjoysport'])
y = data['enjoysport']

# Convert categorical variables to numerical
X = pd.get_dummies(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Decision Tree Classifier
clf = DecisionTreeClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy) """

""" 
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

# Step 1: Load the dataset from CSV
data = pd.read_csv('adult.csv')

# Step 2: Extract the column containing the data you want to analyze
# For example, if your CSV file has a column named 'values', you can do:
values = data['age']

# Step 3: Fit a normal distribution to the data or estimate its parameters
mu, sigma = norm.fit(values)

# Step 4: Optionally, visualize the original data and the fitted normal distribution
plt.hist(values, bins=30, density=True, alpha=0.6, color='g')

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, sigma)
plt.plot(x, p, 'k', linewidth=2)

plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Histogram and Fitted Normal Distribution')
plt.show()

# Print mean and standard deviation of the fitted normal distribution
print("Mean:", mu)
print("Standard Deviation:", sigma) """


""" 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load the data
data = pd.read_csv('adult.csv')  # replace 'data.csv' with the path to your CSV file

# Step 2: Preprocess the data
# Define features and target
X = data.drop('income', axis=1)
y = data['income']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),  # Scale numerical features
        ('cat', OneHotEncoder(), categorical_cols)   # One-hot encode categorical features
    ])

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create a pipeline with preprocessing and SVM model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('svm', SVC(kernel='linear'))  # You can change the kernel as needed
])

# Train the model
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)

# Print accuracy and classification report
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))
 """

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score

# Data preparation
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 
                'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 
                    'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 
                 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 
             'Strong', 'Strong', 'Weak', 'Strong'],
    'enjoysport': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

# Create DataFrame
df = pd.DataFrame(data)

# Convert categorical data to numeric using LabelEncoder
label_encoders = {}
for column in df.columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Split the data into features and target
X = df.drop('enjoysport', axis=1)
y = df['enjoysport']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the CategoricalNB model
model = CategoricalNB()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Check the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Test the model with a new sample
new_sample = pd.DataFrame({
    'Outlook': ['Sunny'], 
    'Temperature': ['Cool'], 
    'Humidity': ['High'], 
    'Wind': ['Weak']
})

# Convert the new sample to numerical form using the same LabelEncoders
for column in new_sample.columns:
    new_sample[column] = label_encoders[column].transform(new_sample[column])

# Predict whether the person enjoys the sport or not
prediction = model.predict(new_sample)
print(f"Prediction: {label_encoders['enjoysport'].inverse_transform(prediction)[0]}")
