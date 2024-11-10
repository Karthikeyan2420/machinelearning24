# import pandas as pd
# corona_df=pd.read_csv('corona_dataset.csv')
# print(corona_df)

# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.boxplot(x="Country/Region",y="Deaths",data=corona_df)
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the CSV file
# file_path = 'corona_dataset.csv'  # Replace with your actual file path
# data = pd.read_csv(file_path)

# # Select the first fifteen rows
# first_fifteen = data.head(15)

# # Plot a histogram of the 'Country' and 'Confirmed Deaths' columns
# plt.figure(figsize=(10, 6))
# plt.bar(first_fifteen['Country/Region'], first_fifteen['Deaths'])
# plt.xlabel('Country')
# plt.ylabel('Confirmed Deaths')
# plt.title('Confirmed Deaths by Country (First Fifteen Rows)')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()


# corona_df.Latitude.plot.hist()
# plt.show()
# corona_df.Longitude.value_counts().plot.bar()
# plt.show()

import pandas as pd
corona_df=pd.read_csv('corona_dataset.csv')
# import seaborn as sns
# import matplotlib.pyplot as plt

# sns.boxplot( x="WHO Region", y='Deaths / 100 Cases', data=corona_df, )
# plt.show()

# import seaborn as sns
# import matplotlib.pyplot as plt 

# sns.pairplot(corona_df, hue='WHO Region', height=2)
# plt.show()
# import seaborn as sns
# import matplotlib.pyplot as plt 

# from scipy.stats import skew
# from scipy.stats import kurtosis
# # Assuming adult_df is your DataFrame
# skewness = skew(corona_df["Deaths"], axis=0, bias=True)
# #kurtosis1=kurtosis(adult_df, axis=0, bias=True)
# # Display skewness value
# print("Skewness:", skewness)

# Create a histogram
# plt.hist(corona_df["WHO Region"], bins=50, color='blue', alpha=0.7)
# plt.title("Corona Details")
# plt.xlabel("WHO Region")
# plt.ylabel("Deaths / 100 Cases")
# plt.show()

""" import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load your dataset
# Example: df = pd.read_csv("your_dataset.csv")

# Step 2: Preprocess your data
# Assuming "df" contains both numerical and categorical columns
# Separate numerical and categorical columns
numerical_cols = corona_df.select_dtypes(include=['int']).columns
categorical_cols = corona_df.select_dtypes(include=['object']).columns

# One-hot encode categorical columns
encoder = OneHotEncoder()
encoded_categorical_cols = encoder.fit_transform(corona_df[categorical_cols])
encoded_categorical_cols = pd.DataFrame(encoded_categorical_cols, columns=encoder.get_feature_names_out(categorical_cols))

# Concatenate numerical and encoded categorical columns
X = pd.concat([corona_df[numerical_cols], encoded_categorical_cols], axis=1)
y = corona_df["Confirmed"]

# Step 3: Split your dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Instantiate and train the KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 6: Tune hyperparameters (optional) """
""" import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load your dataset
corona_df = pd.read_csv("corona_dataset.csv")

# Step 2: Preprocess your data
# Assuming "df" contains both numerical and categorical columns
# Separate numerical and categorical columns
numerical_cols = corona_df.select_dtypes(include=['int64']).columns
categorical_cols = corona_df.select_dtypes(include=['object']).columns

# One-hot encode categorical columns
encoder = OneHotEncoder()
encoded_categorical_cols = encoder.fit_transform(corona_df[categorical_cols])
encoded_categorical_cols = pd.DataFrame(encoded_categorical_cols.toarray(), columns=encoder.get_feature_names_out(categorical_cols))

# Concatenate numerical and encoded categorical columns
X = pd.concat([corona_df[numerical_cols], encoded_categorical_cols], axis=1)
y = corona_df["Confirmed"]

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

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load the dataset
df = pd.read_csv("corona_dataset.csv")

# Separate features (X) and target variable (y)
X = df.drop(columns=['New deaths']).values  # Features
y = df['New deaths'].values  # Target variable

# Encode categorical features to numerical values
label_encoders = [LabelEncoder() for _ in range(X.shape[1])]
for i in range(X.shape[1]):
    X[:, i] = label_encoders[i].fit_transform(X[:, i])

gnb = GaussianNB()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
