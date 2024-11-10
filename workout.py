import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA

# Load the data
data = pd.read_csv("adult1.csv")

# Handling missing values
imputer = SimpleImputer(strategy="most_frequent")
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Encoding categorical variables
categorical_cols = data.select_dtypes(include=["object"]).columns
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])
print(label_encoders)
# Standardize numerical features
scaler = StandardScaler()
data[data.select_dtypes(include=["number"]).columns] = scaler.fit_transform(data.select_dtypes(include=["number"]))

# Dropping unnecessary columns
data.drop(["education"], axis=1, inplace=True)  # Dropping 'education' column as 'education_num' represents similar information

# Implementing k-medoids clustering
k = 3  # Number of clusters
kmedoids = KMedoids(n_clusters=k, metric="euclidean", random_state=42)
clusters = kmedoids.fit_predict(data)

# Applying PCA for dimensionality reduction
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

# Plotting the clusters
plt.figure(figsize=(8, 6))
for i in range(k):
    plt.scatter(data_pca[clusters == i, 0], data_pca[clusters == i, 1], label=f'Cluster {i+1}')

plt.title('K-Medoids Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()
