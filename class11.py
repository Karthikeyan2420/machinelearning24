""" from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Standardize the features
X = StandardScaler().fit_transform(X)

# Apply PCA
pca = PCA(n_components=3)  # Reduce to 2 dimensions
X_pca = pca.fit_transform(X)

# Plot the results
plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')

plt.show() """


""" from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Standardize the features
X = StandardScaler().fit_transform(X)

# Apply PCA
pca = PCA(n_components=3)  # Reduce to 2 dimensions
X_pca = pca.fit_transform(X)

# Plot the results
plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')

plt.show() """

""" from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Sample text data
text = "new text in this line"

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Display the word cloud using matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()  """

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Step 1: Read the CSV file
csv_file = "adult.csv"
df = pd.read_csv(csv_file)

# Step 2: Extract text data from the 'text' column
# Combine all text into a single string
text = " ".join(df['nativeCountry'].astype(str))

# Step 3: Generate the WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Step 4: Display the WordCloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

