""" Multinomial regression is a statistical method used to model the relationship between multiple 
categorical dependent variables and one or more independent variables. It is an extension of binary logistic 
regression. """
""" 

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the multinomial logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)  """



""" 
#survival analysis
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# Example dataset
data = {
    'time': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    'event': [1, 1, 0, 1, 1, 0, 1, 0, 1, 1],  # 1 for event occurred, 0 for event censored
    'group': ['A', 'A', 'B', 'A', 'B', 'A', 'B', 'B', 'A', 'A']  # Example grouping variable
}

df = pd.DataFrame(data)

# Fit Kaplan-Meier estimator
kmf = KaplanMeierFitter()
kmf.fit(df['time'], event_observed=df['event'])

# Plot Kaplan-Meier survival curve
kmf.plot()
plt.title('Kaplan-Meier Survival Curve')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.show()

# Compute median survival time
median_survival_time = kmf.median_survival_time_
print("Median Survival Time:", median_survival_time)

# Compare survival curves between groups
groups = df['group'].unique()
for group in groups:
    group_data = df[df['group'] == group]
    kmf.fit(group_data['time'], event_observed=group_data['event'])
    kmf.plot(label=group)
plt.title('Kaplan-Meier Survival Curve by Group')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.legend()
plt.show() """



""" 
import requests
from bs4 import BeautifulSoup
url = 'https://www.wikipedia.org/'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
links = soup.find_all('a')

for link in links:
    print(link.get('href'))  """


""" 
from math import sqrt

def euclidean_distance(point1, point2):
  
    distance = sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))
    return distance

# Example usage:
point1 = (1, 2)
point2 = (4, 6)
distance = euclidean_distance(point1, point2)
print("Euclidean distance:", distance)



def manhattan_distance(point1, point2):
    
    distance = sum(abs(a - b) for a, b in zip(point1, point2))
    return distance

# Example usage:
point1 = (1, 2)
point2 = (4, 6)
distance = manhattan_distance(point1, point2)
print("Manhattan distance:", distance)
 """

import numpy as np
import matplotlib.pyplot as plt

# Function to calculate Manhattan distance between two points
def manhattan_distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

# Define points
points = [(1, 2), (4, 6), (7, 3), (2, 9)]
start_point = (2, 2)  # Example start point
destination_point = (8, 8)  # Example destination point

# Plotting the points
plt.figure(figsize=(8, 8))
plt.scatter([point[0] for point in points], [point[1] for point in points], color='blue', label='Points')
plt.scatter(start_point[0], start_point[1], color='green', label='Start Point', marker='s', s=100)
plt.scatter(destination_point[0], destination_point[1], color='red', label='Destination Point', marker='s', s=100)

for i, (x, y) in enumerate(points):
    plt.text(x, y, f'({x}, {y})', fontsize=12, ha='right', va='bottom')

# Calculate and plot Manhattan distances
for point in points:
    start_distance = manhattan_distance(start_point, point)
    dest_distance = manhattan_distance(destination_point, point)
    
    plt.plot([start_point[0], point[0]], [start_point[1], point[1]], linestyle='--', color='green', alpha=0.5)
    plt.plot([destination_point[0], point[0]], [destination_point[1], point[1]], linestyle='--', color='red', alpha=0.5)
    
    plt.text((start_point[0] + point[0]) / 2, (start_point[1] + point[1]) / 2, f'{start_distance:.2f}', fontsize=10, color='green')
    plt.text((destination_point[0] + point[0]) / 2, (destination_point[1] + point[1]) / 2, f'{dest_distance:.2f}', fontsize=10, color='red')

plt.title('Manhattan Distance between Fixed Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.legend()
plt.show()

