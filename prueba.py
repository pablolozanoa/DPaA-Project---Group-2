import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import pandas as pd
df = pd.read_csv("nasa.csv")

#Removing object data types and unnecessary features (IDs)
df.drop(['Name', 'Neo Reference ID', 'Close Approach Date', 'Orbiting Body', 'Orbit Determination Date', 'Equinox'], axis=1, inplace=True)

#Rename label "Hazardous" --> "Class"
df.rename(columns={'Hazardous': 'Class'}, inplace=True)

#Use 0 (true) or 1 (false) in "Class"
df['Class'].replace({True: 0, False: 1}, inplace=True)

#Calculate the correlation matrix
correlation_matrix = df.corr()

#Plot a heatmap of the correlation matrix
plt.figure(figsize=(20, 20))
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix between Features', size=15)
plt.show()

c_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.9:
            col = correlation_matrix.columns[i]
            c_features.add(col)

for feature in c_features:
    print("The correlated feature {} is going to be eliminated".format(feature))
print('')

df.drop(labels=c_features, axis=1, inplace=True)

#Calculate the correlation matrix
correlation_matrix = df.corr()

#Plot a heatmap of the correlation matrix
plt.figure(figsize=(20, 20))
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix between Features', size=15)
plt.show()

# Get the list of features (columns) in the DataFrame
features = df.columns

# Calculate the number of rows and columns for the subplots
num_features = len(features)
num_rows = (num_features + 2) // 3  # Round up to the nearest integer
num_cols = min(num_features, 3)

# Create subplots with the specified number of rows and columns
fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, num_rows * 4))

# Flatten the axes array if necessary
if num_rows == 1:
    axes = axes.reshape(1, -1)
elif num_cols == 1:
    axes = axes.reshape(-1, 1)

# Create a separate boxplot for each feature
for i, feature in enumerate(features):
    row = i // num_cols
    col = i % num_cols
    df[[feature]].boxplot(ax=axes[row, col])

    # Set title and labels
    axes[row, col].set_title(f'Boxplot of {feature}')
    axes[row, col].set_ylabel('Values')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()