# PCA 로 3차원으로 데이터를 시각화 했을 때의 결과. SVM 을 사용하지 않은 이유 설명할 때 사용한 코드.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# Load the training dataset
train_file_path = 'Classification_training_data.csv'  # Update this path
train_data = pd.read_csv(train_file_path)

# Splitting features and target in the training dataset
X_train = train_data.loc[:, '0HZ':'MAX0']
y_train = train_data['leaktype']

# Standardizing the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Applying PCA to reduce the data to 3 dimensions
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_scaled)

# Load the new test dataset
test_file_path = 'Classification_testing_data.csv'  # Update this path
test_data = pd.read_csv(test_file_path)

# Splitting features and target in the test dataset
X_test_new = test_data.loc[:, '0HZ':'MAX0']
y_test_new = test_data['leaktype']

# Standardizing the test data
X_test_new_scaled = scaler.transform(X_test_new)

# Applying PCA to the test data
X_test_new_pca = pca.transform(X_test_new_scaled)

# Encoding the target labels as numeric values
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_new_encoded = label_encoder.transform(y_test_new)

# Custom class names
class_names = ['in', 'noise', 'normal', 'other', 'out']

# Plotting the 3D scatter plot for the training data
def plot_3d_scatter(X, y, title, class_names):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', edgecolor='k', s=40)
    ax.set_title(title)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    
    # Creating custom legend
    handles, _ = scatter.legend_elements()
    legend_labels = [class_names[int(label)] for label in np.unique(y)]
    ax.legend(handles, legend_labels, title="Classes")
    
    plt.show()

# Plotting the 3D scatter plot for the training and test data
plot_3d_scatter(X_train_pca, y_train_encoded, '3D PCA Plot of Training Data', class_names)
# plot_3d_scatter(X_test_new_pca, y_test_new_encoded, '3D PCA Plot of Test Data', class_names)
