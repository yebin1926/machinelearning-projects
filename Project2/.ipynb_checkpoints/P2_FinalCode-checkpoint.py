# 최종 결과코드입니다

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn.svm import SVC

# Load the training dataset
train_file_path = 'Classification_training_data.csv'  # Update this path
train_data = pd.read_csv(train_file_path)

# Splitting features and target in the training dataset
X_train = train_data.loc[:, 'lrate':'MAX0'].drop(columns=['leaktype'])
y_train = train_data['leaktype']

# Standardizing the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Implementing Decision Tree classification to identify top 20 features
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train_scaled, y_train)

# Identifying the top 70 most important features
feature_importances = decision_tree.feature_importances_
features = X_train.columns

# Create a DataFrame for feature importances
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Select the top 20 features
top_70_features = feature_importance_df.head(70)['Feature']
# Create new datasets with the top 20 features
X_train_top_70 = X_train[top_70_features]

# Standardizing the new datasets
X_train_top_70_scaled = scaler.fit_transform(X_train_top_70)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_top_70_scaled, y_train)

# Evaluating the model on the validation set
def confusion_matrix_metrics(y_true, y_pred):
    unique_classes = np.unique(y_true)
    cm = {cls: {cls_: 0 for cls_ in unique_classes} for cls in unique_classes}
    
    for true, pred in zip(y_true, y_pred):
        cm[true][pred] += 1
    
    accuracy = np.sum([cm[cls][cls] for cls in unique_classes]) / len(y_true)
    
    precision = {cls: cm[cls][cls] / sum(cm[cls].values()) if sum(cm[cls].values()) > 0 else 0 for cls in unique_classes}
    recall = {cls: cm[cls][cls] / sum([cm[cls_][cls] for cls_ in unique_classes]) if sum([cm[cls_][cls] for cls_ in unique_classes]) > 0 else 0 for cls in unique_classes}
    
    return cm, accuracy, precision, recall

# cm_val, accuracy_val, precision_val, recall_val = confusion_matrix_metrics(y_val, y_val_pred)

# Load the new test dataset
test_file_path = 'Classification_testing_data.csv'  # Update this path
test_data = pd.read_csv(test_file_path)

# Dropping irrelevant columns in the test dataset
#test_data = test_data.loc[:,'0HZ':'MAX0']
# Splitting features and target in the test dataset
X_test_new = test_data.loc[:,'lrate':'MAX0'].drop(columns=['leaktype'])
X_test_new=X_test_new[top_70_features]
y_test_new = test_data['leaktype']

# Standardizing the test data
X_test_new_scaled = scaler.transform(X_test_new)

# Making predictions on the new test dataset
y_pred_new = knn.predict(X_test_new_scaled)
# y_pred_new = decision_tree.predict(X_test_new_scaled)
# y_pred_new = svm.predict(X_test_new_scaled)

# Evaluating the model on the new test dataset
cm_new, accuracy_new, precision_new, recall_new = confusion_matrix_metrics(y_test_new, y_pred_new)

# Function to plot confusion matrix
def plot_confusion_matrix(cm, title):
    cm_df = pd.DataFrame(cm)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{title} Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Plotting confusion matrices
# plot_confusion_matrix(cm_val, "Validation")
plot_confusion_matrix(cm_new, "Test")

# Displaying accuracy, precision, and recall
def display_metrics(accuracy, precision, recall, title):
    print(f"{title} Accuracy: {accuracy}")
    print(f"{title} Precision: {precision}")
    print(f"{title} Recall: {recall}")

# display_metrics(accuracy_val, precision_val, recall_val, "Validation")
display_metrics(accuracy_new, precision_new, recall_new, "Test")

