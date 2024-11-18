# 1. 결정트리를 사용해 중요한 열 (important features)을 뽑는걸 시각화하고, 그에 대한 confusion matrix를 나타내는 코드
# 2. Decision Tree 를 사용했을 떄의 결과 -- KNN 모델보다 성능이 떨어지는 것을 볼 수 있다.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Load the training dataset
train_file_path = 'Classification_training_data.csv'  # Update this path
train_data = pd.read_csv(train_file_path)

# Splitting features and target in the training dataset
X_train = train_data.loc[:, 'lrate':'MAX0'].drop(columns=['leaktype'])
y_train = train_data['leaktype']

# Standardizing the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Splitting the training data into training and validation sets


# Implementing Decision Tree classification
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train_scaled, y_train)

# Making predictions on the validation set

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


# Load the new test dataset
test_file_path = 'Classification_testing_data.csv'  # Update this path
test_data = pd.read_csv(test_file_path)

# Dropping irrelevant columns in the test dataset
# test_data = test_data.loc[:,'0HZ':'MAX0']

# Splitting features and target in the test dataset
X_test_new = test_data.loc[:, 'lrate':'MAX0'].drop(columns=['leaktype'])
y_test_new = test_data['leaktype']

# Standardizing the test data
X_test_new_scaled = scaler.transform(X_test_new)

# Making predictions on the new test dataset
y_pred_new = decision_tree.predict(X_test_new_scaled)

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

plot_confusion_matrix(cm_new, "Test")

# Displaying accuracy, precision, and recall
def display_metrics(accuracy, precision, recall, title):
    print(f"{title} Accuracy: {accuracy}")
    print(f"{title} Precision: {precision}")
    print(f"{title} Recall: {recall}")


display_metrics(accuracy_new, precision_new, recall_new, "Test")

# Identifying the top 20 most important features
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
top_20_features = feature_importance_df.head(70)

# Display the top 20 features
print("Top 70 most important features:")
print(top_20_features)

# Plot the top 20 features
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=top_20_features)
plt.title('Top 70 Most Important Features')
plt.show()


