import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# 1. Prepare labels from epoch
labels = epochs.events[:, 2] 

# 2. Split into training and test sets
'''
The code randomly splits the feature data (features) and labels (labels) into training (80%) and test (20%) sets. 
Stratification preserves the label distribution, ensuring each class is fairly represented in both subsets.
'''
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

# 3. Feature Standardization (important for SVMs and most ML algorithms)
'''
EEG features are standardized—transformed so each feature has zero mean and unit variance. 
This step improves model convergence and accuracy, especially for 
algorithms sensitive to feature scaling like SVMs.
'''
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Train the classifier
'''
An SVM classifier with a radial basis function (RBF) kernel is initialized and trained on the standardized 
training data. This allows the model to learn the mapping from EEG-derived features to class labels.
'''
clf = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
clf.fit(X_train, y_train)

# 5. Evaluate the classifier
'''
The trained model predicts the labels for the test data. A performance summary is printed, 
including accuracy, precision, recall, F1-score, and the confusion matrix, 
which shows the distribution of true versus predicted classes.
'''
y_pred = clf.predict(X_test)
print("Classification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# 6. (Optional) Cross-validation for robust performance estimate
'''
performs 5-fold cross-validation to robustly estimate the classification model's accuracy 
and generalizability using scikit-learn's cross_val_score function.
'''
scores = cross_val_score(clf, scaler.transform(features), labels, cv=5)
print("Cross-validation accuracy: {:.2f}% (+/- {:.2f})".format(scores.mean()*100, scores.std()*100))
