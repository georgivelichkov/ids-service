# The program's primary function is to analyze labeled datasets of network traffic using supervised machine learning (ML) algorithms
# The program will decide if a network traffic instance is an anomaly or normal based on the NSL-KDD dataset.

import joblib
import numpy as np
# Import the required libraries
import pandas as pd

# %%-----------------------------------------------------------------------------------------------------------------------------

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""     LoadDataset()
To make use of a dataset, we need to develop a LoadDataset() function that can convert a
.csv file into a usable set of features. If the file is not found or the format is incorrect,
an error message will be shown.
"""


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Load the dataset
def load_dataset(file_path):  # Loads the network traffic dataset from a CSV file and returns it as a Pandas DataFrame.
    try:
        dataset = pd.read_csv(file_path)  # NSL-KDD Dataset (both training and testing dataset separately)
        return dataset
    except FileNotFoundError:
        print("\nFile not found. Please check the file path and try again.\n")
        return None
    except pd.errors.ParserError:
        print("\nUnable to load the file. Please check if the file is in the correct format.\n")
        return None


# %%-----------------------------------------------------------------------------------------------------------------------------

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""     FeatureSelection()
To improve the accuracy and efficiency of our machine learning model, it is recommended to
apply a feature selection algorithm that can reduce the dimensionality of our dataset,
prevent overfitting, and accelerate training. The FeatureSelection() method can be employed
to select a specific feature selection algorithm and specify the number of features to be
chosen (k number) before initiating the training process.
"""


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def feature_selection(X_train, X_test, k, y_train):
    try:
        # Feature Selection
        clf = LogisticRegression(penalty='l1', solver='liblinear', random_state=50)  # L1-based feature selection
        clf.fit(X_train, y_train)

        # Select top k features based on L1-based feature selection
        coefs = abs(clf.coef_[0])
        top_k_features = np.argsort(coefs)[-k:]

        # Extract top k features
        Xtrain_selected = X_train[:, top_k_features]
        Xtest_selected = X_test[:, top_k_features]

        return Xtrain_selected, Xtest_selected, top_k_features
    except ValueError:
        print("The input data is not valid. Please check if the data is in the correct format. \n")

    # %%-----------------------------------------------------------------------------------------------------------------------------


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""     PreprocessData()
To effectively use our Machine Learning algorithms, it is essential to prepare our data in
advance. This involves encoding categorical data, such as 'Protocol' into a binary sequence
and scaling features with high variance to enhance overall performance.
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Preprocess the data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


def preprocess_data(dataset):
    try:
        X = dataset.iloc[:, 1:-1].values  # Do not include first and last feature (id and class label)
        y = dataset.iloc[:, -1].values  # Only include last feature (class label) (Anomaly = 1, Normal = 0)

        # Encoding categorical data
        ct1 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1, 2, 3])], remainder='passthrough')
        X = np.array(ct1.fit_transform(X))

        # Scaling numerical data
        ct2 = ColumnTransformer(transformers=[('scaler', MinMaxScaler(), [0, 4, 5, 9, 12, 15, 16, 21, 22, 30, 31])],
                                remainder='passthrough')
        X = np.array(ct2.fit_transform(X))

        return X, y

    except ValueError:
        print("Preprocessing failed due to unexpected data type. Please check the dataset.\n")
        return None, None


# %%-----------------------------------------------------------------------------------------------------------------------------

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""     TrainModel()
To train the machine learning model, this method requires the X and y training data from
our dataset as well as the name of the algorithm to be used. There are several algorithms
supported by this method, including KNN, Random Forest, SVM, XGBoost, Logistic Regression,
Naive Bayes, and Decision Tree. Once the training is complete, the method will return the
trained classifier which can be used for subsequent evaluation.
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Train a machine learning model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


def train_model(X, y, algorithm):
    if algorithm == 'Random Forest':
        classifier = RandomForestClassifier(n_estimators=100, random_state=42,
                                            class_weight={0: 1, 1: 20})
    elif algorithm == 'KNN':
        classifier = KNeighborsClassifier(n_neighbors=10)
    elif algorithm == 'Logistic Regression':
        classifier = LogisticRegression(random_state=42, C=5, max_iter=25000)
    elif algorithm == 'Naive Bayes':
        smote = SMOTE()  # Attempt to improve Naive Bayes performance specifically
        X_resampled, y_resampled = smote.fit_resample(X, y)
        classifier = GaussianNB(var_smoothing=0.0001)
        classifier.threshold = 0.3
        classifier.fit(X_resampled, y_resampled)
        return classifier
    elif algorithm == 'Decision Tree':
        classifier = DecisionTreeClassifier(criterion='entropy', random_state=42)
    elif algorithm == 'XGBoost':
        classifier = XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.1, random_state=42)

    try:
        classifier.fit(X, y)
        return classifier

    except ValueError:
        print("Training failed due to invalid algorithm input. Please check the values/name.\n")
        return None


# %%-----------------------------------------------------------------------------------------------------------------------------


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""     EvaluateModel()
This approach builds on the TrainModel() method by taking the trained model classifier
as input and generating a set of evaluation metrics, including Accuracy, Confusion Matrix,
Precision, Recall, F1-Score, and AUC-Score. The Cross-validation score can be adjusted by
specifying the number of folds as an argument.
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Evaluate a machine learning model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, accuracy_score, confusion_matrix


# Print a list of evaluation metrics for each algorithm utilized in training, use the return value of TrainModel() as parameter input
def evaluate_model(classifier, X_test, y_test, algorithm):
    if classifier is None or X_test is None or y_test is None:
        print("Invalid input. Please check if the model, test data, and algorithm are provided.")
        return

    # Evaluate the model
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    confusion_mat = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred) * 100
    recall = recall_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc_score = auc(fpr, tpr) * 100

    print(f"\nResults for {algorithm} algorithm:")
    print(f"Accuracy: {accuracy:.2f}%")  # Accuracy
    print(f"Precision: {precision:.2f}%")  # Proportion of predicted positives that are truly positive
    print(f"Recall: {recall:.2f}%")  # Proportion of actual positives that are correctly identified
    print(f"F1-Score: {f1:.2f}%")  # Harmonic mean of precision and recall
    print(f"AUC: {auc_score:.2f}%")  # area under the ROC ((Receiver Operating Characteristic) curve
    print(f"Confusion matrix: {confusion_mat}")  # Ratio of TP, FP, FN, & TN

    # Calculate cross-validation score (cv = 5 cross fold validation)
    cv_score = cross_val_score(classifier, X_test, y_test, cv=5, scoring='accuracy').mean() * 100
    print(f"Cross-validation score: {cv_score:.2f}%")


# %%-----------------------------------------------------------------------------------------------------------------------------

def main():
    # Load the dataset
    train_dataset = load_dataset('./data/KDDTrain.csv')
    test_dataset = load_dataset('./data/KDDTest.csv')

    if (train_dataset is not None) & (test_dataset is not None):

        # Preprocess the datasets
        X_train, y_train = preprocess_data(train_dataset)
        X_test, y_test = preprocess_data(test_dataset)

        # Feature selection (X of the training set, X of the testing set, FS algorithm name, k number of features)
        Xtrain_selected, Xtest_selected, top_k_features = feature_selection(X_train, X_test, 41, y_train)

        # Train the models
        knn_classifier = train_model(Xtrain_selected, y_train, 'KNN')
        nb_classifier = train_model(Xtrain_selected, y_train, 'Naive Bayes')
        rf_classifier = train_model(Xtrain_selected, y_train, 'Random Forest')
        lr_classifier = train_model(Xtrain_selected, y_train, 'Logistic Regression')
        dt_classifier = train_model(Xtrain_selected, y_train, 'Decision Tree')
        xgboost_classifier = train_model(Xtrain_selected, y_train, 'XGBoost')

        # Evaluate the models
        evaluate_model(knn_classifier, Xtest_selected, y_test, 'KNN')
        evaluate_model(nb_classifier, Xtest_selected, y_test, 'Naive Bayes')
        evaluate_model(rf_classifier, Xtest_selected, y_test, 'Random Forest')
        evaluate_model(lr_classifier, Xtest_selected, y_test, 'Logistic Regression')
        evaluate_model(dt_classifier, Xtest_selected, y_test, 'Decision Tree')
        evaluate_model(xgboost_classifier, Xtest_selected, y_test, 'XGBoost')

        # Save to file in the current working directory
        joblib.dump(knn_classifier, 'knn_model.pkl')

    else:
        print("Missing one or more datasets.\n")  # One or more of the datasets could not be found


# %%-----------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
