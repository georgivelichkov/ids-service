# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


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

# Feature selection and data preprocessing before executing training/evaluation
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif

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
        Xtrain_selected = X_train[:, top_k_features-1]
        Xtest_selected = X_test[:, top_k_features-1]

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