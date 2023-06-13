import json

import joblib as joblib
import numpy as np

from utils.prediction_utils import load_dataset, preprocess_data, feature_selection


def access_log_classification():
    csv_file = load_dataset('./data/access_log.csv')

    predictions = np.array(make_prediction(csv_file))

    sanitize_data(predictions, csv_file)

    return {'message': 'Access Log classification has been completed!'}


def sanitize_data(predictions, csv_file):
    blacklisted_records = []
    whitelisted_records = []
    blacklisted = np.where(predictions == 1)
    whitelisted = np.where(predictions == 0)

    for blacklist_index in blacklisted[0]:
        id = csv_file.iloc[blacklist_index][0]
        ip = csv_file.iloc[blacklist_index][csv_file.columns[-1]]
        blacklist_record = {"id": str(id), "ip": ip}
        blacklisted_records.append(blacklist_record)

    for whitelist_index in whitelisted[0]:
        id = csv_file.iloc[whitelist_index][0]
        ip = csv_file.iloc[whitelist_index][csv_file.columns[-1]]
        whitelist_record = {"id": str(id), "ip": ip}
        whitelisted_records.append(whitelist_record)

    save_predictions(blacklisted_records, './data/blacklist.json')
    save_predictions(whitelisted_records, './data/whitelist.json')


def make_prediction(record):
    model = joblib.load('./model/knn_model.pkl')
    train_dataset = load_dataset('./model/KDDTrain.csv')

    X_train, y_train = preprocess_data(train_dataset)
    X_test, y_test = preprocess_data(record)

    Xtrain_selected, Xtest_selected, top_k_features = feature_selection(X_train, X_test, 41, y_train)

    prediction = model.predict(Xtest_selected)
    return prediction


def save_predictions(predictions, filepath):
    # Serializing json
    json_object = json.dumps(predictions, indent=4)

    # Writing to sample.json
    with open(filepath, "w") as outfile:
        outfile.write(json_object)
