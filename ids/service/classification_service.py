import json

import joblib as joblib

from utils.prediction_utils import load_dataset, preprocess_data, feature_selection


def access_log_classification():
    csv_file = load_dataset('./data/access_log.csv')

    predictions = make_prediction(csv_file)

    blacklisted = predictions[predictions == 1]
    whitelisted = predictions[predictions == 0]

    save_predictions(blacklisted, './data/blacklist.json')
    save_predictions(whitelisted, './data/whitelist.json')

    return {'message': str(len(blacklisted)) + ' ips have been blacklisted!'}


def make_prediction(record):
    model = joblib.load('./model_train/knn_model.pkl')
    train_dataset = load_dataset('./model_train/data/KDDTrain.csv')

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
