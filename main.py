import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from feature_extractor import fingerprint_features
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import argparse
import os
import joblib

def save_model(model, valid_data):
    if not os.path.isdir("Model1"):
        os.makedirs("Model1")
    valid_data.to_csv("Model1/dataset_for_validation.csv")
    filename = 'model1.sav'
    joblib.dump(model, f"Model1/{filename}")


def get_test_size(train_prop: 0.7, test_prop: 0.2):
    valid_prop = 1 - train_prop - test_prop
    test_size = 1 - train_prop
    valid_size = valid_prop/test_size
    return test_size, valid_size


def get_fingerprint_features_for_numpy(dataset, col):
    dataset["fingerprint_features"] = dataset[col].apply(
        lambda x: np.asarray(fingerprint_features(x)).astype("float32")
    )
    return dataset["fingerprint_features"].apply(pd.Series).to_numpy()


def split_data(dataset_name, test_size, valid_size):
    single = pd.read_csv(dataset_name)
    X = get_fingerprint_features_for_numpy(single, "smiles")
    y = single["P1"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)
    X_test, X_valid, y_test, y_valid = train_test_split(
        X_test, y_test, test_size=valid_size, random_state=42, stratify=y_test)
    single = single.iloc[y_test.index].reset_index(drop=True)
    return X_train, X_test, y_train, y_test, single


def train_model(test_size, valid_size, model="Model1"):
    X_train, X_test, y_train, y_test, single = split_data(
        "dataset/dataset_single.csv", test_size, valid_size)
    print("Fit on training set")
    if model == "Model1":
        classifier = SVC(kernel='rbf', random_state=0, class_weight={0:0.82, 1:0.18},)
        classifier.fit(X_train, y_train)
        # scores = cross_val_score(classifier, X_train, y_train, cv=10)
        # print(f"Scores: {scores}")
        return classifier, single


def evaluate_model(model=1):
    if os.path.isfile("Model1/dataset_for_validation.csv"):
        dataset_valid = pd.read_csv(
            "Model1/dataset_for_validation.csv")
        loaded_model = joblib.load("Model1/model1.sav")
        print("Predict P1 for valid set")
        X_valid = get_fingerprint_features_for_numpy(dataset_valid, "smiles")
        y_pred = loaded_model.predict(X_valid)
        print("Classification Report of the model on valid set")
        print(classification_report(dataset_valid["P1"], y_pred))
    else:
        print("You need to train the model first with arg '--train=Model1'")


if __name__ == "__main__":
    # Definition des arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str,
                        choices=["Model1", "Model2"], help="Choose the model you want to train:\n - 'Model1': SVC\n - 'Model2': Other")
    parser.add_argument('--evaluate', type=str, choices=["SVC", "Model2"], help="Evaluate SVC or Model2")
    parser.add_argument('--predict',
                        help="Prédire la présence du gêne P1 dans la molecule")
    args = parser.parse_args()

    if args.train:
        # Set the proportion of Train (70%) Test (20%) and Valid (10%) by default
        test_size, valid_size = get_test_size(0.7, 0.2)
        print(
            f"Train / test / valid split with the proportion of :\n - {70}% for training\n - {20}% for testing\n - {10}% to valid our model")
        print(f"Start of training {args.train}")
        if args.train == "Model1":
            classifier, dataset_valid = train_model(
                test_size, valid_size, args.train)
        # if args.train == "Model2":
            # train_model()
        save_model(classifier, dataset_valid)

    if args.evaluate == "SVC":
        evaluate_model()

    if args.predict:
        print(args.predict)
        print(f"Predict the presence of P1 genom in molecule: {args.predict}")
        data = pd.DataFrame(data=[args.predict], columns=["smiles"])
        loaded_model = joblib.load("Model1/model1.sav")
        X_valid = get_fingerprint_features_for_numpy(data, "smiles")
        y_pred = loaded_model.predict(X_valid)
        if y_pred[0] == 1:
            print(f"The P1 genom is present")
        else:
            print("The P1 genom isn't present")
