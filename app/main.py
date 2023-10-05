import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from feature_extractor import fingerprint_features
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import argparse
import os
import joblib
from flask import Flask, request
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

appFlask = Flask(__name__)

def save_model(model, valid_data, model_name):
    if not os.path.isdir(model_name):
            os.makedirs(model_name)
    valid_data.to_csv(f"{model_name}/dataset_for_validation.csv")
    filename = f'{model_name}.sav'
    joblib.dump(model, f"{model_name}/{filename}")
    print(f"{model_name} succesfully saved")


def get_fingerprint_features_for_numpy(dataset, col):
    dataset["fingerprint_features"] = dataset[col].apply(
        lambda x: np.asarray(fingerprint_features(x)).astype("float32")
    )
    return dataset["fingerprint_features"].apply(pd.Series).to_numpy()


def get_vectorize_features_for_numpy(dataset, col):
    vectorizer = CountVectorizer(binary=True)
    matrix = vectorizer.fit_transform(dataset[col])
    return matrix.toarray(), vectorizer


def split_data(dataset_name, test_size, model):
    dataset = pd.read_csv(dataset_name)
    vectorizer = None
    if model == "Model1":
        X = get_fingerprint_features_for_numpy(dataset, "smiles")
    else:
        X, vectorizer = get_vectorize_features_for_numpy(dataset, "smiles")
    y = dataset["P1"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test, dataset.iloc[y_test.index].reset_index(drop=True), vectorizer

def train_model(test_size, model="Model1"):
    X_train, X_test, y_train, y_test, dataset_for_test, vectorizer = split_data(
        "dataset/dataset_single.csv", test_size, model)
    print(f"Fit {model} model on training set")
    if model == "Model1":
        classifier = SVC(kernel='rbf', random_state=0, class_weight={0:0.82, 1:0.18},)
        classifier.fit(X_train, y_train)
        return classifier, dataset_for_test, None
    else:
        mnb = MultinomialNB()
        mnb.fit(X_train, y_train, sample_weight=compute_sample_weight(class_weight={0:0.82, 1:0.18}, y=y_train))
        return mnb, dataset_for_test, vectorizer


def evaluate_model(model="Model1"):
    if os.path.isfile(f"{model}/dataset_for_validation.csv"):
        dataset_valid = pd.read_csv(
            f"{model}/dataset_for_validation.csv")
        loaded_model = joblib.load(f"{model}/{model}.sav")
        print("Predict P1 for valid set")
        if model=="Model1":
            X_valid = get_fingerprint_features_for_numpy(dataset_valid, "smiles")
        else:
            vectorizer = joblib.load(f"{model}/vectorizer.pkl")
            matrix = vectorizer.transform(dataset_valid["smiles"])
            X_valid = matrix.toarray()
        y_pred = loaded_model.predict(X_valid)
        print("Classification Report of the model on valid set")
        print(pd.DataFrame(classification_report(dataset_valid["P1"], y_pred, output_dict=True)).transpose())
        print("Confusion Matrix")
        print(pd.DataFrame(confusion_matrix(dataset_valid["P1"], y_pred)))
    else:
        print("You need to train the model first with arg '--train <model_name>'")

@appFlask.route("/predict", methods = ['POST','GET'])
def predict():
    if request.method == 'POST':
        molecular_formula = request.form['molecular_formula']
        data = pd.DataFrame(data=[molecular_formula], columns=["smiles"])
        print(request.form)
        try:
            if request.form.get("Model1"):
                X_valid = get_fingerprint_features_for_numpy(data, "smiles")
                model = joblib.load("Model1/Model1.sav")
            if request.form.get("Model2"):
                vectorizer = joblib.load(f"Model2/vectorizer.pkl")
                matrix = vectorizer.transform(data["smiles"])
                X_valid = matrix.toarray()
                model = joblib.load("Model2/Model2.sav")
            y_pred = model.predict(X_valid)
            if y_pred[0] == 0:
                return f'The P1 gene is probably not present on molecular {molecular_formula}'
            else:
                return f'The P1 gene is probably present on molecular {molecular_formula}'
        except:
            return 'The molecule name is not valid'
    return '''<form method = "post">
    <p>Enter Formula:</p>
    <p><input type = "text" name = "molecular_formula" /></p>
    <p><input type = "submit" name = "Model1" value = "Predict with SVC" /></p>
    <p><input type = "submit" name = "Model2" value = "Predict with Model2" /></p>
    </form>'''


if __name__ == "__main__":
    # Definition des arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str,
                        choices=["Model1", "Model2"], help="Choose the model you want to train: - 'Model1': SVC - 'Model2': MultinomialNB")
    parser.add_argument('--evaluate', type=str, choices=["Model1", "Model2"], help="Evaluate SVC or Model2")
    parser.add_argument('--predict', action='store_true',
                        help="Prédire la présence du gêne P1 dans la molecule")
    args = parser.parse_args()

    if args.train:
        # Set the proportion of Train (70%) Test (30%)
        print(
            f"Train / test split with the proportion of :\n - {70}% for training\n - {30}% for testing")
        print(f"Start of training {args.train}")
        if args.train in ["Model1", "Model2"]:
            classifier, dataset_valid, vectorizer = train_model(
                0.3, args.train)
            save_model(classifier, dataset_valid, args.train)
            if vectorizer:
                joblib.dump(vectorizer, f"Model2/vectorizer.pkl")

    if args.evaluate in ["Model1", "Model2"]:
        evaluate_model(args.evaluate)

    if args.predict:
        print("To predict the presence of P1 gene go to this url http://localhost:5000/predict")
        appFlask.run(debug=False)
