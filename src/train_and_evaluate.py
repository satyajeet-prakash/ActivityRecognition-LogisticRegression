import os
from tabnanny import verbose
import yaml
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from get_data import read_params
import argparse
import joblib
import json


def eval_metrics(actual, pred, average):
    score_accuracy = accuracy_score(actual, pred)
    score_precision = precision_score(actual, pred, average=average)
    score_recall = recall_score(actual, pred, average=average)
    score_f1 = f1_score(actual, pred, average=average)
    return score_accuracy, score_precision, score_recall, score_f1


def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    solver = config["algorithm_params"]["solver"]
    average = config["algorithm_params"]["average"]

    target = config["base"]["target_col"]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(columns=target, axis=1)
    test_x = test.drop(columns=target, axis=1)

    logr = LogisticRegression(
        solver=solver,
        random_state=random_state
    )
    logr.fit(train_x, train_y)

    predicted_qualities = logr.predict(test_x)
    (score_accuracy, score_precision, score_recall,
     score_f1) = eval_metrics(test_y, predicted_qualities, average)

    print("LogisticRegression model (solver=%s):" % (solver))
    print("  Accuracy Score: %s" % score_accuracy)
    print("  Precision Score: %s" % score_precision)
    print("  Recall Score: %s" % score_recall)
    print("  F1 Score: %s" % score_f1)

    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]

    with open(scores_file, "w") as f:
        scores = {
            "accuracy_score": '{:.2f}'.format(score_accuracy),
            "precision_score": '{:.2f}'.format(score_precision),
            "recall_score":  '{:.2f}'.format(score_recall),
            "f1_score":  '{:.2f}'.format(score_f1)
        }
        json.dump(scores, f, indent=4)

    with open(params_file, "w") as f:
        params = {
            "solver": solver
        }
        json.dump(params, f, indent=4)

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")

    joblib.dump(logr, model_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)
