import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from get_data import read_params
from urllib.parse import urlparse
import argparse
import mlflow


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

    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)

    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
        logr = LogisticRegression(
            solver=solver,
            random_state=random_state
        )
        logr.fit(train_x, train_y)

        predicted_qualities = logr.predict(test_x)
        (score_accuracy, score_precision, score_recall,
         score_f1) = eval_metrics(test_y, predicted_qualities, average)

        mlflow.log_param("solver", solver)
        mlflow.log_param("average", average)

        mlflow.log_metric("accuracy", score_accuracy)
        mlflow.log_metric("precision", score_precision)
        mlflow.log_metric("recall", score_recall)
        mlflow.log_metric("f1", score_f1)

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                logr,
                "model",
                registered_model_name=mlflow_config["registered_model_name"])
        else:
            mlflow.sklearn.load_model(logr, "model")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)
