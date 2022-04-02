import os
import yaml
import argparse
import pandas as pd


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def combine_data(config_path):
    config = read_params(config_path)
    listOfDirs = config["data_source"]["dirs_list"]
    df = pd.DataFrame()
    for dir in listOfDirs:
        os.chdir(os.path.join(config["data_source"]["s3_source"], dir))
        listOfFiles = os.listdir()
        for file in listOfFiles:
            df_new = pd.read_csv(file, header=4, on_bad_lines='skip')
            df_new['LABEL'] = dir
            df = pd.concat([df, df_new], ignore_index=True)
        os.chdir(os.path.join("..", ".."))
        df.drop('# Columns: time', axis=1, inplace=True)
    if not("combined_AReM_data.csv" in os.listdir(os.path.join(os.getcwd(), "data/combined/"))):
        df.to_csv(config["data_source"]["comb_source"])


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = combine_data(config_path=parsed_args.config)
