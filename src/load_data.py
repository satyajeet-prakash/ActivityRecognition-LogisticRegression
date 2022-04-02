import argparse
from get_data import read_params, get_data


def missing_values(df):
    try:
        columns = list(df.columns)
        for column in columns:
            if column != 'LABEL':
                df[column] = df[column].fillna(df[column].mean())
        return df
    except Exception as e:
        response = {"response": str(e)}
        return response


def zeros_values(df):
    try:
        columns = list(df.columns)
        for column in columns:
            if column != 'LABEL':
                df[column] = df[column].replace(0, df[column].mean())
        return df
    except Exception as e:
        response = {"response": str(e)}
        return response


def dataset_outliers(df):
    try:
        q = df['var_rss13'].quantile(.95)
        df_new = df[df['var_rss13'] < q]

        q = df_new['avg_rss12'].quantile(.02)
        df_new = df_new[df_new['avg_rss12'] > q]

        q = df_new['avg_rss13'].quantile(.03)
        df_new = df_new[df_new['avg_rss13'] > q]

        q = df_new['avg_rss13'].quantile(.99)
        df_new = df_new[df_new['avg_rss13'] < q]

        q = df_new['avg_rss23'].quantile(.02)
        df_new = df_new[df_new['avg_rss23'] > q]

        q = df_new['avg_rss23'].quantile(.95)
        df_new = df_new[df_new['avg_rss23'] < q]

        q = df_new['var_rss12'].quantile(.93)
        df_new = df_new[df_new['var_rss12'] < q]

        q = df_new['var_rss23'].quantile(.95)
        df_new = df_new[df_new['var_rss23'] < q]
        return df_new
    except Exception as e:
        response = {"response": str(e)}
        return response


def load_and_save(config_path):
    config = read_params(config_path)
    df = get_data(config_path)
    df = missing_values(df)
    df = zeros_values(df)
    df = dataset_outliers(df)
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    df.to_csv(raw_data_path, sep=",", index=False)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_and_save(config_path=parsed_args.config)
