import pytest
import yaml
import os
import json


params_path = "params.yaml"
schema_path = os.path.join("tests", "schema_in.json")


@pytest.fixture
def config(config_path=params_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


@pytest.fixture
def schema_in(schema_path=schema_path):
    with open(schema_path) as json_file:
        schema = json.load(json_file)
    return schema
