import pytest
from email import message
from prediction_service.prediction import form_response, api_response
import prediction_service


input_data = {
    "correct_range": {
        "avg_rss12": 13,
        "var_rss12": 5,
        "avg_rss13": 6,
        "var_rss13": 7,
        "avg_rss23": 18,
        "var_rss23": 8
    },
    "incorrect_range": {
        "avg_rss12": 131,
        "var_rss12": 50,
        "avg_rss13": 6,
        "var_rss13": 7,
        "avg_rss23": 18,
        "var_rss23": 8
    },
}


def test_form_response_incorrect_range(data=input_data["incorrect_range"]):
    with pytest.raises(prediction_service.prediction.NotInRange):
        form_response(data)


def test_api_response_incorrect_range(data=input_data["incorrect_range"]):
    res = api_response(data)
    assert res["response"] == prediction_service.prediction.NotInRange(
    ).message
