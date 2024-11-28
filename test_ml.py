import pytest
# TODO: add necessary import
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data
import pandas as pd
from sklearn.model_selection import train_test_split



@pytest.fixture
def mock_data():
    data = {
        "workclass": ["Private", "Self-emp-not-inc", "Private", "State-gov"],
        "education": ["Bachelors", "HS-grad", "Masters", "Assoc-acdm"],
        "marital-status": ["Never-married", "Married-civ-spouse", "Divorced", "Married-civ-spouse"],
        "occupation": ["Tech-support", "Exec-managerial", "Sales", "Protective-serv"],
        "relationship": ["Not-in-family", "Husband", "Unmarried", "Husband"],
        "race": ["White", "Black", "Asian-Pac-Islander", "White"],
        "sex": ["Male", "Female", "Female", "Male"],
        "native-country": ["United-States", "United-States", "India", "United-States"],
        "age": [25, 38, 28, 40],
        "hours-per-week": [40, 50, 45, 60],
        "salary": ["<=50K", ">50K", "<=50K", ">50K"],
    }
    return pd.DataFrame(data)


# TODO: implement the first test. Change the function name and input as needed
def test_train_model(mock_data):
    """
    # Check if train_model uses the expected algorithm
    """

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X, y, encoder, lb = process_data(
        mock_data, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier), "The model is not a RandomForestClassifier"

# TODO: implement the second test. Change the function name and input as needed
def test_compute_model_metrics(mock_data):
    """
    # Verify compute_model_metrics returns expected value types
    """

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X, y, encoder, lb = process_data(
        mock_data, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X, y)
    preds = inference(model, X)
    precision, recall, f1 = compute_model_metrics(y, preds)
    assert isinstance(precision, float), "Precision is not a float"
    assert isinstance(recall, float), "Recall is not a float"
    assert isinstance(f1, float), "F1 score is not a float"


# TODO: implement the third test. Change the function name and input as needed
def test_train_test_split(mock_data):
    """
    # Check if train/test split has expected sizes
    """
    
    train, test = train_test_split(mock_data, test_size=0.2, random_state=42)
    assert len(train) == 3, "Training set size is incorrect"
    assert len(test) == 1, "Test set size is incorrect"
    assert isinstance(train, pd.DataFrame), "Training set is not a DataFrame"
    assert isinstance(test, pd.DataFrame), "Test set is not a DataFrame"
