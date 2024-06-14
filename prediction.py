import joblib
import pandas as pd

# Load the trained model and target encoder
model = joblib.load('model/Random_Forest_Model.joblib')
result_target = joblib.load('model/encoder_target.joblib')

def prediction(data):
    """Make prediction on preprocessed data.

    Args:
        data (Pandas DataFrame): Preprocessed data.

    Returns:
        str: Prediction result.
    """
    prediction = model.predict(data)
    return result_target.inverse_transform(prediction)[0]
