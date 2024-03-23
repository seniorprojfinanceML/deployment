import bentoml
from bentoml.io import JSON, NumpyNdarray
import mlflow
import numpy as np
import os

svc = bentoml.Service("XGBoostService")
URL = os.environ.get('URL')
mlflow.set_tracking_uri(URL)

@svc.api(input=JSON(), output=NumpyNdarray()) 
def predict_xgboost(data) :
    if 'input' not in data:
        raise ValueError("input is not provided")
    model_name = 'xgr'
    if 'model' in data:
        model_name = data['model']
    uri = f"models:/{model_name}@production"
    if 'alias' in data:
        uri = f"models:/{model_name}@{data['alias']}"
    elif 'version' in data:
        uri = f"models:/{model_name}/{data['version']}"
    model = mlflow.xgboost.load_model(uri)
    prediction = model.predict(data["input"])
    # model_name = "xgr"
    
    # # model = mlflow.pyfunc.load_model(uri)
    # # prediction = model.predict(data)
    return prediction
    # return np.zeros(shape=2)
