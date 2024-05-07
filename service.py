import bentoml
from bentoml.io import JSON, NumpyNdarray, Text
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
    del model
    # model_name = "xgr"
    
    # # model = mlflow.pyfunc.load_model(uri)
    # # prediction = model.predict(data)
    return prediction
    # return np.zeros(shape=2)

@svc.api(input=JSON(), output=NumpyNdarray()) 
def predict(data) :
    if 'input' not in data:
        raise ValueError("input is not provided")
    if 'model' in data:
        model_name = data['model']
    else:
        raise ValueError("model is not provided")
    uri = f"models:/{model_name}@production"
    if 'alias' in data:
        uri = f"models:/{model_name}@{data['alias']}"
    elif 'version' in data:
        uri = f"models:/{model_name}/{data['version']}"
    model = mlflow.pyfunc.load_model(uri)
    prediction = model.predict(data["input"])
    del model
    return prediction

# a trigger to start the service

@svc.api(input=JSON(), output=Text()) 
def predict_checkdata(data) :
    if 'input' not in data:
        raise ValueError("input is not provided")
    if 'model' in data:
        model_name = data['model']
    else:
        raise ValueError("model is not provided")
    uri = f"models:/{model_name}@production"
    if 'alias' in data:
        uri = f"models:/{model_name}@{data['alias']}"
    elif 'version' in data:
        uri = f"models:/{model_name}/{data['version']}"
    model = mlflow.pyfunc.load_model(uri)
    model_name = str(model)
    del model
    return model_name