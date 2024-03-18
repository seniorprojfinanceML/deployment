import bentoml
from bentoml.io import NumpyNdarray
import mlflow
import numpy as np
import os

svc = bentoml.Service("XGBoostService")
URL = os.environ.get('URL')
mlflow.set_tracking_uri(URL)

@svc.api(input=NumpyNdarray(), output=NumpyNdarray()) 
def predict(data: np.ndarray) :
    model_name = "xgr"
    uri = f"models:/{model_name}@production"
    model = mlflow.xgboost.load_model(uri)
    prediction = model.predict(data)
    # model = mlflow.pyfunc.load_model(uri)
    # prediction = model.predict(data)
    return prediction