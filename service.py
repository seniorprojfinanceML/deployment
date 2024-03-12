import bentoml
from bentoml.io import NumpyNdarray
import mlflow
import numpy as np
from dotenv import dotenv_values
svc = bentoml.Service("XGBoostService")
URI = dotenv_values("mlflow.env")["URI"]
mlflow.set_tracking_uri(URI)

@svc.api(input=NumpyNdarray(), output=NumpyNdarray()) 
def predict(data: np.ndarray) :
    model_name = "xgr"
    uri = f"models:/{model_name}@production"
    model = mlflow.xgboost.load_model(uri)
    prediction = model.predict(data)
    # model = mlflow.pyfunc.load_model(uri)
    # prediction = model.predict(data)
    return prediction