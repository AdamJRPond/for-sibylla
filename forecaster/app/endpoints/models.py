from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

from forecaster.app import schemas, utils


router = APIRouter()

@router.post("/predict")
def predict(model_name: str = Body(..., description="Name of model")):
    try:
        model_result = utils.get_model_prediction(f"models:/{model_name}/latest")
        return {"nextDayPrice": model_result}

    except Exception as e:
        return JSONResponse(status_code=404, content={"error": e})
