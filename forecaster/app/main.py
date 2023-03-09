from fastapi import FastAPI, APIRouter

from forecaster.app.endpoints import models

__title__ = 'Forecaster API'
__version__ = '0.1'

tags_metadata = [
    {
        "name": "Models",
        "description": "Serve predictions for forecaster LSTM models",
    },
]

api_router = APIRouter()
api_router.include_router(models.router, prefix="/models", tags=["Models"])


app = FastAPI(
    version=__version__,
    title=__title__,
    description=("The best API in the world"),
    openapi_tags=tags_metadata,
)

app.include_router(api_router)
