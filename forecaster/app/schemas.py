from pydantic import BaseModel


class InputData(BaseModel):
    input_data: dict
