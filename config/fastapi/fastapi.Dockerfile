FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9-slim

WORKDIR /forecaster

COPY ./config/fastapi/requirements.txt ./
RUN pip install -r requirements.txt
RUN pip install boto3
RUN pip install alpha-vantage

COPY ./forecaster ./forecaster

CMD ["uvicorn", "forecaster.app.main:app", "--host=0.0.0.0", "--port=8000"]