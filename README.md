# For Sibylla

## Summary

This project accomplishes THREE primary things:
1. Train an LSTM model from scratch to predict stock prices
2. Setup an MlFlow server and backend for experiment tracking, artifact storage and model registry
3. Deploy a web server that allows us to make a prediction using one of the saved models



## Notes

- I decided to use a 'vanilla' Pytorch implementation just to demonstrate lower-level control of the model training flow, but am familiar with Pytorch Lightning as a higher level API
  
- I used docker for the MLFlow server but not for the training code just to demonstrate communication between services outside of docker
  
- I haven't included unittests at this stage just due to time, but the `tests` folder exists just to show potential project organisation
  
- I usually use extensive type-hints and docstrings in production code but have been less stringent for the purposes of this demonstration
- This took about a days work to complete, I would love to explore where this sort of project could be taken given more time and attention




# Getting Started

## Pre-requisites
### Pytorch
Use the top-level requirements.txt file to install dependencies for the model training code. The code will use a GPU if available, but will run on a CPU if not.

### MLFlow
The MLFlow server will run on any machine with [Docker-compose](https://docs.docker.com/compose/) installed. (See [here](https://docs.docker.com/v17.09/engine/installation/) for guidance on Docker installation)

### MinIO
MinIO is a S3 replica that can easily be spun up with Docker, and is being used as the storage backend for MLFlow in this project. 

**When docker has started all of the services you must login to the console and create a bucket called 'mlflow'. Credentials are stored in the provided `.env` file** 

### **It will not work without this!**

## Data

The model is trained using data sourced from the `alpha_vantage` API that provides ticker data for stocks

## MLFlow

To spin up the MLFlow server components, from the root of the project directory:

```bash
sudo docker-compose up -d
```

**This takes quite a while to install all of the required binaries**

At this point, please set the following two environment variables so your training code knows where to send the collected tracking metrics and artifacts and provides access to S3 clone (MinIO):

```bash
export MLFLOW_TRACKING_URI=http://localhost:5009
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minio
export AWS_SECRET_ACCESS_KEY=minio123
```

The MLFlow server can be found at `http://localhost:5009/` and the associated MinIO console, used for the artifact storage backend, can be found at `http://localhost:9001/`. All credentials are saved to version control in the `.env` file

All artifacts, including finished models, checkpoints and example images are all able to explored through the MLFlow UI. 

***The artifacts are also stored locally for the purposes of demonstration, incase you wanted to explore any of the files outside of the MLFlow UI***

Each run also stores the parameters used for training, as well as keeping track of metrics and provided graphical views over time.



## Train Model
To train the model, use the `train` script:

```bash
python3 forecaster/train.py
```

# Results
NOTE: I've used the best practices for reproducibility as suggested by the Pytorch documentation, but the results aren't expected to be great. This is just for demonstration

I ran the training script using the currently committed configuration for a total of 100 Epochs, this took <5 mins on a single CPU on my laptop.

## Using the API for prediction
To test the API, go to `http://localhost:8000/docs` to access the OpenAPI docs. In the body of the request just use the string `"ibm"`, which is the stock used for this demo, and the API should return the predicted price for the stock the next day, using the ML model we trained and registered within MlFlow.

Normally of course the API would take the data to run inference on inside the body of the request, but i just loaded some data in the backend for prediction just to make the demonstration simpler.
