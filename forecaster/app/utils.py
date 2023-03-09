import mlflow
import numpy as np
import torch

from forecaster.config import config
from forecaster.classes import Normalizer
from forecaster.utils import download_data, prepare_data_x


def get_model_prediction(model_uri):
    mlflow.set_tracking_uri("http://web:5000")

    _, data_close_price, _, _ = download_data(config)
    scaler = Normalizer()
    normalized_data_close_price = scaler.fit_transform(data_close_price)

    _, pred_data = prepare_data_x(normalized_data_close_price, window_size=config["data"]["window_size"])

    x = torch.tensor(pred_data).float().to(config["training"]["device"]).unsqueeze(0).unsqueeze(2)

    model = mlflow.pytorch.load_model(model_uri=model_uri)
    model.eval()
    prediction = model(x)
    prediction = prediction.cpu().detach().numpy()

    p_range = 10
    to_plot_data_y_test_pred = np.zeros(p_range)

    to_plot_data_y_test_pred[p_range - 1] = scaler.inverse_transform(prediction)
    to_plot_data_y_test_pred = np.where(to_plot_data_y_test_pred == 0, None, to_plot_data_y_test_pred)

    return round(to_plot_data_y_test_pred[p_range - 1], 2)
