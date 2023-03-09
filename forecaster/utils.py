import numpy as np
from alpha_vantage.timeseries import TimeSeries


def download_data(config):
    ts = TimeSeries(key=config["alpha_vantage"]["key"])
    data, _ = ts.get_daily_adjusted(config["alpha_vantage"]["symbol"], outputsize=config["alpha_vantage"]["outputsize"])

    data_date = [date for date in data.keys()]
    data_date.reverse()

    data_close_price = [float(data[date][config["alpha_vantage"]["key_adjusted_close"]]) for date in data.keys()]
    data_close_price.reverse()
    data_close_price = np.array(data_close_price)

    n_data_points = len(data_date)
    display_date_range = "from " + data_date[0] + " to " + data_date[n_data_points - 1]

    return data_date, data_close_price, n_data_points, display_date_range


def prepare_data_x(x, window_size):
    n_row = x.shape[0] - window_size + 1
    output = np.lib.stride_tricks.as_strided(x, shape=(n_row, window_size), strides=(x.strides[0], x.strides[0]))
    return output[:-1], output[-1]


def prepare_data_y(x, window_size):
    output = x[window_size:]
    return output
