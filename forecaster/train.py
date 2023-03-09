import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.pyplot import figure
from torch.utils.data import DataLoader


def run_epoch(dataloader, is_training=False):
    epoch_loss = 0

    if is_training:
        model.train()
    else:
        model.eval()

    for _, (x, y) in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()

        batchsize = x.shape[0]

        x = x.to(device)
        y = y.to(device)
        out = model(x)
        loss = criterion(out.contiguous(), y.contiguous())

        if is_training:
            loss.backward()
            optimizer.step()

        epoch_loss += (loss.detach().item() / batchsize)

    lr = scheduler.get_last_lr()[0]
    return epoch_loss, lr


if __name__ == "__main__":
    import mlflow

    from forecaster.classes import Normalizer, LSTMModel, TimeSeriesDataset
    from forecaster.config import config
    from forecaster.utils import download_data, prepare_data_x, prepare_data_y

    with mlflow.start_run():
        # mlflow.log_params(conf.params)

        # Set torch flags to maintain reproducibility
        torch.manual_seed(123)
        torch.use_deterministic_algorithms(True)

        # CUDA for PyTorch
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.backends.cudnn.benchmark = True
            print("CUDA is available")

        else:
            device = torch.device("cpu")
            print("No GPU found.")

    data_date, data_close_price, n_data_points, display_date_range = download_data(config)

    # normalize
    scaler = Normalizer()
    normalized_data_close_price = scaler.fit_transform(data_close_price)

    data_x, data_x_unseen = prepare_data_x(normalized_data_close_price, window_size=config["data"]["window_size"])
    data_y = prepare_data_y(normalized_data_close_price, window_size=config["data"]["window_size"])

    # split dataset
    split_index = int(data_y.shape[0] * config["data"]["train_split_size"])
    data_x_train = data_x[:split_index]
    data_x_val = data_x[split_index:]
    data_y_train = data_y[:split_index]
    data_y_val = data_y[split_index:]

    dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
    dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

    print("Train data shape", dataset_train.x.shape, dataset_train.y.shape)
    print("Validation data shape", dataset_val.x.shape, dataset_val.y.shape)

    train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=True)

    model = LSTMModel(input_size=config["model"]["input_size"], hidden_layer_size=config["model"]["lstm_size"],
                      num_layers=config["model"]["num_lstm_layers"], output_size=1, dropout=config["model"]["dropout"])

    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.1)

    for epoch in range(config["training"]["num_epoch"]):
        loss_train, lr_train = run_epoch(train_dataloader, is_training=True)
        loss_val, lr_val = run_epoch(val_dataloader)
        scheduler.step()

        mlflow.log_metric("Training Loss", loss_train, step=epoch)
        mlflow.log_metric("Validation Loss", loss_val, step=epoch)

        print(f'Epoch[{epoch + 1}/{config["training"]["num_epoch"]}] | loss_t: {loss_train:.4f} | loss_v: {loss_val:.4f}')

    torch.save({'model_state_dict': model.state_dict()}, 'models/model_finished.pt')

    mlflow.pytorch.log_model(model, "models", registered_model_name="ibm")

    # here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date
    train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=False)
    val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=False)

    model.eval()

    # predict on the training data, to see how well the model managed to learn and memorize
    predicted_train = np.array([])

    for idx, (x, y) in enumerate(train_dataloader):
        x = x.to(device)
        out = model(x)
        out = out.cpu().detach().numpy()
        predicted_train = np.concatenate((predicted_train, out))

    # predict on the validation data, to see how the model does
    predicted_val = np.array([])

    for idx, (x, y) in enumerate(val_dataloader):
        x = x.to(device)
        out = model(x)
        out = out.cpu().detach().numpy()
        predicted_val = np.concatenate((predicted_val, out))

    # prepare data for plotting
    to_plot_data_y_train_pred = np.zeros(n_data_points)
    to_plot_data_y_val_pred = np.zeros(n_data_points)

    to_plot_data_y_train_pred[config["data"]["window_size"]:split_index + config["data"]["window_size"]] = scaler.inverse_transform(predicted_train)
    to_plot_data_y_val_pred[split_index + config["data"]["window_size"]:] = scaler.inverse_transform(predicted_val)

    to_plot_data_y_train_pred = np.where(to_plot_data_y_train_pred == 0, None, to_plot_data_y_train_pred)
    to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)

    # plots
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(data_date, data_close_price, label="Actual prices", color=config["plots"]["color_actual"])
    plt.plot(data_date, to_plot_data_y_train_pred, label="Predicted prices (train)", color=config["plots"]["color_pred_train"])
    plt.plot(data_date, to_plot_data_y_val_pred, label="Predicted prices (validation)", color=config["plots"]["color_pred_val"])
    plt.title("Compare predicted prices to actual prices")
    xticks = [data_date[i] if ((i % config["plots"]["xticks_interval"] == 0 and (n_data_points - i) > config["plots"]["xticks_interval"]) or i == n_data_points - 1) else None for i in range(n_data_points)]
    x = np.arange(0, len(xticks))
    plt.xticks(x, xticks, rotation='vertical')
    plt.grid(visible=None, which='major', axis='y', linestyle='--')
    plt.legend()
    plt.savefig("plot_1.png")

    # prepare data for plotting the zoomed in view of the predicted prices vs. actual prices
    to_plot_data_y_val_subset = scaler.inverse_transform(data_y_val)
    to_plot_predicted_val = scaler.inverse_transform(predicted_val)
    to_plot_data_date = data_date[split_index + config["data"]["window_size"]:]

    # plots
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(to_plot_data_date, to_plot_data_y_val_subset, label="Actual prices", color=config["plots"]["color_actual"])
    plt.plot(to_plot_data_date, to_plot_predicted_val, label="Predicted prices (validation)", color=config["plots"]["color_pred_val"])
    plt.title("Zoom in to examine predicted price on validation data portion")
    xticks = [to_plot_data_date[i] if ((i % int(config["plots"]["xticks_interval"] / 5) == 0 and (len(to_plot_data_date) - i) > config["plots"]["xticks_interval"] / 6) or i == len(to_plot_data_date) - 1) else None for i in range(len(to_plot_data_date))]
    xs = np.arange(0, len(xticks))
    plt.xticks(xs, xticks, rotation='vertical')
    plt.grid(visible=None, which='major', axis='y', linestyle='--')
    plt.legend()
    plt.savefig("plot_2.png")

    # Log plots to MlFlow
    mlflow.log_artifact("plot_1.png")
    mlflow.log_artifact("plot_2.png")
