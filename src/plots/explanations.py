import numpy as np
import torch
from matplotlib import pyplot as plt


def plt_explanations_metrics(data, explanations, filepath=None):
    yr, doy, era5_main, era5_hourly, era5_daily = explanations

    fig, ax = plt.subplots(3, 1, figsize=(6, 10), tight_layout=True, sharex=True)

    values = torch.mean(era5_main, dim=(0, 2, 3)).numpy()
    variables = list(data.era5_main.data_vars)
    ax[0].barh(variables, values)
    ax[0].set_title("Local hourly metrics")

    values = torch.mean(era5_hourly, dim=(0, 2, 3)).numpy()
    variables = list(data.era5_hourly.data_vars)
    ax[1].barh(variables, values)
    ax[1].set_title("Remote hourly metrics")

    values = torch.mean(era5_daily, dim=(0, 2, 3)).numpy()
    variables = list(data.era5_daily.data_vars)
    ax[2].barh(variables, values)
    ax[2].set_title("Remote daily metrics")

    if filepath is not None:
        plt.savefig(filepath)
        plt.close()
    else:
        plt.show()


def plt_explanations_locations(data, explanations, filepath=None):
    yr, doy, era5_main, era5_hourly, era5_daily = explanations

    fig, ax = plt.subplots(2, 1, figsize=(6, 6), tight_layout=True, sharex=True)

    values = torch.mean(era5_hourly, dim=(0, 1, 2)).numpy()
    variables = list(data.era5_hourly.location.values)
    ax[0].barh(variables, values)
    ax[0].set_title("Remote hourly locations")

    values = torch.mean(era5_daily, dim=(0, 1, 2)).numpy()
    variables = list(data.era5_daily.location.values)
    ax[1].barh(variables, values)
    ax[1].set_title("Remote daily locations")

    if filepath is not None:
        plt.savefig(filepath)
        plt.close()
    else:
        plt.show()
