import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import torch


@dataclass
class DefileExport:
    filepath: str
    plot: bool
    plotdir: str

    def save(self, test_dataset, test_pred):
        test_dataset.set_transform(False)
        predictions = []
        test_pred_pred = torch.cat(test_pred["pred"], dim=0).squeeze()
        test_pred_obs = torch.cat(test_pred["obs"], dim=0)
        test_pred_mask = torch.cat(test_pred["mask"], dim=0)

        for i in range(len(test_dataset)):
            count, year, doy, era5_hourly, era5_daily, mask = test_dataset[i]
            pred = era5_hourly.copy()
            pred = pred.assign(count_pred=("time", test_pred_pred[i, :]))
            pred = pred.assign(count=("time", test_pred_obs[i].repeat(24)))
            pred = pred.assign(mask=("time", test_pred_mask[i]))
            predictions.append(pred)

        predictions = xr.concat(predictions, dim="date")
        predictions["time"] = predictions.time.astype(str)
        predictions.to_netcdf(self.filepath)
        print(predictions)

        # if self.plot:
        #     os.makedirs(self.plotdir)
        #     self.plt_timeseries(predictions)
        #     self.plt_scatter(predictions)

    # def plt_timeseries(self, predictions):
    #     mask = mask.detach().cpu().numpy()
    #     count = count.detach().cpu().numpy()
    #     y_pred = y_pred.detach().cpu().numpy()

    #     fig, ax = plt.subplots(2,3, figsize = (10, 4), tight_layout=True)
    #     ax = ax.flatten()
    #     for i,k in enumerate(np.random.randint(0, y_pred.shape[0], size=6)):
    #         ax[i].plot(np.arange(0, 24), y_pred[k,0,:])
    #         ax[i].plot(np.arange(0, 24), mask[k,:])
    #         ax[i].plot(np.arange(0, 24), count[k].repeat(24))
    #         ax[i].set_xlabel("hours")
    #         ax[i].set_ylabel("Bird counts (log10)")
    #     plt.savefig(path)

    # def plt_scatter(self, predictions):
    #     y_pred = y_pred.detach().cpu().numpy()
    #     y = y.detach().cpu().numpy()

    #     fig, ax = plt.subplots(1,1, figsize = (4, 4), tight_layout=True)
    #     plt.scatter(y.squeeze(), y_pred.squeeze())
    #     plt.xlabel("Observed counts")
    #     plt.ylabel("Predicted masked counts")
    #     plt.savefig(path)
