import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


@dataclass
class DefileExport:
    filepath: str
    plot: bool
    plotdir: str

    def save(self, test_dataset, test_pred):
        test_dataset.set_transform(False)
        predictions = []

        for i in range(len(test_dataset)):
            count, year, doy, era5_hourly, era5_daily, mask = test_dataset[i]
            pred = era5_hourly.copy()
            pred = pred.assign(estimated_hourly_counts=("time", test_pred["pred"][i, 0, :]))
            pred = pred.assign(masked_total_counts_=("", test_pred["obs"][i]))
            pred = pred.assign(mask=("time", test_pred["mask"][i]))
            predictions.append(pred)

        predictions = xr.concat(predictions, dim="date")
        # ugly way to deal with 'obs' dimensions
        predictions = predictions.assign(
            masked_total_counts=("date", predictions["masked_total_counts_"].values.squeeze())
        )
        predictions = predictions.drop_dims("")
        predictions = predictions.assign(
            estimated_masked_total_counts=(
                predictions["estimated_hourly_counts"] * predictions["mask"]
            ).sum(dim="time")
        )

        predictions["time"] = predictions.time.astype(str)
        predictions.to_netcdf(self.filepath)
        print(predictions)

        os.makedirs(self.plotdir)
        self.plt_counts_distribution(predictions)
        self.plt_true_vs_prediction(predictions)
        self.plt_timeseries(predictions)
        self.plt_doy_sum(predictions)

    def plt_counts_distribution(self, data):
        true_count = data.masked_total_counts.values
        pred_count = data.estimated_masked_total_counts.values

        plt.hist(true_count, label="True count", alpha=0.5)
        plt.hist(pred_count, label="Predicted count", alpha=0.5)
        plt.xlabel("Counts (log-scale)")
        plt.ylabel("Histogram")
        plt.legend()
        plt.savefig(f"{self.plotdir}/counts_distribution.jpg")
        plt.close()

    def plt_true_vs_prediction(self, data):
        true_count = data.masked_total_counts.values
        pred_count = data.estimated_masked_total_counts.values

        x = np.linspace(min(true_count), max(true_count), 100)
        coef = np.polyfit(true_count, pred_count, 5)
        y = np.polyval(coef, x)

        plt.scatter(true_count, pred_count, c="black", s=5, alpha=0.4)
        plt.plot(x, y, c="red")
        plt.plot(x, x, "--", c="black")
        plt.xlabel("True count (log-scale)")
        plt.ylabel("Predicted count (log-scale)")
        plt.savefig(f"{self.plotdir}/true_vs_prediction.jpg")
        plt.close()

    def plt_timeseries(self, data):
        fig, ax = plt.subplots(4, 4, figsize=(12, 8), tight_layout=True)
        ax = ax.flatten()
        for i, k in enumerate(np.random.randint(0, len(data.date), size=16)):
            subs = data.isel(date=k)

            ax[i].plot(np.arange(0, 24), subs.estimated_hourly_counts)
            for k, m in enumerate(subs.mask.values):
                if m == 1:
                    ax[i].add_patch(Rectangle((k, 0), m, 10, color="yellow"))
                    obs = subs.masked_total_counts.item() / subs.mask.sum().item()
                    ax[i].plot([k, k + 1], [obs, obs], c="tab:red")

            ax[i].set_ylim(0, max(subs.estimated_hourly_counts.max(), obs) + 0.1)
            ax[i].set_xlabel("hours")
            ax[i].set_ylabel("Bird counts (log10)")
        fig.savefig(f"{self.plotdir}/timeseries.jpg")
        plt.close()

    def plt_doy_sum(self, data):
        data = data.assign_coords(doy=data.date.dt.dayofyear)

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        data.groupby("doy").sum().masked_total_counts.plot(ax=ax, label="True")
        data.groupby("doy").sum().estimated_masked_total_counts.plot(ax=ax, label="Prediction")
        plt.legend()
        fig.savefig(f"{self.plotdir}/timeseries.jpg")
        plt.close()
