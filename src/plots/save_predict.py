import numpy as np
from matplotlib import pyplot as plt


def plt_predict(data, species=None, filepath=None):
    pred_count = np.expm1(data.pred_log_hourly_count)
    n = len(pred_count)

    # Grid sized to the actual number of forecast days rather than a hard-coded 2x3, which
    # only matched forecast_day=5 and otherwise either raised an IndexError (more days) or
    # left blank panels with a broken ylabel placement (fewer days).
    ncols = min(3, n)
    nrows = -(-n // ncols)  # ceil division
    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=(10 / 3 * ncols, 5 / 2 * nrows),
        tight_layout=True,
        sharex=True,
        sharey=True,
    )
    ax = np.atleast_1d(ax).flatten()

    for k in range(n):
        subset = pred_count.isel(date=k)

        ax[k].bar(np.arange(24), subset.values)

        ax[k].set_title(subset.date.dt.strftime("%Y-%m-%d").item())
        ax[k].set_xticks(np.arange(0, 24, 3), [str(h) + "h" for h in np.arange(0, 24, 3)])

        ax[k].text(
            0.05,
            0.93,
            f"Total = {np.sum(subset.values):.0f}",
            transform=ax[k].transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="gray", alpha=0.25),
        )
        ax[k].set_xlim(6, 21)

    for row_start in range(0, n, ncols):
        ax[row_start].set_ylabel("Forecasted individual \ncounts (#)")

    for k in range(n, nrows * ncols):
        ax[k].set_visible(False)

    plt.suptitle(f"Defile Bird Forecasts - {species}")

    if filepath is not None:
        plt.savefig(filepath)
        plt.close()
    else:
        plt.show()
