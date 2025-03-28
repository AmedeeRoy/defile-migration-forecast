import numpy as np
from matplotlib import pyplot as plt


def plt_predict(data, species=None, filepath=None):
    pred_count = np.expm1(data.pred_log_hourly_count)

    fig, ax = plt.subplots(2, 3, figsize=(10, 5), tight_layout=True, sharex=True, sharey=True)
    ax = ax.flatten()
    for k in range(len(pred_count)):
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

    ax[0].set_ylabel("Forecasted individual \ncounts (#)")
    ax[3].set_ylabel("Forecasted individual \ncounts (#)")
    plt.suptitle(f"Defile Bird Forecasts - {species}")

    if filepath is not None:
        plt.savefig(filepath)
        plt.close()
    else:
        plt.show()
