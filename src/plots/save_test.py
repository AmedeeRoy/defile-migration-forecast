import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


def plt_counts_distribution(true_count, pred_count, filepath=None):
    bins = np.linspace(0, true_count.max(), 100)
    plt.hist(
        true_count,
        label="True count",
        alpha=0.5,
        edgecolor="k",
        bins=bins,
    )
    plt.hist(
        pred_count,
        label="Predicted count",
        alpha=0.5,
        edgecolor="k",
        bins=bins,
    )
    # plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Average hourly counts per periods")
    plt.ylabel("Histogram")
    plt.legend()

    if filepath is not None:
        plt.savefig(filepath)
        plt.close()
    else:
        plt.show()


def plt_true_vs_prediction(true_count, pred_count, log_transformed=True, filepath=None):
    if log_transformed:
        true_count = np.log1p(true_count)
        pred_count = np.log1p(pred_count)

    x = np.linspace(min(true_count), max(true_count), 100)
    coef = np.polyfit(true_count, pred_count, 1)
    y = np.polyval(coef, x)

    plt.scatter(true_count, pred_count, c="black", s=5, alpha=0.4)
    plt.plot(x, y, c="red")
    plt.plot(x, x, "--", c="black")
    plt.xlabel(f"True count {'(log transformed)' if log_transformed else ''}")
    plt.ylabel(f"Predicted count {'(log transformed)' if log_transformed else ''}")

    if filepath is not None:
        plt.savefig(filepath)
        plt.close()
    else:
        plt.show()


def plt_timeseries(data, log_transformed=True, global_y_lim=False, filepath=None):
    n_rows, n_cols = 4, 4

    daily_average = data.mean(dim="time")["obs_count"]
    valid_indices = np.where(daily_average > 0)[0]
    weights = data.pred_log_hourly_count[valid_indices].sum(dim="time").values
    sampled_indices = np.random.choice(
        valid_indices, size=n_rows * n_cols, p=weights / weights.sum()
    )
    # Sort sampled indices by date
    sampled_indices = sampled_indices[np.argsort(daily_average[sampled_indices].date)]

    all_obs = []
    all_pred = []
    all_mask = []
    all_pred_first = []

    for d in daily_average[sampled_indices].date:
        subs = data.sel(date=d)
        if log_transformed:
            obs = np.log1p(subs["obs_count"])
            pred = subs["pred_log_hourly_count"]
        else:
            obs = subs["obs_count"]
            pred = np.expm1(subs["pred_log_hourly_count"])

        # If there is a single observation on that day, the structure of pred is different (no date dimension) and the plot need to be done differently.
        if "date" in pred.dims:
            pred_first = pred.isel(date=0)  # If date is a dimension
            mask = subs.mask.values
            # mask = subs.mask.sum(dim="date").values  # summing over all observations.
            obs = obs.values
        else:
            pred_first = pred
            obs = [obs.values]
            mask = [subs.mask.values]

        all_obs.append(obs)
        all_pred.append(pred)
        all_pred_first.append(pred_first)
        all_mask.append(mask)

    # Compute maximum y value across all subplots
    ymax = (
        max(
            np.max([np.max(p.values) for p in all_pred]),
            np.max([np.max(o) for o in all_obs]),
        )
        + 0.1
    )

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2 * n_rows), tight_layout=True)
    ax = ax.flatten()

    for i, d in enumerate(daily_average[sampled_indices].date):
        # plot the prediction (only the first prediction is show - all should be the same for the day)
        ax[i].plot(np.arange(0, 24), all_pred_first[i])

        # find the max y value for drawing the rectangle
        if not global_y_lim:
            ymax = max(all_pred[i].max(), max(all_obs[i])) + 0.1

        # Plot the mask as yellow transparent background
        for k, m in enumerate(np.sum(all_mask[i], axis=0)):
            ax[i].add_patch(
                Rectangle((k, 0), 1, ymax, color=(1, 1, 0, min(1, m)))
            )  # RGBA: (1, 1, 0) is yellow, 'm' controls the alpha

        for u, o in enumerate(all_obs[i]):
            first_nonzero = np.argmax(all_mask[i][u] > 0)
            last_nonzero = len(all_mask[i][u]) - 1 - np.argmax(np.flip(all_mask[i][u]) > 0)
            ax[i].plot([first_nonzero, last_nonzero + 1], [o, o], c="tab:red")

        ax[i].set_xticks([0, 6, 12, 18, 24])

        ax[i].text(
            0.02,
            0.92,
            d.date.dt.strftime("%Y-%m-%d").item(),
            transform=ax[i].transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )

        if global_y_lim and i % n_cols != 0:  # Hide y-axis labels except for the first column
            ax[i].set_yticklabels([])  # Hide y-axis labels

        if i < (n_rows - 1) * n_cols:  # Hide x labels except for bottom row
            ax[i].set_xticklabels([])  # Hide x-axis labels

    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    if filepath is not None:
        plt.savefig(filepath)
        plt.close()
    else:
        plt.show()


def plt_doy_sum(data, filepath=None):
    # Convert data into a DataFrame with only the necessary columns
    data_df = data[["obs_count", "pred_count", "date"]].to_dataframe().reset_index()

    # Extract doy and year from date
    data_df["doy"] = data_df["date"].dt.dayofyear
    data_df["year"] = data_df["date"].dt.year

    # Get unique years and determine layout for subplots
    unique_years = np.unique(data_df["year"].values)
    n_years = len(unique_years)

    # Define the number of columns (2 columns if more than 4 years)
    n_cols = 2 if n_years > 4 else 1
    n_rows = (n_years + n_cols - 1) // n_cols  # Calculate number of rows

    # Set up subplots with dynamic row/column layout
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), sharex=True)

    # Flatten the axes array if necessary
    if n_years == 1:
        axes = [axes]  # In case there's only one year, make sure axes is iterable
    else:
        axes = axes.flatten()

    # Plot for each year
    for ax, y in zip(axes, unique_years):
        yearly_data = data_df[data_df["year"] == y]
        yearly_data.groupby("doy").mean().obs_count.plot(ax=ax, label="True")
        yearly_data.groupby("doy").mean().pred_count.plot(ax=ax, label="Prediction")
        ax.set_ylabel(f"Average hourly count ({y})")
        ax.legend()

    # Set x-label for the last row's axes
    for ax in axes[-n_cols:]:  # Only the last row should have x-label
        ax.set_xlabel("Day of Year")

    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)
        plt.close()
    else:
        plt.show()
