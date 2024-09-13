import numpy as np
import xarray as xr


class Transformer:
    def __init__(self, type: str, data: xr.Dataset = None, params=None):

        possible_type = ["standardization", "log_transform", "minmax_normalization"]
        if type not in possible_type:
            raise ValueError(f"type needs to be either {possible_type}")
        self.type = type

        # Validation to ensure only one of dataset or transformer is provided
        if data is not None and params is not None:
            raise ValueError(
                "Both data and params cannot be provided at the same time."
            )

        self.params = {}

        if data is not None:
            self.compute_param(data)

        if params is not None:
            self.params = params

    def __str__(self):
        return f"Transformer(type:{self.type}, params:{self.params})"

    def __repr__(self):
        return f"Transformer(type:{self.type}, params:{self.params})"

    def compute_param(self, data):
        if self.type == "standardization":
            mean = data.mean().item()
            std = data.std().item()
            self.params = mean, std

        elif self.type == "log_transform":
            clipped_data = data.clip(min=1e-9)  # Ensure no non-positive values
            log_data = np.log(clipped_data)
            mean = log_data.mean().item()
            std = log_data.std().item()
            self.params = mean, std

        elif self.type == "minmax_normalization":
            data_min = data.min().item()
            data_max = data.max().item()
            self.params = data_min, data_max

    def apply(self, data):
        if self.type == "standardization":
            mean, std = self.params
            return (np.log(data.clip(min=1e-9)) - mean) / std

        elif self.type == "log_transform":
            data_min, data_max = self.params
            return (data - data_min) / (data_max - data_min)

        elif self.type == "minmax_normalization":
            data_min, data_max = self.params
            return (data - data_min) / (data_max - data_min)

    def export(self):
        return {"type": self.type, "params": self.params}


class DataTransformer:
    def __init__(self, dataset: xr.Dataset = None, transformers=None):
        # Validation to ensure only one of dataset or transformer is provided
        if dataset is not None and transformers is not None:
            raise ValueError(
                "Both dataset and transformer cannot be provided at the same time."
            )

        # Initialize the transformation dictionary
        self.transformation_dict = {
            "sun_azimuth": "minmax_normalization",
            "sun_altitude": "minmax_normalization",
            "temperature_2m": "standardization",
            "dewpoint_temperature_2m": "standardization",
            "total_precipitation": "log_transform",
            "surface_pressure": "standardization",
            "u_component_of_wind_10m": "minmax_normalization",
            "v_component_of_wind_10m": "minmax_normalization",
            "u_component_of_wind_100m": "minmax_normalization",
            "v_component_of_wind_100m": "minmax_normalization",
            "instantaneous_10m_wind_gust": "log_transform",
            "high_cloud_cover": "minmax_normalization",
            "low_cloud_cover": "minmax_normalization",
            "medium_cloud_cover": "minmax_normalization",
            "total_cloud_cover": "minmax_normalization",
            "surface_solar_radiation_downwards": "minmax_normalization",
            "surface_sensible_heat_flux": "standardization",
            "surface_net_solar_radiation": "minmax_normalization",
        }

        self.transformers = {}

        # If provided, use the precomputed transformer; otherwise, initialize an empty dictionary
        if dataset is not None:
            self.compute_transformers(dataset)

        if transformers is not None:
            self.transformers = {
                key: Transformer(type=t["type"], params=t["params"])
                for key, t in transformers.items()
            }

    def __str__(self):
        return f"DataTransformer({self.transformers})"

    def __repr__(self):
        return f"DataTransformer({self.transformers})"

    # Function to compute transformation parameters
    def compute_transformers(self, dataset: xr.Dataset):
        for var in dataset.data_vars:
            if var in self.transformation_dict:
                self.transformers[var] = Transformer(
                    type=self.transformation_dict[var], data=dataset[var]
                )
            else:
                print(f"No transformation defined for {var}. Skipping...")

    # Function to apply transformations
    def apply_transformers(self, dataset: xr.Dataset):
        for var in dataset.data_vars:
            if var in self.transformers and var in self.transformation_dict:
                dataset[var] = self.transformers[var].apply(data=dataset[var])
            else:
                print(f"No transformation or parameters found for {var}. Skipping...")
        return dataset

    # Function to export the transformation parameters
    def export(self):
        return {key: t.export() for key, t in self.transformers.items()}
