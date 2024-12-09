from typing import Dict, Iterable, Callable
from logging import Logger
from src.config import Config, get_config, get_logger
import pandas as pd
from pathlib import Path
from functools import partial, reduce

import xarray as xr
import numpy as np

config: Config = get_config()
logger: Logger = get_logger(config, __name__)

g = 9.81
rho = 1.25
c_p = 1005.0
l_v = 2.4536e6


def time_to_datetimeindex(dataset: xr.Dataset, time_init: pd.Timestamp) -> xr.Dataset:
    time_seconds = dataset["time"].values
    datetime_index = time_init + pd.to_timedelta(time_seconds, unit="s")
    datetime_index = datetime_index.values.astype("datetime64[ns]")
    dataset = dataset.assign_coords(time=datetime_index)
    return dataset


def map_datasets_to_experiments(experiments: Dict, origin_date_time: pd.Timestamp):
    datasets = {"av_3d": [], "av_xy": [], "av_xz": [], "pr": [], "ts": [], "xy": []}
    for experiment_name, experiment in experiments.items():
        experiment["outputs"] = {}
        experiment["inputs"] = {}

        if not all(
            item in experiment.keys()
            for item in ["job_name_positive", "job_name_negative"]
        ):
            # This is honestly just horrible code. But I did name the jobs stupidly.
            output_path = list(
                Path(config.path.data.jobs).glob(
                    f"**/slurb_s_{experiment_name}*/OUTPUT"
                )
            )[0]
            for ds_name in datasets.keys():
                ds = xr.open_dataset(
                    list(
                        output_path.glob(
                            f"**/slurb_s_{experiment_name}*_{ds_name}.000.nc"
                        )
                    )[0],
                    chunks={"time": "auto"},
                )
                ds = time_to_datetimeindex(ds, origin_date_time)
                # Offset aggregation period labels to period center
                if len(ds_name.split("_")) > 1:
                    ds["time"] = ds["time"] - pd.Timedelta(15, "m")
                ds["second_of_day"] = (
                    ds.time.dt.hour * 3600 + ds.time.dt.minute * 60 + ds.time.dt.second
                )
                datasets[ds_name].append(ds)
                experiment["outputs"][ds_name] = ds

            # Include SLUrb driver as well. We need the urban fraction from there later, as some of the
            # variables are currently not available as aggregate outputs.
            slurb_driver_path = list(
                Path(config.path.data.jobs).glob(f"**/slurb_s_{experiment_name}*/INPUT")
            )[0]
            ds = xr.open_dataset(
                list(slurb_driver_path.glob("**/*_slurb"))[0],
            )
            experiment["inputs"]["slurb_driver"] = ds
            continue

        experiment["outputs"]["positive"] = {}
        experiment["outputs"]["negative"] = {}
        experiment["inputs"]["positive"] = {}
        experiment["inputs"]["negative"] = {}

        output_path = (
            Path(config.path.data.jobs) / experiment["job_name_positive"] / "OUTPUT"
        )
        for ds_name in datasets.keys():
            ds = xr.open_dataset(
                output_path / f"{experiment["job_name_positive"]}_{ds_name}.000.nc",
                chunks={"time": "auto"},
            )
            ds = time_to_datetimeindex(ds, origin_date_time)
            # Offset aggregation period labels to period center
            if len(ds_name.split("_")) > 1:
                ds["time"] = ds["time"] - pd.Timedelta(15, "m")
            ds["second_of_day"] = (
                ds.time.dt.hour * 3600 + ds.time.dt.minute * 60 + ds.time.dt.second
            )
            datasets[ds_name].append(ds)
            experiment["outputs"]["positive"][ds_name] = ds

        slurb_driver_path = (
            Path(config.path.data.jobs) / experiment["job_name_positive"] / "INPUT"
        )
        ds = xr.open_dataset(
            slurb_driver_path / f"{experiment["job_name_positive"]}_slurb",
        )

        experiment["inputs"]["positive"]["slurb_driver"] = ds

        output_path = (
            Path(config.path.data.jobs) / experiment["job_name_negative"] / "OUTPUT"
        )
        for ds_name in datasets.keys():
            ds = xr.open_dataset(
                output_path / f"{experiment["job_name_negative"]}_{ds_name}.000.nc",
                chunks={"time": "auto"},
            )
            ds = time_to_datetimeindex(ds, origin_date_time)
            # Offset aggregation period labels to period center
            if len(ds_name.split("_")) > 1:
                ds["time"] = ds["time"] - pd.Timedelta(15, "m")
            ds["second_of_day"] = (
                ds.time.dt.hour * 3600 + ds.time.dt.minute * 60 + ds.time.dt.second
            )
            experiment["outputs"]["negative"][ds_name] = ds

        slurb_driver_path = (
            Path(config.path.data.jobs) / experiment["job_name_negative"] / "INPUT"
        )
        ds = xr.open_dataset(
            slurb_driver_path / f"{experiment["job_name_negative"]}_slurb",
        )
        experiment["inputs"]["negative"]["slurb_driver"] = ds

    return datasets


def compute_experiment_modifications(experiments: Dict, baseline_values: Dict):
    layer_dimnames = {
        "road": "dz_road",
        "roof": "dz_roof",
        "wall": "dz_wall",
        "win": "dz_window",
    }
    for experiment_name, experiment in experiments.items():
        if "parameter" not in experiment.keys():
            if experiment["type"] == "inflow_driver":
                diff = 0.5 * baseline_values["wspeed"]["value"]
                experiments[experiment_name]["baseline_value"] = baseline_values[
                    "wspeed"
                ]["value"]
                experiments[experiment_name]["diff_abs"] = 2 * diff
                experiments[experiment_name]["diff_rel"] = (2 * diff) / experiments[
                    experiment_name
                ]["baseline_value"]
            continue

        for parameter_key, parameter in experiment["parameter"].items():
            if experiment["type"] == "factor":
                experiment_name_split = experiment_name.split("_")
                if experiment_name_split[0] == "c":
                    diff = parameter["values"][0] * np.sum(
                        np.array(baseline_values[parameter_key]["value"])
                        * np.array(
                            baseline_values[layer_dimnames[experiment_name_split[1]]][
                                "value"
                            ]
                        )
                    )
                    experiments[experiment_name]["baseline_value"] = np.sum(
                        np.array(baseline_values[parameter_key]["value"])
                        * baseline_values[layer_dimnames[experiment_name_split[1]]][
                            "value"
                        ]
                    )
                    experiments[experiment_name]["diff_abs"] = 2 * diff
                    experiments[experiment_name]["diff_rel"] = (2 * diff) / experiments[
                        experiment_name
                    ]["baseline_value"]
                elif experiment_name_split[0] == "lambda":
                    # Summing up conductances C = 1/(1/c1+1/c2+1/c3...)
                    experiments[experiment_name]["baseline_value"] = 1.0 / np.sum(
                        (1.0 / np.array(baseline_values[parameter_key]["value"]))
                        * (
                            1.0
                            / np.array(
                                baseline_values[
                                    layer_dimnames[experiment_name_split[1]]
                                ]["value"]
                            )
                        )
                    )
                    diff = (
                        parameter["values"][0]
                        * 1.0
                        / np.sum(
                            (1.0 / np.array(baseline_values[parameter_key]["value"]))
                            * (
                                1.0
                                / np.array(
                                    baseline_values[
                                        layer_dimnames[experiment_name_split[1]]
                                    ]["value"]
                                )
                            )
                        )
                    )
                    experiments[experiment_name]["diff_abs"] = 2 * diff
                    experiments[experiment_name]["diff_rel"] = (2 * diff) / experiments[
                        experiment_name
                    ]["baseline_value"]
                else:
                    experiments[experiment_name]["baseline_value"] = baseline_values[
                        parameter_key
                    ]["value"]
                    diff = (
                        parameter["values"][0] * baseline_values[parameter_key]["value"]
                    )
                    experiments[experiment_name]["diff_abs"] = 2 * diff
                    experiments[experiment_name]["diff_rel"] = (2 * diff) / experiments[
                        experiment_name
                    ]["baseline_value"]

            elif experiment["type"] == "complement_factor":
                experiments[experiment_name]["baseline_value"] = baseline_values[
                    parameter_key
                ]["value"]
                diff = parameter["values"][0] * (
                    1.0 - baseline_values[parameter_key]["value"]
                )
                experiments[experiment_name]["diff_abs"] = 2 * diff
                experiments[experiment_name]["diff_rel"] = (2 * diff) / experiments[
                    experiment_name
                ]["baseline_value"]

            elif experiment["type"] == "difference":
                experiments[experiment_name]["baseline_value"] = baseline_values[
                    parameter_key
                ]["value"]
                diff = parameter["values"][0]
                experiments[experiment_name]["diff_abs"] = 2 * diff
                experiments[experiment_name]["diff_rel"] = (2 * diff) / experiments[
                    experiment_name
                ]["baseline_value"]


def apply_buffer_zone_to_array(array: xr.DataArray, buffer: int):
    edge_mask = xr.full_like(array, dtype=bool, fill_value=True)
    edge_mask[:buffer, :] = False
    edge_mask[-buffer:, :] = False
    edge_mask[:, :buffer] = False
    edge_mask[:, -buffer:] = False
    array = array.where(edge_mask, other=np.nan)

    return array


def time_filter(
    data_object: xr.DataArray | xr.Dataset,
    start_time: np.datetime64,
    end_time: np.datetime64,
):
    return (start_time <= data_object.time) & (data_object.time <= end_time)


def coord_filter(coord_object: xr.DataArray, start: float, end: float):
    return (coord_object >= start) & (coord_object <= end)


def bbox_filter(obj: xr.Dataset | xr.DataArray, bbox: Iterable[float]):
    filters = []
    for x in ("x", "xu", "x_yz", "xu_yz", "xs"):
        filter_obj = partial(coord_filter, start=bbox[0], end=bbox[2])
        if hasattr(obj, "variables"):
            if x in obj.variables.keys():
                filters.append(filter_obj(obj[x]).compute())
        else:
            if x in obj.coords.keys():
                filters.append(filter_obj(obj[x]).compute())
    for y in ("y", "yv", "y_xz", "yv_xz", "ys"):
        filter_obj = partial(coord_filter, start=bbox[1], end=bbox[3])
        if hasattr(obj, "variables"):
            if y in obj.variables.keys():
                filters.append(filter_obj(obj[y]).compute())
        else:
            if y in obj.coords.keys():
                filters.append(filter_obj(obj[y]).compute())
    return reduce(lambda a, b: a & b, filters)


def lcz_filter(
    obj: xr.Dataset | xr.DataArray,
    target_area: Iterable[float],
    n_tiles: int,
    tile_size: Iterable[float],
    lcz: int,
):
    # Create list of bboxes corresponding to the LCZ:
    boxes = []
    for row in range(n_tiles):
        for col in range(n_tiles):
            x_min = col * tile_size[0]
            y_min = row * tile_size[1]
            x_max = x_min + tile_size[0]
            y_max = y_min + tile_size[1]
            boxes.append([x_min, y_min, x_max, y_max])
    boxes = np.array(boxes)

    # Bottom-left corner is LCZ 5.
    lcz_5 = np.array(
        [(row + col) % 2 == 0 for row in range(n_tiles) for col in range(n_tiles)]
    )
    lcz_2 = ~lcz_5
    boxes = boxes[lcz_2, :] if lcz == 2 else boxes[lcz_5, :]

    # Add offset
    boxes[:, 0] += target_area[0]
    boxes[:, 1] += target_area[1]
    boxes[:, 2] += target_area[0]
    boxes[:, 3] += target_area[1]

    # Apply reduction
    bbox_filters = [bbox_filter(obj, bbox=box) for box in boxes]
    return reduce(lambda a, b: a | b, bbox_filters)


def apply_bbox_filter_dataarray(obj: xr.DataArray, bbox: Iterable[float]):
    for x in ("x", "xu", "x_yz", "xu_yz"):
        if x in obj.coords.keys():
            obj = obj.where((obj[x] >= bbox[0]) & (obj[x] <= bbox[2]), drop=True)
    for y in (
        "y",
        "yv",
        "y_xz",
        "yv_xz",
    ):
        if y in obj.coords.keys():
            obj = obj.where((obj[y] >= bbox[1]) & (obj[y] <= bbox[3]), drop=True)
    return obj


def apply_height_mask(obj: xr.DataArray | xr.Dataset, z_bounds: Iterable[float]):
    for y in ("zu", "zw", "zu_xy", "zw_xy"):
        if y in obj.coords.keys():
            obj = obj.where((obj[y] >= z_bounds[0]) & (obj[y] < z_bounds[1]), drop=True)
    return obj


def aggregate_friction_velocities(
    experiments: Dict, baseline_outputs: Dict, baseline_slurb_driver: xr.Dataset
):
    baseline_outputs["av_xy"]["us*_xy"] = (
        baseline_slurb_driver["urban_fraction"]
        * baseline_outputs["av_xy"]["slurb_us_urb*_xy"]
        + (1.0 - baseline_slurb_driver["urban_fraction"])
        * baseline_outputs["av_xy"]["us*_xy"]
    )
    for experiment_name, experiment in experiments.items():
        if not all(
            item in experiment.keys()
            for item in ["job_name_positive", "job_name_negative"]
        ):
            experiment["outputs"]["av_xy"]["us*_xy"] = (
                experiment["inputs"]["slurb_driver"]["urban_fraction"]
                * experiment["outputs"]["av_xy"]["slurb_us_urb*_xy"]
                + (1.0 - experiment["inputs"]["slurb_driver"]["urban_fraction"])
                * experiment["outputs"]["av_xy"]["us*_xy"]
            )
            continue
        for job_name, modification in zip(
            [experiment["job_name_positive"], experiment["job_name_negative"]],
            ["positive", "negative"],
        ):
            experiment["outputs"][modification]["av_xy"]["us*_xy"] = (
                experiment["inputs"][modification]["slurb_driver"]["urban_fraction"]
                * experiment["outputs"][modification]["av_xy"]["slurb_us_urb*_xy"]
                + (
                    1.0
                    - experiment["inputs"][modification]["slurb_driver"][
                        "urban_fraction"
                    ]
                )
                * experiment["outputs"][modification]["av_xy"]["us*_xy"]
            )


def aggregate_ta_2m(
    experiments: Dict, baseline_outputs: Dict, baseline_slurb_driver: xr.Dataset
):
    baseline_outputs["av_xy"]["ta_2m_agg*_xy"] = baseline_slurb_driver[
        "urban_fraction"
    ] * baseline_outputs["av_xy"]["slurb_t_canyon*_xy"] + (
        1.0 - baseline_slurb_driver["urban_fraction"]
    ) * (baseline_outputs["av_xy"]["ta_2m*_xy"] + 273.15)
    for experiment_name, experiment in experiments.items():
        if not all(
            item in experiment.keys()
            for item in ["job_name_positive", "job_name_negative"]
        ):
            experiment["outputs"]["av_xy"]["ta_2m_agg*_xy"] = experiment["inputs"][
                "slurb_driver"
            ]["urban_fraction"] * experiment["outputs"]["av_xy"][
                "slurb_t_canyon*_xy"
            ] + (1.0 - experiment["inputs"]["slurb_driver"]["urban_fraction"]) * (
                experiment["outputs"]["av_xy"]["ta_2m*_xy"] + 273.15
            )
            continue
        for job_name, modification in zip(
            [experiment["job_name_positive"], experiment["job_name_negative"]],
            ["positive", "negative"],
        ):
            experiment["outputs"][modification]["av_xy"]["ta_2m_agg*_xy"] = experiment[
                "inputs"
            ][modification]["slurb_driver"]["urban_fraction"] * experiment["outputs"][
                modification
            ]["av_xy"]["slurb_t_canyon*_xy"] + (
                1.0
                - experiment["inputs"][modification]["slurb_driver"]["urban_fraction"]
            ) * (experiment["outputs"][modification]["av_xy"]["ta_2m*_xy"] + 273.15)


def compute_relative_humidities(experiments: Dict, baseline_outputs: Dict):
    e_s = 611.2 * np.exp(
        17.67
        * (baseline_outputs["av_xy"]["slurb_t_canyon*_xy"] - 273.15)
        / ((baseline_outputs["av_xy"]["slurb_t_canyon*_xy"] - 273.15) + 243.5)
    )
    baseline_outputs["av_xy"]["slurb_qs_canyon*_xy"] = 0.622 * e_s / (1e5 - e_s)

    baseline_outputs["av_xy"]["slurb_rh_canyon*_xy"] = (
        baseline_outputs["av_xy"]["slurb_q_canyon*_xy"]
        / baseline_outputs["av_xy"]["slurb_qs_canyon*_xy"]
        * 100
    )

    for experiment_name, experiment in experiments.items():
        if not all(
            item in experiment.keys()
            for item in ["job_name_positive", "job_name_negative"]
        ):
            e_s = 611.2 * np.exp(
                17.67
                * (experiment["outputs"]["av_xy"]["slurb_t_canyon*_xy"] - 273.15)
                / (
                    (experiment["outputs"]["av_xy"]["slurb_t_canyon*_xy"] - 273.15)
                    + 243.5
                )
            )
            experiment["outputs"]["av_xy"]["slurb_qs_canyon*_xy"] = (
                0.622 * e_s / (1e5 - e_s)
            )
            experiment["outputs"]["av_xy"]["slurb_rh_canyon*_xy"] = (
                experiment["outputs"]["av_xy"]["slurb_q_canyon*_xy"]
                / experiment["outputs"]["av_xy"]["slurb_qs_canyon*_xy"]
                * 100
            )
            continue
        for job_name, modification in zip(
            [experiment["job_name_positive"], experiment["job_name_negative"]],
            ["positive", "negative"],
        ):
            e_s = 611.2 * np.exp(
                17.67
                * (
                    experiment["outputs"][modification]["av_xy"]["slurb_t_canyon*_xy"]
                    - 273.15
                )
                / (
                    (
                        experiment["outputs"][modification]["av_xy"][
                            "slurb_t_canyon*_xy"
                        ]
                        - 273.15
                    )
                    + 243.5
                )
            )
            experiment["outputs"][modification]["av_xy"]["slurb_qs_canyon*_xy"] = (
                0.622 * e_s / (1e5 - e_s)
            )
            experiment["outputs"][modification]["av_xy"]["slurb_rh_canyon*_xy"] = (
                experiment["outputs"][modification]["av_xy"]["slurb_q_canyon*_xy"]
                / experiment["outputs"][modification]["av_xy"]["slurb_qs_canyon*_xy"]
                * 100
            )


def compute_shf_to_net_rad(experiments: Dict, baseline_outputs: Dict):
    baseline_outputs["av_xy"]["shf_to_net_rad"] = baseline_outputs["av_xy"][
        "shf*_xy"
    ] / (
        baseline_outputs["av_xy"]["rad_sw_in*_xy"]
        + baseline_outputs["av_xy"]["rad_lw_in*_xy"]
        - baseline_outputs["av_xy"]["rad_sw_out*_xy"]
        - baseline_outputs["av_xy"]["rad_lw_out*_xy"]
    )

    for experiment_name, experiment in experiments.items():
        if not all(
            item in experiment.keys()
            for item in ["job_name_positive", "job_name_negative"]
        ):
            experiment["outputs"]["av_xy"]["shf_to_net_rad"] = experiment["outputs"][
                "av_xy"
            ]["shf*_xy"] / (
                experiment["outputs"]["av_xy"]["rad_sw_in*_xy"]
                + experiment["outputs"]["av_xy"]["rad_lw_in*_xy"]
                - experiment["outputs"]["av_xy"]["rad_sw_out*_xy"]
                - experiment["outputs"]["av_xy"]["rad_lw_out*_xy"]
            )
            continue
        for job_name, modification in zip(
            [experiment["job_name_positive"], experiment["job_name_negative"]],
            ["positive", "negative"],
        ):
            experiment["outputs"][modification]["av_xy"]["shf_to_net_rad"] = experiment[
                "outputs"
            ][modification]["av_xy"]["shf*_xy"] / (
                experiment["outputs"][modification]["av_xy"]["rad_sw_in*_xy"]
                + experiment["outputs"][modification]["av_xy"]["rad_lw_in*_xy"]
                - experiment["outputs"][modification]["av_xy"]["rad_sw_out*_xy"]
                - experiment["outputs"][modification]["av_xy"]["rad_lw_out*_xy"]
            )


def compute_rf(experiment: Dict, varname: str, filtering: Iterable[Callable]):
    val = (
        (
            reduce(
                lambda arr, f: arr.where(f, drop=True),
                filtering,
                experiment["outputs"]["positive"]["av_xy"][varname],
            )
            .mean()
            .compute()
            .item()
            - reduce(
                lambda arr, f: arr.where(f, drop=True),
                filtering,
                experiment["outputs"]["negative"]["av_xy"][varname],
            )
            .mean()
            .compute()
            .item()
        )
        / experiment["diff_rel"]
        * 0.1
    )

    return val


def compute_response_factors(
    obj: pd.Series,
    experiments: Dict,
    spatial_filtering: Callable,
    daytime_filter: Callable,
    nighttime_filter: Callable,
):
    filtering = [daytime_filter, spatial_filtering]
    obj["shf_day"] = compute_rf(experiments[obj["ID"]], "shf*_xy", filtering)
    obj["qsws_day"] = compute_rf(experiments[obj["ID"]], "qsws*_xy", filtering)
    obj["ta_2m_day"] = compute_rf(experiments[obj["ID"]], "ta_2m_agg*_xy", filtering)
    obj["slurb_t_c_day"] = compute_rf(
        experiments[obj["ID"]], "slurb_t_c_urb*_xy", filtering
    )

    obj["slurb_rh_can_day"] = compute_rf(
        experiments[obj["ID"]], "slurb_rh_canyon*_xy", filtering
    )
    obj["us_day"] = compute_rf(experiments[obj["ID"]], "us*_xy", filtering)

    filtering = [nighttime_filter, spatial_filtering]

    obj["shf_night"] = compute_rf(experiments[obj["ID"]], "shf*_xy", filtering)
    obj["qsws_night"] = compute_rf(experiments[obj["ID"]], "qsws*_xy", filtering)
    obj["ta_2m_night"] = compute_rf(experiments[obj["ID"]], "ta_2m_agg*_xy", filtering)
    obj["slurb_t_c_night"] = compute_rf(
        experiments[obj["ID"]], "slurb_t_c_urb*_xy", filtering
    )
    obj["slurb_rh_can_night"] = compute_rf(
        experiments[obj["ID"]], "slurb_rh_canyon*_xy", filtering
    )
    obj["us_night"] = compute_rf(experiments[obj["ID"]], "us*_xy", filtering)

    return obj


def abs_diff_to_baseline(
    experiment: Dict,
    baseline_outputs: Dict,
    varname: str,
    filtering: Iterable,
):
    val = (
        reduce(
            lambda arr, f: arr.where(f, drop=True),
            filtering,
            experiment["outputs"]["av_xy"][varname],
        )
        .mean()
        .compute()
        .item()
        - reduce(
            lambda arr, f: arr.where(f, drop=True),
            filtering,
            baseline_outputs["av_xy"][varname],
        )
        .mean()
        .compute()
        .item()
    )

    return val


def compute_abs_diff_to_baseline(
    obj: pd.Series,
    experiments: Dict,
    baseline_outputs: Dict,
    spatial_filtering: Callable,
    daytime_filter: Callable,
    nighttime_filter: Callable,
):
    filtering = [daytime_filter, spatial_filtering]
    obj["shf_day"] = abs_diff_to_baseline(
        experiments[obj["ID"]], baseline_outputs, "shf*_xy", filtering
    )
    obj["qsws_day"] = abs_diff_to_baseline(
        experiments[obj["ID"]], baseline_outputs, "qsws*_xy", filtering
    )
    obj["ta_2m_day"] = abs_diff_to_baseline(
        experiments[obj["ID"]], baseline_outputs, "ta_2m_agg*_xy", filtering
    )
    obj["slurb_t_c_day"] = abs_diff_to_baseline(
        experiments[obj["ID"]], baseline_outputs, "slurb_t_c_urb*_xy", filtering
    )

    obj["slurb_rh_can_day"] = abs_diff_to_baseline(
        experiments[obj["ID"]], baseline_outputs, "slurb_rh_canyon*_xy", filtering
    )

    obj["us_day"] = abs_diff_to_baseline(
        experiments[obj["ID"]], baseline_outputs, "us*_xy", filtering
    )

    filtering = [nighttime_filter, spatial_filtering]

    obj["shf_night"] = abs_diff_to_baseline(
        experiments[obj["ID"]], baseline_outputs, "shf*_xy", filtering
    )
    obj["qsws_night"] = abs_diff_to_baseline(
        experiments[obj["ID"]], baseline_outputs, "qsws*_xy", filtering
    )
    obj["ta_2m_night"] = abs_diff_to_baseline(
        experiments[obj["ID"]], baseline_outputs, "ta_2m_agg*_xy", filtering
    )
    obj["slurb_t_c_night"] = abs_diff_to_baseline(
        experiments[obj["ID"]], baseline_outputs, "slurb_t_c_urb*_xy", filtering
    )
    obj["slurb_rh_can_night"] = abs_diff_to_baseline(
        experiments[obj["ID"]], baseline_outputs, "slurb_rh_canyon*_xy", filtering
    )
    obj["us_night"] = abs_diff_to_baseline(
        experiments[obj["ID"]], baseline_outputs, "us*_xy", filtering
    )

    return obj


def aggregate_slurb_lsm(
    var_slurb: xr.DataArray,
    var_lsm: xr.DataArray,
    lcz_map: xr.DataArray,
    f_urb: Dict[int, float],
):
    lcz_map_interp = lcz_map.interp_like(var_slurb, method="nearest")
    urban_fraction = xr.apply_ufunc(
        lambda x: f_urb.get(x, 0.0),
        lcz_map_interp,
        vectorize=True,
    )
    return urban_fraction * var_slurb + (1 - urban_fraction) * var_lsm


def calc_us_slurb(
    outputs: Dict,
    filters: Iterable,
):
    av_xy = outputs["av_xy"]

    us_ts = (
        reduce(
            lambda arr, f: arr.where(f, drop=True),
            filters,
            (
                ((av_xy["usws*_xy"] ** 2 + av_xy["vsws*_xy"] ** 2) ** (1.0 / 2.0) / rho)
                ** (1.0 / 2.0)
            ),
        )
        .mean(dim=("x", "y"))
        .squeeze()
    )

    return us_ts


def calc_shf_slurb(outputs: Dict, filters: Iterable):
    shf = (
        reduce(
            lambda arr, f: arr.where(f, drop=True), filters, outputs["av_xy"]["shf*_xy"]
        )
        .mean(dim=("x", "y"))
        .squeeze()
    )

    return shf


def calc_qsws_slurb(outputs: Dict, filters: Iterable):
    qsws = (
        reduce(
            lambda arr, f: arr.where(f, drop=True),
            filters,
            outputs["av_xy"]["qsws*_xy"],
        )
        .mean(dim=("x", "y"))
        .squeeze()
    )

    return qsws


def calc_t_can_slurb(outputs: Dict, filters: Iterable):
    t_can = (
        reduce(
            lambda arr, f: arr.where(f, drop=True),
            filters,
            outputs["av_xy"]["slurb_t_canyon*_xy"],
        )
        .mean(dim=("x", "y"))
        .squeeze()
    )

    return t_can


def calc_ta_2m_slurb(
    outputs: Dict,
    filters: Iterable,
    lcz_map: xr.DataArray,
    f_urb: Dict[int, float],
):
    ta_2m = aggregate_slurb_lsm(
        outputs["av_xy"]["slurb_t_canyon*_xy"],
        outputs["av_xy"]["ta_2m*_xy"] + 273.15,
        lcz_map,
        f_urb,
    )

    ta_2m = (
        reduce(
            lambda arr, f: arr.where(f, drop=True),
            filters,
            ta_2m,
        )
        .mean(dim=("x", "y"))
        .squeeze()
    )

    return ta_2m


def calc_rh_can_slurb(outputs: Dict, filters: Iterable):
    av_xy = outputs["av_xy"]
    t_can = calc_t_can_slurb(outputs, filters)

    e_s = 611.2 * np.exp(17.67 * (t_can - 273.15) / ((t_can - 273.15) + 243.5))
    qs_can = 0.622 * e_s / (1e5 - e_s)
    rh_can = (
        reduce(
            lambda arr, f: arr.where(f, drop=True), filters, av_xy["slurb_q_canyon*_xy"]
        )
        .mean(dim=("x", "y"))
        .squeeze()
        / qs_can
        * 100
    )

    return rh_can


def calc_shf_usm(outputs: Dict, filters: Iterable, h_canopy: float):
    xy = outputs["xy"]
    av_surf = outputs["av_surf"]
    # The wall fluxes are summed along vertical dimension, taking into account the resolution.
    # Note that the code won't produce right results with vertical grid stretching.
    dz = xy["zu_xy"].diff("zu_xy").item(1)
    dx = xy["x"].diff("x").item(1)
    dy = xy["y"].diff("y").item(1)

    theta = reduce(lambda arr, f: arr.where(f, drop=True), filters, xy["theta_xy"])
    # Restrict column heating change to roof top level
    theta = theta.where(xy["zu_xy"] < h_canopy, drop=True)

    av_surf = reduce(lambda arr, f: arr.where(f, drop=True), filters, av_surf)

    av_surf["shf"] = av_surf["shf"].where(av_surf["shf"] > -9999.0, other=np.nan)

    upward = av_surf["azimuth"] == -9999.0
    northward = av_surf["azimuth"] == 0.0
    eastward = av_surf["azimuth"] == 90.0
    southward = av_surf["azimuth"] == 180.0
    westward = av_surf["azimuth"] == 270.0

    # Total summed heat flux from surface facets
    shf = (
        (
            (
                av_surf["shf"].where(eastward).sum(dim="s") * dy
                + av_surf["shf"].where(northward).sum(dim="s") * dx
                + av_surf["shf"].where(southward).sum(dim="s") * dx
                + av_surf["shf"].where(westward).sum(dim="s") * dy
            )
            * dz
            * rho
            * c_p
        )
        + av_surf["shf"].where(upward).sum(dim="s") * dx * dy
    ).squeeze()

    # Scale to unit plan area
    shf = shf / (upward.isel(time=0).sum() * dx * dy)

    # Column heating tendency
    sensible_heat_content_tendency = (
        rho * c_p * theta.diff(dim="time", label="upper") / 1800.0
    ).mean(dim=("y", "x")).sum("zu_xy") * dz

    return shf - sensible_heat_content_tendency


def calc_qsws_usm(outputs: Dict, filters: Iterable, h_canopy: float):
    xy = outputs["xy"]
    av_surf = outputs["av_surf"]
    dz = xy["zu_xy"].diff("zu_xy").item(1)
    dx = xy["x"].diff("x").item(1)
    dy = xy["y"].diff("y").item(1)

    q = reduce(lambda arr, f: arr.where(f, drop=True), filters, xy["q_xy"])
    # Restrict column heating change to roof top level
    q = q.where(xy["zu_xy"] < h_canopy, drop=True)

    av_surf = reduce(lambda arr, f: arr.where(f, drop=True), filters, av_surf)

    av_surf["qsws"] = av_surf["qsws"].where(av_surf["qsws"] > -9999.0, other=np.nan)

    upward = av_surf["azimuth"] == -9999.0
    northward = av_surf["azimuth"] == 0.0
    eastward = av_surf["azimuth"] == 90.0
    southward = av_surf["azimuth"] == 180.0
    westward = av_surf["azimuth"] == 270.0

    # Total summed heat flux from surface facets
    qsws = (
        (
            (
                av_surf["qsws"].where(eastward).sum(dim="s") * dy
                + av_surf["qsws"].where(northward).sum(dim="s") * dx
                + av_surf["qsws"].where(southward).sum(dim="s") * dx
                + av_surf["qsws"].where(westward).sum(dim="s") * dy
            )
            * dz
            * rho
            * l_v
        )
        + av_surf["qsws"].where(upward).sum(dim="s") * dx * dy
    ).squeeze()

    # Scale to unit plan area
    qsws = qsws / (upward.isel(time=0).sum() * dx * dy)

    # Column heating tendency
    latent_heat_content_tendency = (
        rho * l_v * q.diff(dim="time", label="upper") / 1800.0
    ).mean(dim=("y", "x")).sum("zu_xy") * dz

    return qsws - latent_heat_content_tendency


def calc_us_usm(outputs: Dict, filters: Iterable):
    av_xy = outputs["av_xy"]
    av_surf = outputs["av_surf"]
    # The wall fluxes are summed along vertical dimension, taking into account the resolution.
    # Note that the code won't produce right results with vertical grid stretching.
    dz = av_xy["zu_xy"].diff("zu_xy").item(1)
    dx = av_xy["x"].diff("x").item(1)
    dy = av_xy["y"].diff("y").item(1)

    pres_drag_x = reduce(
        lambda arr, f: arr.where(f, drop=True), filters, av_xy["pres_drag_norm_x*_xy"]
    )
    pres_drag_y = reduce(
        lambda arr, f: arr.where(f, drop=True), filters, av_xy["pres_drag_norm_y*_xy"]
    )

    av_surf = reduce(lambda arr, f: arr.where(f, drop=True), filters, av_surf)

    av_surf["us"] = av_surf["us"].where(av_surf["us"] > -9999.0, other=np.nan)

    upward = av_surf["azimuth"] == -9999.0
    northward = av_surf["azimuth"] == 0.0
    eastward = av_surf["azimuth"] == 90.0
    southward = av_surf["azimuth"] == 180.0
    westward = av_surf["azimuth"] == 270.0

    # Total summed friction velocity from surface facets
    us_skin_friction = (
        (
            av_surf["us"].where(eastward).sum(dim="s") * dy
            + av_surf["us"].where(northward).sum(dim="s") * dx
            + av_surf["us"].where(southward).sum(dim="s") * dx
            + av_surf["us"].where(westward).sum(dim="s") * dy
        )
        * dz
        + av_surf["us"].where(upward).sum(dim="s") * dx * dy
    ).squeeze()

    us_skin_friction = us_skin_friction / (upward.isel(time=0).sum() * dx * dy)

    # Add contribution from pressure drag force (computed from absolute drag force vector magnitude
    # and finally scaled per unit area).
    us_pres_drag = (
        (
            (
                pres_drag_x.mean(dim=("x", "y")) ** 2
                + pres_drag_y.mean(dim=("x", "y")) ** 2
            )
            ** (1.0 / 2.0)
            / rho
        )
        ** (1.0 / 2.0)
        / (dx * dy)
    ).squeeze()

    # Total sinks of momentum
    us = us_pres_drag + us_skin_friction

    return us


def calc_t_can_usm(outputs: Dict, h_bld: float, filters: Iterable):
    t_can = reduce(
        lambda arr, f: arr.where(f, drop=True), filters, outputs["av_xy"]["theta_xy"]
    )

    t_can = (
        t_can.interp(zu_xy=h_bld / 2.0, method="nearest").mean(dim=("x", "y")).squeeze()
    )

    # Convert from potential temperature to temperature, insignificant but nevertheless.
    t_can = t_can - g / c_p * (h_bld / 2.0)

    return t_can


def calc_ta_2m_usm(outputs: Dict, filters: Iterable):
    t_can = reduce(
        lambda arr, f: arr.where(f, drop=True), filters, outputs["av_xy"]["theta_xy"]
    )

    t_can = t_can.interp(zu_xy=7.0, method="nearest").mean(dim=("x", "y")).squeeze()

    # Convert from potential temperature to temperature, insignificant but nevertheless.
    t_can = t_can - g / c_p * 2.0

    return t_can


def calc_rh_can_usm(outputs: Dict, h_bld: float, filters: Iterable):
    t_can = calc_t_can_usm(outputs, h_bld, filters)

    e_s = 611.2 * np.exp(17.67 * (t_can - 273.15) / ((t_can - 273.15) + 243.5))
    qs_can = 0.622 * e_s / (1e5 - e_s)
    q_can = (
        reduce(
            lambda arr, f: arr.where(f, drop=True), filters, outputs["av_xy"]["q_xy"]
        )
        .interp(zu_xy=h_bld / 2.0, method="nearest")
        .mean(dim=("x", "y"))
        .squeeze()
    )
    rh_can = q_can / qs_can * 100

    return rh_can


def calc_t1_usm(outputs: Dict, filters: Callable):
    av_surf = outputs["av_surf"]
    av_surf = reduce(lambda arr, f: arr.where(f, drop=True), filters, av_surf)

    av_surf["theta1"] = av_surf["theta1"].where(
        av_surf["theta1"] > -9999.0, other=np.nan
    )

    # Total summed friction velocity from surface facets
    t1 = av_surf["theta1"].mean(dim="s").squeeze()

    return t1


def map_func_to_resolutions(func: Callable, outputs: Dict, resolutions: Dict):
    results = []
    for res in resolutions.keys():
        results.append(func(outputs[res]))
    return results


def map_func_to_lcz_and_time_periods(
    func: Callable,
    lcz_filters: Iterable,
    time_filters: Iterable,
    outputs: Dict,
    resolutions: Dict,
    **kwargs,
):
    results = []
    for i, time_filter in enumerate(time_filters):
        results.append([])
        for j, lcz_filter in enumerate(lcz_filters):
            func = partial(func, filters=[lcz_filter, time_filter], **kwargs)
            results[i].append(map_func_to_resolutions(func, outputs, resolutions))

    return results
