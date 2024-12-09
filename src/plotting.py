from logging import Logger
from pathlib import Path
from typing import Dict, Tuple, Callable

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.offsetbox as moffsetbox
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from plottable import ColumnDefinition, Table

from src.config import Config, get_config, get_logger

config: Config = get_config()
logger: Logger = get_logger(config, __name__)

FIGURE_WIDTH_S = 8.3 / 2.56
FIGURE_WIDTH_M = 12 / 2.56
FIGURE_WIDTH_L = 17.5 / 2.56


def format_float_total_digits(value, n=4):
    """
    Ensures max three (or a given amount of) digits is displayed for floats.
    """
    integer_part = str(int(value))
    integer_length = len(integer_part)

    if (int(value) != 0) and (value < 0):
        integer_length -= 1
    decimal_digits = n - integer_length
    decimal_digits = max(decimal_digits, 1)
    formatted_value = f"{value:.{decimal_digits}f}"

    return formatted_value


class SensitivityGridPlot:
    """
    Plots sensitivity test results in a gridded plot.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_vars: Dict,
        figsize: Tuple | str = "auto",
        range: pd.Series | None = None,
    ) -> None:
        self.data: pd.DataFrame = data
        self.figsize: Tuple | str = figsize
        self.targets: Dict = target_vars
        self.fig: Figure | None = None
        self.table: Table | None = None
        self.range: pd.Series | None = range

    def _get_column_definitions(self, targets: Dict):
        col_defs = [
            ColumnDefinition(
                name="Experiment",
                textprops={"ha": "left"},
                width=4.2,
            ),
        ] + [
            ColumnDefinition(
                name=name,
                title=rf"${col["symbol"]}$",
                formatter=format_float_total_digits,
                # formatter="{:.2f}",
                # formatter=lambda x: "{:.1e}".format(x).replace("e-0", "e-").replace("e+0", "e+"),
                textprops={
                    "ha": "center",
                    "family": config.plotting.font_family,
                    "fontsize": config.plotting.font_size,
                },
                border="right",
                group=col["group"],
                cmap=mpl.cm.ScalarMappable(
                    norm=self.norm[name], cmap=mpl.cm.get_cmap("RdBu_r")
                ).to_rgba,
            )
            for name, col in targets.items()
        ]

        return col_defs

    def plot(self) -> None:
        if self.range is None:
            self.vmax = self.data.abs().max(axis=0)
            self.vmin = -self.vmax
        else:
            self.vmax = self.range
            self.vmin = -self.range
        self.norm = self.range.apply(
            lambda x: mpl.colors.CenteredNorm(vcenter=0.0, halfrange=max(x, 1e-2))
        )
        plt.rcParams["font.family"] = config.plotting.font_family
        plt.rcParams["font.serif"] = config.plotting.font_serif
        plt.rcParams["font.size"] = config.plotting.font_size
        # plt.rcParams["mathtext.default"] = "regular"

        if self.figsize == "auto":
            height = (2 + (len(self.data.index)) * 0.6) / 2.56
            width = FIGURE_WIDTH_L
            figsize = (width, height)
        else:
            figsize = self.figsize

        self.fig, self.ax = plt.subplots(
            1, figsize=figsize, layout="constrained", facecolor="white", dpi=120
        )

        col_defs = self._get_column_definitions(self.targets)

        cb = self.fig.colorbar(
            mpl.cm.ScalarMappable(
                norm=mpl.colors.CenteredNorm(vcenter=0.0, halfrange=1.0), cmap="RdBu_r"
            ),
            location="top",
            ax=self.ax,
            orientation="horizontal",
            pad=0.005,
            fraction=0.08,
            aspect=20,
            shrink=0.3,
            ticks=[-1, 1],
            anchor=(0.9, 1.0),
        )

        cb.ax.set_xticklabels(["Negative response", "Positive response"])

        # cb.ax.annotate(
        #    "$RF$ (-)", xy=(-0.18, 0.2), xycoords="axes fraction"
        # )

        self.table = Table(
            self.data,
            column_definitions=col_defs,
            row_dividers=True,
            footer_divider=True,
            ax=self.ax,
            row_divider_kw={"linewidth": 1.5, "color": "white"},
            # col_label_divider_kw={"linewidth": 1, "linestyle": "-", "color":"black"},
            column_border_kw={
                "linewidth": 1.5,
                "linestyle": "-",
                "color": "white",
                "zorder": 1,
            },
            col_label_cell_kw={"height": 1.2},
            footer_divider_kw={"linewidth": 0.5},
        ).autoset_fontcolors()


class SensitivityTestSummaryFigure:
    """
    A multi-panel figure to plot all important values from a sensitivity test case.
    """

    def __init__(
        self,
        experiment: Dict,
        baseline: Dict,
        spatial_filtering: Callable,
        figsize: Tuple | str = "auto",
    ) -> None:
        self.experiment: Dict = experiment
        self.baseline: Dict = baseline
        self.spatial_filtering = spatial_filtering
        self.figsize: Tuple | str = figsize
        self.fig: Figure | None = None

    def plot(self) -> None:
        plt.rcParams["font.family"] = config.plotting.font_family
        plt.rcParams["font.serif"] = config.plotting.font_serif
        plt.rcParams["font.size"] = config.plotting.font_size

        if self.figsize == "auto":
            width = FIGURE_WIDTH_L
            figsize = (width, 0.8 * width)
        else:
            figsize = self.figsize

        self.fig, self.ax = plt.subplots(
            3,
            2,
            sharex=True,
            figsize=figsize,
            layout="constrained",
            facecolor="white",
            dpi=120,
        )

        baseline = self.baseline["av_xy"]

        modifications = {}

        if not all(
            item in self.experiment.keys()
            for item in ["job_name_positive", "job_name_negative"]
        ):
            modifications["modification"] = self.experiment["outputs"]["negative"][
                "av_xy"
            ]
        else:
            modifications["+modification"] = self.experiment["outputs"]["positive"][
                "av_xy"
            ]
            modifications["-modification"] = self.experiment["outputs"]["negative"][
                "av_xy"
            ]

        colors = ("tab:red", "tab:blue")
        xlim = np.datetime64("2018-03-30T04:00"), np.datetime64("2018-03-31T04:00")
        xticks = pd.date_range(
            start="2018-03-30T06:00", end="2018-03-31T04:00", freq="6h"
        )

        # ax[0,0] shf
        baseline["shf*_xy"].where(self.spatial_filtering).mean(
            dim=("x", "y")
        ).squeeze().plot.line(ax=self.ax[0, 0], c="tab:grey", label="baseline")
        for i, (name, data) in enumerate(modifications.items()):
            data["shf*_xy"].where(self.spatial_filtering).mean(
                dim=("x", "y")
            ).squeeze().plot.line(ax=self.ax[0, 0], c=colors[i], label=name)
        self.ax[0, 0].set_ylabel(r"$H$ ($\mathrm{W}~\mathrm{m}^-2$)")

        # ax[0,1] qsws
        baseline["qsws*_xy"].where(self.spatial_filtering).mean(
            dim=("x", "y")
        ).squeeze().plot.line(ax=self.ax[0, 1], c="tab:grey", label="baseline")
        for i, (name, data) in enumerate(modifications.items()):
            data["qsws*_xy"].where(self.spatial_filtering).mean(
                dim=("x", "y")
            ).squeeze().plot.line(ax=self.ax[0, 1], c=colors[i], label=name)
        self.ax[0, 1].set_ylabel(r"$LE$ ($\mathrm{W}~\mathrm{m}^-2$)")

        flux_ylim = (
            min(self.ax[0, 0].get_ylim()[0], self.ax[0, 1].get_ylim()[0]),
            max(self.ax[0, 0].get_ylim()[1], self.ax[0, 1].get_ylim()[1]),
        )
        self.ax[0, 0].set_ylim(flux_ylim)
        self.ax[0, 1].set_ylim(flux_ylim)

        # ax[1,1] T_2m
        (baseline["ta_2m_agg*_xy"] - 273.15).where(self.spatial_filtering).mean(
            dim=("x", "y")
        ).squeeze().plot.line(ax=self.ax[1, 0], c="tab:grey", label="baseline")
        for i, (name, data) in enumerate(modifications.items()):
            (data["ta_2m_agg*_xy"] - 273.15).where(self.spatial_filtering).mean(
                dim=("x", "y")
            ).squeeze().plot.line(ax=self.ax[1, 0], c=colors[i], label=name)
        self.ax[1, 0].set_ylabel(r"$T_{\mathrm{2m}}$ ($^\circ\mathrm{C}$)")

        # ax[0,1] T_C
        (baseline["slurb_t_c_urb*_xy"] - 273.15).where(self.spatial_filtering).mean(
            dim=("x", "y")
        ).squeeze().plot.line(ax=self.ax[1, 1], c="tab:grey", label="baseline")
        for i, (name, data) in enumerate(modifications.items()):
            (data["slurb_t_c_urb*_xy"] - 273.15).where(self.spatial_filtering).mean(
                dim=("x", "y")
            ).squeeze().plot.line(ax=self.ax[1, 1], c=colors[i], label=name)
        self.ax[1, 1].set_ylabel(r"$T_C$ ($^\circ\mathrm{C}$)")

        temp_ylim = (
            min(self.ax[1, 0].get_ylim()[0], self.ax[1, 1].get_ylim()[0]),
            max(self.ax[1, 0].get_ylim()[1], self.ax[1, 1].get_ylim()[1]),
        )
        self.ax[1, 0].set_ylim(temp_ylim)
        self.ax[1, 1].set_ylim(temp_ylim)

        # ax[1,1] T_2m
        baseline["slurb_rh_canyon*_xy"].where(self.spatial_filtering).mean(
            dim=("x", "y")
        ).squeeze().plot.line(ax=self.ax[2, 0], c="tab:grey", label="baseline")
        for i, (name, data) in enumerate(modifications.items()):
            data["slurb_rh_canyon*_xy"].where(self.spatial_filtering).mean(
                dim=("x", "y")
            ).squeeze().plot.line(ax=self.ax[2, 0], c=colors[i], label=name)
        self.ax[2, 0].set_ylabel(r"$RH_{\mathrm{can}}$ (%)")

        # ax[0,1] T_C
        baseline["us*_xy"].where(self.spatial_filtering).mean(
            dim=("x", "y")
        ).squeeze().plot.line(ax=self.ax[2, 1], c="tab:grey", label="baseline")
        for i, (name, data) in enumerate(modifications.items()):
            data["us*_xy"].where(self.spatial_filtering).mean(
                dim=("x", "y")
            ).squeeze().plot.line(ax=self.ax[2, 1], c=colors[i], label=name)
        self.ax[2, 1].set_ylabel(r"$u_*$ ($\mathrm{m}~\mathrm{s}^{-1}$)")

        # Reset axes
        for ax in self.ax.flatten():
            ax.set_title("")
            ax.grid()
            ax.set_xlabel("")

        for ax in self.ax[2, :]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
            ax.set_xlim(xlim)
            ax.set_xlabel(r"Time (local/solar)")
            ax.set_xlim(
                (
                    pd.Timestamp("2018-03-30 04:00:00"),
                    pd.Timestamp("2018-03-31 04:00:00"),
                )
            )
        # Add subplot identifiers

        self.ax[0, 0].add_artist(
            moffsetbox.AnchoredText("(a)", loc="upper left", pad=0.2, borderpad=0)
        )
        self.ax[0, 1].add_artist(
            moffsetbox.AnchoredText("(b)", loc="upper left", pad=0.2, borderpad=0)
        )
        self.ax[1, 0].add_artist(
            moffsetbox.AnchoredText("(c)", loc="upper left", pad=0.2, borderpad=0)
        )
        self.ax[1, 1].add_artist(
            moffsetbox.AnchoredText("(d)", loc="upper left", pad=0.2, borderpad=0)
        )
        self.ax[2, 0].add_artist(
            moffsetbox.AnchoredText("(e)", loc="upper left", pad=0.2, borderpad=0)
        )
        self.ax[2, 1].add_artist(
            moffsetbox.AnchoredText("(f)", loc="upper left", pad=0.2, borderpad=0)
        )

        self.fig.suptitle(self.experiment["long_name"])
