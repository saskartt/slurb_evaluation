import logging
from pathlib import Path
from typing import Any, Optional, Tuple

import dask.config
import numpy as np
import yaml
from dask.distributed import Client, LocalCluster
from dask.system import CPU_COUNT

_logger: Optional[logging.Logger] = None


class Config(dict):
    def __init__(self, *args, **kwargs) -> None:
        super(Config, self).__init__(*args, **kwargs)

    def __getattr__(self, name) -> Any:
        value = self[name]
        if isinstance(value, dict):
            value = Config(value)
        return value


def get_config() -> Config:
    config_file = Path(Path.cwd() / "config.yml")
    with open(config_file, "r") as cfile:
        config = Config(yaml.safe_load(cfile))

    return config


def get_dask_cluster(config: Config) -> Tuple[Any, Client]:
    # First try to connect to a pre-existing scheduler
    dask.config.set({"array.chunk-size": config.dask.array.chunk_size})
    try:
        client = Client(
            f"{config.dask.cluster.scheduler.host}:{config.dask.cluster.scheduler.port}",
            timeout=1.0,
        )
        cluster = client.cluster
    except OSError:
        kwargs = {
            "n_workers": (
                CPU_COUNT
                if config.dask.cluster.n_workers == "auto"
                else config.dask.cluster.n_workers
            ),
            "memory_limit": config.dask.cluster.memory_limit,
            "host": config.dask.cluster.scheduler.host,
            "scheduler_port": config.dask.cluster.scheduler.port,
            "dashboard_address": config.dask.cluster.dashboard.address,
        }
        cluster = LocalCluster(**kwargs)
        client = Client(cluster)

    return cluster, client


def get_logger(config: Config, name: str) -> logging.Logger:
    global _logger
    if _logger is None:
        return set_logger(config, name)
    return _logger


def set_logger(config: Config, name: str):
    global _logger
    _logger = logging.getLogger(name)
    _logger.handlers.clear()
    _logger.setLevel(config.logging.level)
    ch = logging.StreamHandler()
    ch.setLevel(config.logging.level)
    formatter = logging.Formatter(config.logging.format)
    ch.setFormatter(formatter)
    _logger.addHandler(ch)
    _logger.propagate = False

    return _logger


def get_rng(config: Config):
    """_summary_"""
    return np.random.default_rng(config.random.seed)
