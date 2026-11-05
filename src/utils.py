"""
src/utils.py
============
Utility helpers: logging setup, directory creation, timing, serialisation.
"""

import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger with the project-wide format.

    Parameters
    ----------
    name : str
        Logger name, typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(fmt=config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)
        )
        logger.addHandler(handler)
    logger.setLevel(config.LOG_LEVEL)
    return logger


def ensure_dirs(*paths: Path) -> None:
    """
    Create directories (including parents) if they do not already exist.

    Parameters
    ----------
    *paths : Path
        Any number of directory paths to create.
    """
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def save_object(obj: Any, path: Path) -> None:
    """
    Serialise *obj* to disk using pickle.

    Parameters
    ----------
    obj  : Any
        Python object to serialise.
    path : Path
        Destination file path.
    """
    logger = get_logger(__name__)
    ensure_dirs(path.parent)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved → %s", path)


def load_object(path: Path) -> Any:
    """
    Deserialise an object from a pickle file.

    Parameters
    ----------
    path : Path
        Source file path.

    Returns
    -------
    Any
        The deserialised object.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as fh:
        return pickle.load(fh)


class Timer:
    """
    Context-manager / manual stopwatch for measuring elapsed wall time.

    Example
    -------
    >>> with Timer("PCA fit") as t:
    ...     fit_pca()
    >>> print(t.elapsed)
    """

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.elapsed: float = 0.0
        self._logger = get_logger(__name__)

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed = time.perf_counter() - self._start
        self._logger.info("⏱  %s — %.3f s", self.label, self.elapsed)

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> float:
        self.elapsed = time.perf_counter() - self._start
        return self.elapsed


def set_global_seed(seed: int = config.NUMPY_SEED) -> None:
    """
    Fix NumPy random seed for reproducibility.

    Parameters
    ----------
    seed : int
        Seed value (default from config).
    """
    np.random.seed(seed)
