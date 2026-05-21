"""Кэш Streamlit для дашборда (данные, цели модели, без повторной загрузки TFT)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from azs_tft.analytics import load_meta
from azs_tft.model_registry import resolve_active_paths
from azs_tft.preprocess import load_panel
from azs_tft.targets_config import default_model_targets


def _csv_cache_key(path: Path) -> tuple[str, int]:
    p = path.resolve()
    mtime = int(p.stat().st_mtime_ns) if p.exists() else 0
    return str(p), mtime


@st.cache_data(show_spinner=False)
def cached_load_panel(path_str: str, mtime_ns: int) -> pd.DataFrame:
    return load_panel(Path(path_str))


def get_cached_panel(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    key = _csv_cache_key(path)
    return cached_load_panel(key[0], key[1])


@st.cache_data(show_spinner=False)
def cached_targets_list(_run_key: str, params_mtime: int, meta_mtime: int) -> list[str]:
    """Список целей из meta / dataset_params — без load_model и predict."""
    meta_path = resolve_active_paths()["meta"]
    if meta_path.exists():
        meta = load_meta()
        t = meta.get("targets")
        if t:
            return list(t)
    try:
        import torch

        p = resolve_active_paths()["dataset_params"]
        try:
            params = torch.load(p, map_location="cpu", weights_only=False)
        except TypeError:
            params = torch.load(p, map_location="cpu")
    except Exception:
        return default_model_targets()
    t = params.get("target")
    if isinstance(t, str):
        return [t]
    if t:
        return list(t)
    return default_model_targets()


def active_targets() -> list[str]:
    paths = resolve_active_paths()
    run_key = paths.get("run_id") or "legacy"
    pm = int(paths["dataset_params"].stat().st_mtime_ns) if paths["dataset_params"].exists() else 0
    mm = int(paths["meta"].stat().st_mtime_ns) if paths["meta"].exists() else 0
    return cached_targets_list(str(run_key), pm, mm)


@st.cache_data(show_spinner=False)
def cached_station_map(path_str: str, mtime_ns: int, params_mtime: int, run_key: str) -> dict[int, int]:
    from azs_tft.predict import station_index_map

    return station_index_map(Path(path_str))


def get_cached_station_map(path: Path | None) -> dict[int, int]:
    if path is None or not path.exists():
        return {}
    paths = resolve_active_paths()
    run_key = str(paths.get("run_id") or "legacy")
    pm = int(paths["dataset_params"].stat().st_mtime_ns) if paths["dataset_params"].exists() else 0
    ck = _csv_cache_key(path)
    return cached_station_map(ck[0], ck[1], pm, run_key)
