from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path
from typing import BinaryIO, Tuple

import numpy as np
import pandas as pd

from azs_tft.paths import project_root


def _ensure_writable_numeric_arrays(df: pd.DataFrame) -> pd.DataFrame:
    """Гарантирует C-contiguous writable numpy-бэкенд (снижает UserWarning torch.from_numpy в PF)."""
    out = df.copy()
    for c in out.columns:
        if c == "timestamp":
            continue
        s = out[c]
        if pd.api.types.is_numeric_dtype(s):
            a = np.array(s.to_numpy(), copy=True)
            out[c] = np.require(a, requirements=["W", "C"])
    return out


def load_data_dictionary() -> dict:
    p = Path(__file__).resolve().parent / "data_dictionary.json"
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def persist_uploaded_csv(data: bytes | BinaryIO, original_name: str = "upload.csv") -> Path:
    """Сохраняет загруженный CSV во временный каталог, возвращает путь для load_panel."""
    d = Path(tempfile.gettempdir()) / "azs_tft_streamlit"
    d.mkdir(parents=True, exist_ok=True)
    safe = Path(original_name).name or "upload.csv"
    path = d / f"{uuid.uuid4().hex}_{safe}"
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_bytes(data.read())
    return path


def load_panel(csv_path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    if "day_name" not in df.columns and "day_of_week" in df.columns:
        df["day_name"] = df["timestamp"].dt.day_name()
    for col in ["ad_channel", "holiday_name", "weather_condition", "season"]:
        if col in df.columns:
            df[col] = df[col].fillna("none").astype(str)
    for col in df.select_dtypes(include=["float", "int"]).columns:
        if col == "station_id":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    df["time_idx"] = df.groupby("station_id").cumcount().astype(int)
    return _ensure_writable_numeric_arrays(df)


def time_split(df: pd.DataFrame, val_hours: int = 336) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Последние val_hours часов (по всей сети, глобальный cutoff) — валидация."""
    tmax = df["timestamp"].max()
    cutoff = tmax - pd.Timedelta(hours=val_hours)
    train = df[df["timestamp"] <= cutoff].copy()
    val = df[df["timestamp"] > cutoff].copy()
    return train, val


def default_csv(which: str = "5stations") -> Path:
    root = project_root()
    if which == "full":
        return root / "detailed_data.csv"
    return root / "5stations_data.csv"
