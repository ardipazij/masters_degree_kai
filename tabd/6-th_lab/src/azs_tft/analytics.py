from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from azs_tft.model_registry import resolve_active_paths
from azs_tft.paths import artifacts_dir
from azs_tft.preprocess import load_panel, time_split

REQUIRED_COLUMNS = [
    "timestamp",
    "station_id",
    "total_fuel_sales",
    "shop_total_revenue",
    "total_traffic",
    "promotion_fuel_active",
    "price_AI92",
    "competitor_price_AI92",
]


def validate_csv_schema(csv_path: Path) -> pd.DataFrame:
    """Проверка CSV перед обучением / прогнозом. Возвращает таблицу статусов."""
    rows: list[dict[str, Any]] = []
    try:
        head = pd.read_csv(csv_path, nrows=5)
        cols = set(head.columns)
        for c in REQUIRED_COLUMNS:
            rows.append(
                {
                    "проверка": f"колонка `{c}`",
                    "статус": "OK" if c in cols else "ОШИБКА",
                    "детали": "есть" if c in cols else "отсутствует",
                }
            )
        df = load_panel(csv_path)
        dup = df.duplicated(subset=["station_id", "timestamp"]).sum()
        rows.append(
            {
                "проверка": "дубликаты station_id+timestamp",
                "статус": "OK" if dup == 0 else "ПРЕДУПРЕЖДЕНИЕ",
                "детали": str(int(dup)),
            }
        )
        rows.append(
            {
                "проверка": "объём данных",
                "статус": "OK",
                "детали": f"{len(df):,} строк, {df['station_id'].nunique()} АЗС",
            }
        )
        na = df[REQUIRED_COLUMNS].isna().mean().max()
        rows.append(
            {
                "проверка": "пропуски в ключевых полях",
                "статус": "OK" if na < 0.05 else "ПРЕДУПРЕЖДЕНИЕ",
                "детали": f"макс. доля NaN: {na:.1%}",
            }
        )
    except Exception as e:
        rows.append({"проверка": "чтение файла", "статус": "ОШИБКА", "детали": str(e)})
    return pd.DataFrame(rows)


def read_training_metrics_csv(run_id: str | None = None) -> pd.DataFrame | None:
    """Кривая loss: для выбранного run_id или последнего лога."""
    if run_id:
        p = artifacts_dir() / "runs" / run_id / "metrics.csv"
        if p.exists():
            return pd.read_csv(p)
    logs = artifacts_dir() / "logs"
    if not logs.exists():
        return None
    candidates = sorted(logs.rglob("metrics.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
    return pd.read_csv(candidates[0]) if candidates else None


def load_meta() -> dict[str, Any]:
    paths = resolve_active_paths()
    meta_p = paths["meta"]
    if meta_p.exists():
        return json.loads(meta_p.read_text(encoding="utf-8"))
    return {}


def list_runs_table() -> pd.DataFrame:
    """Таблица всех обучений для дашборда."""
    from azs_tft.model_registry import import_legacy_checkpoint, list_runs

    import_legacy_checkpoint()
    rows = list_runs()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def promotion_effect_stats(df: pd.DataFrame, metric: str = "total_fuel_sales") -> pd.DataFrame:
    if "promotion_fuel_active" not in df.columns or metric not in df.columns:
        return pd.DataFrame()
    g = df.groupby("promotion_fuel_active")[metric].mean().reset_index()
    g["promotion_fuel_active"] = g["promotion_fuel_active"].map({0: "Без акции", 1: "С акцией"})
    g.columns = ["режим", "среднее"]
    return g


def station_risk_table(csv_path: Path, top_n: int = 10) -> pd.DataFrame:
    """Сравнение среднего прогноза и базиса по АЗС (упрощённо по хвосту истории)."""
    from azs_tft.predict import build_validation_dataloader, forecast_steps_table, station_index_map

    df = load_panel(csv_path)
    tail_days = 14
    cutoff = df["timestamp"].max() - pd.Timedelta(days=tail_days)
    baseline = df[df["timestamp"] > cutoff].groupby("station_id")["total_fuel_sales"].mean()
    mapping = station_index_map(csv_path)
    rows = []
    for sid, idx in mapping.items():
        try:
            tbl = forecast_steps_table(csv_path, sample_idx=idx)
            fc = float(tbl.filter(like="прогноз_total_fuel").iloc[:, 0].mean())
        except Exception:
            fc = float("nan")
        b = float(baseline.get(sid, float("nan")))
        rel = (fc - b) / b if b and b == b and fc == fc else float("nan")
        name = df.loc[df["station_id"] == sid, "station_name"].iloc[0] if "station_name" in df.columns else str(sid)
        rows.append({"station_id": sid, "АЗС": name, "базис_14д": b, "прогноз_ср": fc, "отклонение_%": rel * 100 if rel == rel else None})
    out = pd.DataFrame(rows).sort_values("отклонение_%", na_position="last")
    return out.head(top_n)
