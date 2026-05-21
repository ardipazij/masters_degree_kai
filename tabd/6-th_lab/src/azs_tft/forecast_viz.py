from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

ProgressReporter = Callable[[int, int, str], None]

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from azs_tft.paths import artifacts_dir
from azs_tft.dashboard_cache import active_targets
from azs_tft.predict import (
    build_validation_dataloader,
    extract_decoder_target_array,
    extract_prediction_array,
    forecast_steps_table,
    load_dataset_params,
    load_model,
    station_index_map,
    unpack_predict_raw_x,
)
from azs_tft.preprocess import load_panel, time_split
from azs_tft.targets_config import FUEL_GRADE_TARGETS, target_label, target_select_options


def target_options_for_csv(csv_path: Path | None = None) -> dict[str, str]:
    """Подписи показателей без загрузки TFT (из meta / dataset_params)."""
    try:
        return target_select_options(active_targets())
    except Exception:
        return target_select_options()


def _target_col(label: str, options: dict[str, str] | None = None) -> str:
    opts = options or target_select_options()
    return opts.get(label, label)


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return float("nan")
    yt, yp = y_true[mask], y_pred[mask]
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def forecast_series_for_station(
    csv_path: Path,
    station_id: int,
    target: str = "total_fuel_sales",
    history_hours: int = 72,
    forecast_tbl: pd.DataFrame | None = None,
    on_progress: ProgressReporter | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    История (факт) + прогноз по шагам для одной АЗС.
    Возвращает (history_df, forecast_df, sample_idx).
    """
    df = load_panel(csv_path)
    mapping = station_index_map(csv_path)
    if station_id not in mapping:
        raise KeyError(f"АЗС {station_id} нет в валидационном окне TFT")
    idx = mapping[station_id]
    tbl = forecast_tbl if forecast_tbl is not None else forecast_steps_table(
        csv_path, sample_idx=idx, on_progress=on_progress
    )
    pred_col = f"прогноз_{target}"
    if pred_col not in tbl.columns:
        pred_col = [c for c in tbl.columns if c.startswith("прогноз_")][0]
    fc = tbl[["ч_вперёд", pred_col]].rename(columns={pred_col: "значение"})
    fc["тип"] = "прогноз"
    fc["час_ось"] = fc["ч_вперёд"].values

    sub = df[df["station_id"] == station_id].sort_values("timestamp")
    hist = sub.tail(history_hours)[["timestamp", target]].copy()
    hist["тип"] = "факт"
    hist["час_ось"] = np.arange(-len(hist), 0)
    hist = hist.rename(columns={target: "значение"})
    # псевдо-квантили по разбросу остатков на истории
    resid_std = float(sub[target].diff().dropna().std()) if len(sub) > 2 else 0.0
    if resid_std > 0:
        fc["p10"] = fc["значение"] - 1.28 * resid_std
        fc["p90"] = fc["значение"] + 1.28 * resid_std
    else:
        fc["p10"] = fc["значение"]
        fc["p90"] = fc["значение"]
    return hist, fc, idx


def plot_forecast_plotly(
    csv_path: Path,
    station_id: int,
    target_label: str = "Сумма топливо",
    history_hours: int = 72,
    target_options: dict[str, str] | None = None,
    on_progress: ProgressReporter | None = None,
) -> go.Figure:
    opts = target_options or target_options_for_csv(csv_path)
    target = _target_col(target_label, opts)
    if on_progress:
        on_progress(1, 5, "Запуск прогноза")
    hist, fc, _ = forecast_series_for_station(
        csv_path, station_id, target, history_hours, on_progress=on_progress
    )
    if on_progress:
        on_progress(5, 5, "Отрисовка графика")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist["час_ось"],
            y=hist["значение"],
            mode="lines",
            name="Факт (история)",
            line=dict(color="#17becf", width=2),
        )
    )
    if "p10" in fc.columns:
        fig.add_trace(
            go.Scatter(
                x=fc["час_ось"],
                y=fc["p90"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fc["час_ось"],
                y=fc["p10"],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(139,69,19,0.25)",
                line=dict(width=0),
                name="P10–P90 (оценка)",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=fc["час_ось"],
            y=fc["значение"],
            mode="lines+markers",
            name="Прогноз (медиана)",
            line=dict(color="#ff7f0e", dash="dash"),
        )
    )
    fig.add_vline(x=0, line_width=1, line_dash="solid", line_color="white")
    sid_name = ""
    df = load_panel(csv_path)
    if "station_name" in df.columns:
        sid_name = df.loc[df["station_id"] == station_id, "station_name"].iloc[0]
    fig.update_layout(
        title=f"Прогноз TFT: {target_label} — {sid_name or station_id} (история {history_hours} ч., прогноз {len(fc)} ч.)",
        xaxis_title="Часы (отрицательные = история, положительные = прогноз)",
        yaxis_title="л/час или руб/час",
        template="plotly_dark",
        height=480,
    )
    return fig


def plot_fuel_grades_forecast_plotly(
    csv_path: Path,
    station_id: int,
    history_hours: int = 72,
    on_progress: ProgressReporter | None = None,
) -> go.Figure:
    """Прогноз TFT по каждой марке топлива (один инференс, несколько подграфиков)."""
    df = load_panel(csv_path)
    mapping = station_index_map(csv_path)
    if station_id not in mapping:
        raise KeyError(f"АЗС {station_id} нет в валидационном окне TFT")
    idx = mapping[station_id]
    tbl = forecast_steps_table(csv_path, sample_idx=idx, on_progress=on_progress)

    fuel_cols = [
        c.replace("прогноз_", "")
        for c in tbl.columns
        if c.startswith("прогноз_") and c.replace("прогноз_", "") in FUEL_GRADE_TARGETS
    ]
    if not fuel_cols:
        fuel_cols = [c for c in active_targets() if c in FUEL_GRADE_TARGETS] or list(FUEL_GRADE_TARGETS)

    n = len(fuel_cols)
    n_cols = 2
    n_rows = (n + n_cols - 1) // n_cols
    titles = [target_label(c) for c in fuel_cols]
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles, vertical_spacing=0.12)

    sub = df[df["station_id"] == station_id].sort_values("timestamp")
    for i, col in enumerate(fuel_cols):
        if on_progress:
            on_progress(i + 1, n, f"График: {target_label(col)}")
        r, c = i // n_cols + 1, i % n_cols + 1
        pred_col = f"прогноз_{col}"
        if pred_col not in tbl.columns:
            continue
        fc = tbl[["ч_вперёд", pred_col]].rename(columns={pred_col: "значение"})
        fc["час_ось"] = fc["ч_вперёд"].values
        hist = sub.tail(history_hours)[["timestamp", col]].copy()
        hist["час_ось"] = np.arange(-len(hist), 0)
        hist = hist.rename(columns={col: "значение"})
        fig.add_trace(
            go.Scatter(x=hist["час_ось"], y=hist["значение"], mode="lines", name="факт", line=dict(color="#2563eb")),
            row=r,
            col=c,
        )
        fig.add_trace(
            go.Scatter(
                x=fc["час_ось"],
                y=fc["значение"],
                mode="lines+markers",
                name="прогноз",
                line=dict(color="#ea580c", dash="dash"),
            ),
            row=r,
            col=c,
        )
        fig.update_xaxes(title_text="ч", row=r, col=c)
        fig.update_yaxes(title_text="л/ч", row=r, col=c)

    fig.update_layout(
        template="plotly_white",
        height=280 * n_rows,
        title="Прогноз по маркам топлива (TFT)",
        showlegend=False,
        margin=dict(t=60, b=40),
    )
    return fig


def scenario_promo_compare(
    csv_path: Path,
    station_id: int,
    promo_on: bool = True,
    on_progress: ProgressReporter | None = None,
) -> pd.DataFrame:
    """
    Сравнение среднего прогноза топлива: базовый val vs сценарий «акция на топливо» на decoder-горизонте.
    Упрощение: меняем promotion_fuel_active на последних max_prediction_length часах для АЗС.
    """
    from pytorch_forecasting import TimeSeriesDataSet

    df = load_panel(csv_path)
    _, val_df = time_split(df)
    params = load_dataset_params()
    meta_horizon = 24
    meta_path = artifacts_dir() / "tft_meta.json"
    if meta_path.exists():
        import json

        meta_horizon = int(json.loads(meta_path.read_text(encoding="utf-8")).get("max_prediction_length", 24))

    def _predict_on_frame(frame: pd.DataFrame) -> float:
        ds = TimeSeriesDataSet.from_parameters(params, frame, predict=True, stop_randomization=True)
        loader = ds.to_dataloader(train=False, batch_size=128, num_workers=0)
        model = load_model()
        mapping = station_index_map(csv_path)
        if station_id not in mapping:
            return float("nan")
        pred_out = model.predict(loader, mode="raw", return_x=True)
        raw, _ = unpack_predict_raw_x(pred_out)
        pred = extract_prediction_array(raw)
        si = mapping[station_id]
        tlist = list(ds.target) if not isinstance(ds.target, str) else [ds.target]
        ti = tlist.index("total_fuel_sales") if "total_fuel_sales" in tlist else 0
        return float(pred[si, :, ti].mean())

    if on_progress:
        on_progress(1, 3, "Базовый сценарий — инференс")
    base_mean = _predict_on_frame(val_df)
    scen = val_df.copy()
    mask = scen["station_id"] == station_id
    tail_idx = scen.loc[mask].sort_values("timestamp").tail(meta_horizon).index
    scen.loc[tail_idx, "promotion_fuel_active"] = 1 if promo_on else 0
    if on_progress:
        on_progress(2, 3, "Сценарий с акцией — инференс")
    scen_mean = _predict_on_frame(scen)
    if on_progress:
        on_progress(3, 3, "Сводка")
    return pd.DataFrame(
        [
            {"сценарий": "Базовый (как в данных)", "ср._прогноз_топливо": base_mean},
            {"сценарий": "Акция на топливо (decoder)", "ср._прогноз_топливо": scen_mean},
            {"сценарий": "Δ, %", "ср._прогноз_топливо": (scen_mean - base_mean) / base_mean * 100 if base_mean else None},
        ]
    )


def evaluate_validation_metrics(
    csv_path: Path,
    max_batches: int = 50,
    on_progress: ProgressReporter | None = None,
) -> pd.DataFrame:
    """MAE и R² на валидации (нормализованный масштаб GroupNormalizer)."""
    if on_progress:
        on_progress(1, 4, "Подготовка валидации")
    model, loader, ds = build_validation_dataloader(csv_path, batch_size=64)
    targets = list(ds.target) if not isinstance(ds.target, str) else [ds.target]

    try:
        if on_progress:
            on_progress(2, 4, f"Инференс (до {max_batches or 'всех'} батчей)")
        trainer_kw = {"limit_predict_batches": max_batches} if max_batches else {}
        pred_out = model.predict(
            loader, mode="raw", return_x=True, trainer_kwargs=trainer_kw
        )
        if on_progress:
            on_progress(3, 4, "Расчёт MAE и R²")
        raw, x = unpack_predict_raw_x(pred_out)
    except Exception as e:
        return pd.DataFrame([{"цель": "ошибка", "MAE": None, "R²": None, "детали": str(e)}])

    try:
        pred = extract_prediction_array(raw)
    except (KeyError, TypeError) as e:
        return pd.DataFrame([{"цель": "ошибка", "MAE": None, "R²": None, "детали": str(e)}])

    act = extract_decoder_target_array(x)
    if act is None:
        return pd.DataFrame([{"цель": "ошибка", "MAE": None, "R²": None, "детали": "нет decoder_target в x"}])

    n_t = min(pred.shape[-1], act.shape[-1], len(targets))
    pred_flat = pred.reshape(-1, pred.shape[-1])
    act_flat = act.reshape(-1, act.shape[-1])

    note = f"валидация, до {max_batches or 'всех'} батчей, нормализованный масштаб"
    rows = []
    for j in range(n_t):
        p_j, a_j = pred_flat[:, j], act_flat[:, j]
        mae = float(np.abs(p_j - a_j).mean())
        r2 = _r2_score(a_j, p_j)
        rows.append(
            {
                "цель": target_label(targets[j]),
                "колонка": targets[j],
                "MAE": round(mae, 4),
                "R²": round(r2, 4) if r2 == r2 else None,
                "детали": note,
            }
        )
    if on_progress:
        on_progress(4, 4, "Готово")
    return pd.DataFrame(rows)


def evaluate_validation_mae(csv_path: Path, max_batches: int = 50) -> pd.DataFrame:
    """Обратная совместимость."""
    return evaluate_validation_metrics(csv_path, max_batches=max_batches)
