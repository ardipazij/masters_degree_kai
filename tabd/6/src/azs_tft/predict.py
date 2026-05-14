from __future__ import annotations

import inspect
import logging
import os
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from azs_tft.paths import artifacts_dir
from azs_tft.preprocess import load_panel, time_split

_QUIET_DONE = False


def _quiet_inference_logs() -> None:
    """Меньше шума в терминале при predict из Streamlit (Lightning/PF)."""
    global _QUIET_DONE
    if _QUIET_DONE:
        return
    _QUIET_DONE = True
    os.environ.setdefault("LIGHTNING_DISABLE_TIPS", "1")
    warnings.filterwarnings("ignore", message=".*The given NumPy array is not writable.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*already saved during checkpointing.*", category=UserWarning)
    # Lightning + jaxtyping: DeprecationWarning, не UserWarning
    warnings.filterwarnings("ignore", message=".*LeafSpec.*")
    warnings.filterwarnings("ignore", message=".*does not have many workers.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*tensorboardX.*", category=UserWarning)
    for name in (
        "lightning",
        "lightning.pytorch",
        "lightning.pytorch.trainer",
        "lightning.fabric",
        "pytorch_lightning",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")


def unpack_predict_raw_x(pred_out: Any) -> tuple[Any, Any]:
    """
    Совместимость версий pytorch-forecasting / Lightning:
    predict(..., mode='raw', return_x=True) может вернуть (raw, x), (raw, x, y) или иной кортеж.
    """
    if isinstance(pred_out, dict):
        if "output" in pred_out and "x" in pred_out:
            return pred_out["output"], pred_out["x"]
        if "prediction" in pred_out and "x" in pred_out:
            return pred_out, pred_out["x"]
    if isinstance(pred_out, (list, tuple)):
        if len(pred_out) >= 2:
            return pred_out[0], pred_out[1]
        raise ValueError(f"predict вернул последовательность длины {len(pred_out)}, ожидалось ≥ 2")
    raise TypeError(f"Неожиданный тип результата predict: {type(pred_out)}")


def load_model(ckpt: Path | None = None) -> TemporalFusionTransformer:
    root = artifacts_dir()
    path = ckpt or (root / "tft_azs.ckpt")
    if not path.exists():
        raise FileNotFoundError(f"Нет чекпоинта: {path}. Сначала обучите модель: python -m azs_tft.train")
    _quiet_inference_logs()
    return TemporalFusionTransformer.load_from_checkpoint(str(path))


def load_dataset_params() -> dict[str, Any]:
    p = artifacts_dir() / "tft_dataset_params.pt"
    if not p.exists():
        raise FileNotFoundError(f"Нет параметров датасета: {p}")
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(p, map_location="cpu")


def build_validation_dataloader(
    csv_path: Path,
    batch_size: int = 128,
    val_hours: int = 336,
    num_workers: int | None = None,
) -> tuple[TemporalFusionTransformer, Any, TimeSeriesDataSet]:
    _quiet_inference_logs()
    if num_workers is None:
        num_workers = min(4, max(0, (os.cpu_count() or 4) - 1))
    df = load_panel(csv_path)
    _, val_df = time_split(df, val_hours=val_hours)
    params = load_dataset_params()
    ds = TimeSeriesDataSet.from_parameters(params, val_df, predict=True, stop_randomization=True)
    dl_kw: dict[str, Any] = {"train": False, "batch_size": batch_size, "num_workers": num_workers}
    if num_workers > 0:
        dl_kw["persistent_workers"] = True
    loader = ds.to_dataloader(**dl_kw)
    model = load_model()
    model.eval()
    return model, loader, ds


def plot_prediction_example(
    csv_path: Path,
    sample_idx: int = 0,
) -> plt.Figure:
    model, loader, _ = build_validation_dataloader(csv_path)
    pred_out = model.predict(loader, mode="raw", return_x=True)
    raw, x = unpack_predict_raw_x(pred_out)
    plt.close("all")
    kwargs: dict[str, Any] = {"idx": sample_idx, "add_loss_to_title": True}
    if "return_fig" in inspect.signature(model.plot_prediction).parameters:
        kwargs["return_fig"] = True
    out = model.plot_prediction(x, raw, **kwargs)
    if isinstance(out, plt.Figure):
        return out
    return plt.gcf()


def forecast_steps_table(csv_path: Path, sample_idx: int = 0) -> pd.DataFrame:
    """Таблица шагов прогноза по одному ряду из батча (индекс sample_idx в первом батче)."""
    model, loader, ds = build_validation_dataloader(csv_path)
    pred_out = model.predict(loader, mode="raw", return_x=True)
    raw, x = unpack_predict_raw_x(pred_out)
    if "prediction" not in raw:
        raise KeyError("В выходе predict нет ключа 'prediction'")
    pred = raw["prediction"].float().detach().cpu().numpy()
    if pred.ndim == 4:
        pred = pred[:, :, 0, :]
    elif pred.ndim != 3:
        raise ValueError(f"Неожиданная форма prediction: {pred.shape}")
    b, t, n_t = pred.shape
    si = int(np.clip(sample_idx, 0, b - 1))
    row = pred[si]
    targets = list(ds.target) if not isinstance(ds.target, str) else [ds.target]
    n_t = min(n_t, len(targets))
    data: dict[str, Any] = {"ч_вперёд": np.arange(1, row.shape[0] + 1)}
    for i in range(n_t):
        data[f"прогноз_{targets[i]}"] = row[:, i]
    if isinstance(x, dict) and "decoder_target" in x:
        act = x["decoder_target"][si].float().detach().cpu().numpy()
        if act.ndim == 3:
            act = act[:, 0, :]
        for i in range(min(n_t, act.shape[-1])):
            data[f"факт_{targets[i]}"] = act[:, i]
    return pd.DataFrame(data)


def variable_importance_tft(csv_path: Path) -> pd.DataFrame:
    """Пытается извлечь веса внимания TFT; при ошибке — линейные корреляции с продажами."""
    try:
        model, loader, _ = build_validation_dataloader(csv_path)
        pred_out = model.predict(loader, mode="raw", return_x=True)
        raw, _ = unpack_predict_raw_x(pred_out)
        interp = model.interpret_output(raw, reduction="sum")
        if "attention" in interp:
            imp = interp["attention"]
            while imp.dim() > 1:
                imp = imp.sum(dim=0)
            imp = imp.detach().cpu().numpy()
            names = list(model.encoder_variable_names)
            m = min(len(names), len(imp))
            return (
                pd.DataFrame({"variable": names[:m], "importance": imp[:m]})
                .sort_values("importance", ascending=False)
                .head(25)
            )
    except Exception:
        pass
    df = load_panel(csv_path)
    cols = [
        "total_traffic",
        "promotion_fuel_active",
        "ad_active",
        "competitor_price_AI92",
        "price_AI92",
        "temperature",
        "is_weekend",
        "is_holiday",
    ]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame()
    corr = df[cols + ["total_fuel_sales"]].corr(numeric_only=True)["total_fuel_sales"].drop("total_fuel_sales")
    out = corr.abs().sort_values(ascending=False).reset_index()
    out.columns = ["variable", "importance"]
    return out.head(25)
