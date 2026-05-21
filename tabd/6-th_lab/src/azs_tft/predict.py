from __future__ import annotations

import inspect
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Callable

ProgressReporter = Callable[[int, int, str], None]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from azs_tft.model_registry import resolve_active_paths
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
    pytorch-forecasting 1.7+: predict() → Prediction(output=..., x=...).
    Старые версии: (raw_dict, x) или dict с ключами output/x.
    """
    if pred_out is None:
        raise ValueError("predict вернул None")
    if hasattr(pred_out, "output"):
        return pred_out.output, getattr(pred_out, "x", None)
    if isinstance(pred_out, dict):
        if "output" in pred_out:
            return pred_out["output"], pred_out.get("x")
        if "prediction" in pred_out and "x" in pred_out:
            return pred_out, pred_out["x"]
    if isinstance(pred_out, (list, tuple)) and len(pred_out) >= 2:
        return pred_out[0], pred_out[1]
    raise TypeError(f"Неожиданный тип результата predict: {type(pred_out)}")


def forward_output_as_dict(raw: Any) -> dict[str, Any]:
    """Словарь выхода forward (dict или namedtuple / OutputMixIn)."""
    if isinstance(raw, dict):
        return raw
    if hasattr(raw, "_asdict"):
        return raw._asdict()
    fields = getattr(raw, "_fields", None)
    if fields:
        return {f: getattr(raw, f) for f in fields}
    raise TypeError(f"Не удалось преобразовать выход сети в dict: {type(raw)}")


def tensors_to_numpy(value: Any) -> np.ndarray:
    """Тензор или список тензоров (MultiLoss) → numpy."""
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    if isinstance(value, (list, tuple)):
        parts = [tensors_to_numpy(v) for v in value]
        shaped = []
        for a in parts:
            if a.ndim == 2:
                a = a[:, :, np.newaxis]
            shaped.append(a.reshape(a.shape[0], a.shape[1], -1))
        return np.concatenate(shaped, axis=-1) if len(shaped) > 1 else shaped[0]
    raise TypeError(f"Ожидался tensor или list[tensor], получено {type(value)}")


def extract_prediction_array(raw: Any) -> np.ndarray:
    """Массив прогноза [batch, time, n_targets] из сырого выхода сети."""
    fwd = forward_output_as_dict(raw)
    if "prediction" not in fwd:
        keys = list(fwd.keys())[:15]
        raise KeyError(f"В выходе сети нет 'prediction'. Ключи: {keys}")
    pred = tensors_to_numpy(fwd["prediction"])
    while pred.ndim > 3:
        pred = pred[..., 0]
    if pred.ndim == 2:
        pred = pred[:, :, np.newaxis]
    return pred


def extract_decoder_target_array(x: Any) -> np.ndarray | None:
    if not isinstance(x, dict) or "decoder_target" not in x:
        return None
    act = tensors_to_numpy(x["decoder_target"])
    while act.ndim > 3:
        act = act[..., 0]
    if act.ndim == 2:
        act = act[:, :, np.newaxis]
    return act


def restore_network_output(model: TemporalFusionTransformer, raw: Any) -> Any:
    """PF TFT ожидает OutputMixIn с .iget(); plain dict — только без attention."""
    if hasattr(raw, "iget"):
        return raw
    if isinstance(raw, dict):
        try:
            return model.to_network_output(**raw)
        except Exception:
            return raw
    fwd = forward_output_as_dict(raw)
    try:
        return model.to_network_output(**fwd)
    except Exception:
        return fwd


_model_by_key: dict[tuple[str, int], TemporalFusionTransformer] = {}


def load_model(ckpt: Path | None = None) -> TemporalFusionTransformer:
    if ckpt is None:
        ckpt = resolve_active_paths()["checkpoint"]
    if not ckpt.exists():
        raise FileNotFoundError(f"Нет чекпоинта: {ckpt}. Сначала обучите модель: python -m azs_tft.train")
    key = (str(ckpt.resolve()), int(ckpt.stat().st_mtime_ns))
    cached = _model_by_key.get(key)
    if cached is not None:
        return cached
    _quiet_inference_logs()
    model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt))
    _model_by_key[key] = model
    return model


def load_dataset_params() -> dict[str, Any]:
    p = resolve_active_paths()["dataset_params"]
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


def station_index_map(csv_path: Path, val_hours: int = 336) -> dict[int, int]:
    """Соответствие station_id → индекс ряда в TimeSeriesDataSet (режим predict)."""
    df = load_panel(csv_path)
    _, val_df = time_split(df, val_hours=val_hours)
    params = load_dataset_params()
    ds = TimeSeriesDataSet.from_parameters(params, val_df, predict=True, stop_randomization=True)
    mapping: dict[int, int] = {}
    stations_sorted = sorted(val_df["station_id"].unique())
    for i in range(len(ds)):
        try:
            sample = ds[i]
            g = sample["groups"]
            sid = int(g.squeeze().cpu().numpy()) if hasattr(g, "cpu") else int(g)
        except Exception:
            sid = int(stations_sorted[i]) if i < len(stations_sorted) else i
        mapping[sid] = i
    if len(mapping) < len(stations_sorted):
        for i, sid in enumerate(stations_sorted):
            mapping.setdefault(int(sid), i)
    return mapping


def plot_prediction_example(
    csv_path: Path,
    sample_idx: int = 0,
    on_progress: ProgressReporter | None = None,
) -> plt.Figure:
    if on_progress:
        on_progress(1, 3, "Подготовка данных")
    model, loader, _ = build_validation_dataloader(csv_path)
    if on_progress:
        on_progress(2, 3, "Инференс TFT")
    pred_out = model.predict(loader, mode="raw", return_x=True)
    if on_progress:
        on_progress(3, 3, "Построение графика")
    raw, x = unpack_predict_raw_x(pred_out)
    out_for_plot = restore_network_output(model, raw)
    plt.close("all")
    kwargs: dict[str, Any] = {"idx": sample_idx, "add_loss_to_title": True}
    sig = inspect.signature(model.plot_prediction)
    if "return_fig" in sig.parameters:
        kwargs["return_fig"] = True
    if "plot_attention" in sig.parameters:
        kwargs["plot_attention"] = hasattr(out_for_plot, "iget")
    out = model.plot_prediction(x, out_for_plot, **kwargs)
    if isinstance(out, plt.Figure):
        return out
    return plt.gcf()


def forecast_steps_table(
    csv_path: Path,
    sample_idx: int = 0,
    on_progress: ProgressReporter | None = None,
) -> pd.DataFrame:
    """Таблица шагов прогноза по одному ряду из батча (индекс sample_idx в первом батче)."""
    total = 4

    def p(step: int, msg: str) -> None:
        if on_progress:
            on_progress(step, total, msg)

    p(1, "Подготовка валидационного датасета")
    model, loader, ds = build_validation_dataloader(csv_path)
    p(2, "Инференс TFT (1–3 мин на GPU, дольше на CPU)")
    pred_out = model.predict(loader, mode="raw", return_x=True)
    p(3, "Разбор выхода модели")
    raw, x = unpack_predict_raw_x(pred_out)
    pred = extract_prediction_array(raw)
    b, t, n_t = pred.shape
    si = int(np.clip(sample_idx, 0, b - 1))
    row = pred[si]
    targets = list(ds.target) if not isinstance(ds.target, str) else [ds.target]
    n_t = min(n_t, len(targets))
    data: dict[str, Any] = {"ч_вперёд": np.arange(1, row.shape[0] + 1)}
    for i in range(n_t):
        data[f"прогноз_{targets[i]}"] = row[:, i]
    act = extract_decoder_target_array(x)
    if act is not None:
        act_row = act[si]
        for i in range(min(n_t, act_row.shape[-1])):
            data[f"факт_{targets[i]}"] = act_row[:, i]
    p(4, "Формирование таблицы")
    return pd.DataFrame(data)


def variable_importance_tft(
    csv_path: Path,
    on_progress: ProgressReporter | None = None,
) -> pd.DataFrame:
    """Пытается извлечь веса внимания TFT; при ошибке — линейные корреляции с продажами."""
    try:
        if on_progress:
            on_progress(1, 3, "Подготовка данных")
        model, loader, _ = build_validation_dataloader(csv_path)
        if on_progress:
            on_progress(2, 3, "Инференс TFT для интерпретации")
        pred_out = model.predict(loader, mode="raw", return_x=True)
        if on_progress:
            on_progress(3, 3, "Расчёт важности признаков")
        raw, _ = unpack_predict_raw_x(pred_out)
        interp = model.interpret_output(forward_output_as_dict(raw), reduction="sum")
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
