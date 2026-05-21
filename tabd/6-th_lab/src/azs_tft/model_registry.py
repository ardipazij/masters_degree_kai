from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from azs_tft.paths import artifacts_dir

REGISTRY_NAME = "models_registry.json"
ACTIVE_NAME = "active_model.json"
RUNS_SUBDIR = "runs"


def _registry_path() -> Path:
    return artifacts_dir() / REGISTRY_NAME


def _active_path() -> Path:
    return artifacts_dir() / ACTIVE_NAME


def _runs_root() -> Path:
    p = artifacts_dir() / RUNS_SUBDIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_registry() -> dict[str, Any]:
    p = _registry_path()
    if not p.exists():
        return {"runs": [], "active_run_id": None}
    return json.loads(p.read_text(encoding="utf-8"))


def _save_registry(data: dict[str, Any]) -> None:
    _registry_path().write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _dataset_label(csv_path: Path) -> str:
    name = csv_path.name
    if name in ("detailed_data.csv", "5stations_data.csv"):
        return name
    return f"upload:{name}"


def _metrics_from_logger(log_dir: str | Path | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not log_dir:
        return out
    metrics_file = Path(log_dir) / "metrics.csv"
    if not metrics_file.exists():
        candidates = sorted(Path(log_dir).rglob("metrics.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            return out
        metrics_file = candidates[0]
    try:
        m = pd.read_csv(metrics_file)
        if "val_loss" in m.columns:
            v = m["val_loss"].dropna()
            if len(v):
                out["val_loss_min"] = float(v.min())
                out["val_loss_last"] = float(v.iloc[-1])
        for col in ("train_loss", "train_loss_epoch"):
            if col in m.columns:
                t = m[col].dropna()
                if len(t):
                    out[f"{col}_last"] = float(t.iloc[-1])
                    break
        out["epochs_logged"] = int(m["epoch"].max()) if "epoch" in m.columns and len(m) else None
        out["metrics_csv"] = str(metrics_file)
    except Exception as e:
        out["metrics_error"] = str(e)
    return out


def register_run(
    csv_path: Path,
    checkpoint: Path,
    dataset_params_src: Path,
    meta: dict[str, Any],
    *,
    log_dir: str | Path | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    limit_train_batches: int | None = None,
    set_active: bool = True,
) -> str:
    """
    Сохраняет версию модели в artifacts/runs/<run_id>/ и дописывает реестр.
    По-прежнему обновляет tft_azs.ckpt в корне artifacts (совместимость).
    """
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = _runs_root() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dst = run_dir / "tft.ckpt"
    params_dst = run_dir / "dataset_params.pt"
    shutil.copy2(checkpoint, ckpt_dst)
    shutil.copy2(dataset_params_src, params_dst)

    metrics = _metrics_from_logger(log_dir)
    run_meta = {
        **meta,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "csv_path": str(csv_path.resolve()),
        "dataset_label": _dataset_label(csv_path),
        "checkpoint": str(ckpt_dst),
        "dataset_params": str(params_dst),
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "limit_train_batches": limit_train_batches,
        **metrics,
    }
    (run_dir / "meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if metrics.get("metrics_csv"):
        try:
            shutil.copy2(metrics["metrics_csv"], run_dir / "metrics.csv")
        except OSError:
            pass

    reg = _load_registry()
    reg["runs"] = [r for r in reg.get("runs", []) if r.get("run_id") != run_id]
    reg["runs"].insert(
        0,
        {
            "run_id": run_id,
            "created_at": run_meta["created_at"],
            "dataset_label": run_meta["dataset_label"],
            "csv_path": run_meta["csv_path"],
            "val_loss_min": metrics.get("val_loss_min"),
            "train_loss_last": metrics.get("train_loss_last") or metrics.get("train_loss_epoch_last"),
            "epochs_requested": epochs,
            "checkpoint": str(ckpt_dst),
        },
    )
    if set_active:
        reg["active_run_id"] = run_id
        _active_path().write_text(json.dumps({"run_id": run_id}, indent=2), encoding="utf-8")
    _save_registry(reg)
    return run_id


def list_runs() -> list[dict[str, Any]]:
    reg = _load_registry()
    runs = reg.get("runs", [])
    active = reg.get("active_run_id") or _get_active_id()
    for r in runs:
        r["is_active"] = r.get("run_id") == active
    return runs


def get_run(run_id: str) -> dict[str, Any] | None:
    run_dir = _runs_root() / run_id
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    for r in _load_registry().get("runs", []):
        if r.get("run_id") == run_id:
            return r
    return None


def _get_active_id() -> str | None:
    ap = _active_path()
    if ap.exists():
        try:
            return json.loads(ap.read_text(encoding="utf-8")).get("run_id")
        except json.JSONDecodeError:
            pass
    return _load_registry().get("active_run_id")


def set_active(run_id: str) -> None:
    meta = get_run(run_id)
    if meta is None and not (_runs_root() / run_id / "tft.ckpt").exists():
        raise FileNotFoundError(f"Запуск {run_id} не найден")
    reg = _load_registry()
    reg["active_run_id"] = run_id
    _save_registry(reg)
    _active_path().write_text(json.dumps({"run_id": run_id}, indent=2), encoding="utf-8")
    # синхронизация legacy-файлов для старого кода
    run_dir = _runs_root() / run_id
    ckpt = run_dir / "tft.ckpt"
    params = run_dir / "dataset_params.pt"
    root = artifacts_dir()
    if ckpt.exists():
        shutil.copy2(ckpt, root / "tft_azs.ckpt")
    if params.exists():
        shutil.copy2(params, root / "tft_dataset_params.pt")
    if (run_dir / "meta.json").exists():
        shutil.copy2(run_dir / "meta.json", root / "tft_meta.json")


def resolve_active_paths() -> dict[str, Path]:
    """
    Пути активной модели. Fallback: legacy tft_azs.ckpt в корне artifacts.
    """
    root = artifacts_dir()
    run_id = _get_active_id()
    if run_id:
        run_dir = _runs_root() / run_id
        ckpt = run_dir / "tft.ckpt"
        params = run_dir / "dataset_params.pt"
        meta = run_dir / "meta.json"
        if ckpt.exists() and params.exists():
            return {"checkpoint": ckpt, "dataset_params": params, "meta": meta if meta.exists() else root / "tft_meta.json", "run_id": run_id}
    return {
        "checkpoint": root / "tft_azs.ckpt",
        "dataset_params": root / "tft_dataset_params.pt",
        "meta": root / "tft_meta.json",
        "run_id": None,
    }


def import_legacy_checkpoint() -> str | None:
    """Добавляет в реестр существующий tft_azs.ckpt, если реестр пуст."""
    root = artifacts_dir()
    ckpt = root / "tft_azs.ckpt"
    params = root / "tft_dataset_params.pt"
    if not ckpt.exists() or not params.exists():
        return None
    reg = _load_registry()
    if reg.get("runs"):
        return reg.get("active_run_id")
    meta_old = {}
    if (root / "tft_meta.json").exists():
        meta_old = json.loads((root / "tft_meta.json").read_text(encoding="utf-8"))
    csv = Path(meta_old.get("csv_path", "unknown.csv"))
    run_id = "legacy_import"
    run_dir = _runs_root() / run_id
    if not run_dir.exists():
        run_dir.mkdir(parents=True)
        shutil.copy2(ckpt, run_dir / "tft.ckpt")
        shutil.copy2(params, run_dir / "dataset_params.pt")
        (run_dir / "meta.json").write_text(json.dumps({**meta_old, "run_id": run_id}, ensure_ascii=False, indent=2), encoding="utf-8")
    reg["runs"] = [
        {
            "run_id": run_id,
            "created_at": meta_old.get("created_at", "imported"),
            "dataset_label": _dataset_label(csv) if csv.exists() else str(csv),
            "csv_path": str(csv),
            "val_loss_min": None,
            "checkpoint": str(run_dir / "tft.ckpt"),
        }
    ]
    reg["active_run_id"] = run_id
    _save_registry(reg)
    _active_path().write_text(json.dumps({"run_id": run_id}, indent=2), encoding="utf-8")
    return run_id
