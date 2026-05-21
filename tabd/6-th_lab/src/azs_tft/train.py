from __future__ import annotations

import argparse
import json
import os
import shutil
import warnings
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer, MultiNormalizer
from pytorch_forecasting.metrics import MAE, MultiLoss

from azs_tft.model_registry import register_run
from azs_tft.paths import artifacts_dir
from azs_tft.preprocess import default_csv, load_panel, time_split
from azs_tft.targets_config import FUEL_GRADE_TARGETS, default_model_targets


def _target_names(ds: TimeSeriesDataSet) -> list[str]:
    t = ds.target
    return [t] if isinstance(t, str) else list(t)


def build_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    max_encoder_length: int = 168,
    max_prediction_length: int = 24,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    targets = default_model_targets(include_total=True)
    n_targets = len(targets)
    static_categoricals = ["road_type", "direction", "settlement_size"]
    static_reals = [
        "distance_to_city_km",
        "total_pumps",
        "shop_area_m2",
        "has_car_wash",
        "has_cafe",
        "has_shop",
        "competitors_within_5km",
        "corporate_customer_ratio",
        "staff_engagement_score",
        "customer_loyalty_score",
    ]
    time_varying_known_categoricals = ["weather_condition", "season", "ad_channel"]
    time_varying_known_reals = [
        "hour",
        "day_of_week",
        "week_of_year",
        "month",
        "quarter",
        "is_weekend",
        "is_holiday",
        "is_rush_hour",
        "is_night",
        "promotion_fuel_active",
        "promotion_shop_active",
        "promotion_cafe_active",
        "ad_active",
        "competitor_price_AI92",
        "competitor_price_AI95",
        "competitor_price_DT",
        "price_AI92",
        "price_AI95",
        "price_AI98",
        "price_DT_EURO",
        "price_DT_TANEKO",
        "price_DT_SUMMER",
        "price_DT_WINTER",
        "traffic_Passengers_cars",
        "traffic_Truck_short",
        "traffic_Truck",
        "traffic_Truck_long",
        "traffic_Transporter",
        "traffic_Undefined",
        "total_traffic",
        "temperature",
        "precipitation_mm",
        "visibility_km",
        "wind_speed_ms",
        "is_snow",
        "is_rain",
        "is_fog",
    ]
    target_normalizer = MultiNormalizer(
        [GroupNormalizer(groups=["station_id"]) for _ in range(n_targets)]
    )
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=targets,
        group_ids=["station_id"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=static_categoricals,
        static_reals=static_reals,
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=[],
        target_normalizer=target_normalizer,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )
    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=True, stop_randomization=True)
    return training, validation


def train(
    csv_path: Path,
    max_epochs: int = 30,
    batch_size: int = 64,
    limit_train_batches: int | None = None,
    num_workers: int = 0,
) -> Path:
    warnings.filterwarnings(
        "ignore",
        message=".*The given NumPy array is not writable.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*already saved during checkpointing.*",
        category=UserWarning,
    )
    warnings.filterwarnings("ignore", message=".*LeafSpec.*")

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    df = load_panel(csv_path)
    train_df, val_df = time_split(df, val_hours=336)
    training, validation = build_datasets(train_df, val_df)
    targets = _target_names(training)
    n_targets = len(targets)
    kw: dict = {"batch_size": batch_size, "num_workers": num_workers}
    if num_workers > 0:
        kw["persistent_workers"] = True
    train_loader = training.to_dataloader(train=True, shuffle=True, **kw)
    val_loader = validation.to_dataloader(train=False, **{k: v for k, v in kw.items() if k != "shuffle"})

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=32,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=16,
        loss=MultiLoss([MAE() for _ in range(n_targets)]),
        output_size=[1 for _ in range(n_targets)],
        reduce_on_plateau_patience=3,
    )

    out_dir = artifacts_dir()
    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    lr_log = LearningRateMonitor()
    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir),
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    logger = CSVLogger(save_dir=str(out_dir / "logs"))

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        gradient_clip_val=0.1,
        logger=logger,
        callbacks=[early_stop, lr_log, ckpt_cb],
        enable_checkpointing=True,
        limit_train_batches=limit_train_batches if limit_train_batches else 1.0,
    )
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if ckpt_cb.best_model_path:
        shutil.copy(ckpt_cb.best_model_path, out_dir / "tft_azs.ckpt")
        ckpt_path = out_dir / "tft_azs.ckpt"
    else:
        fallback = out_dir / "tft_last.ckpt"
        trainer.save_checkpoint(str(fallback))
        shutil.copy(fallback, out_dir / "tft_azs.ckpt")
        ckpt_path = out_dir / "tft_azs.ckpt"
    params_path = out_dir / "tft_dataset_params.pt"
    torch.save(training.get_parameters(), params_path)
    meta = {
        "csv_path": str(csv_path.resolve()),
        "targets": targets,
        "fuel_grade_targets": list(FUEL_GRADE_TARGETS),
        "max_encoder_length": training.max_encoder_length,
        "max_prediction_length": training.max_prediction_length,
        "checkpoint": str(ckpt_path),
    }
    with open(out_dir / "tft_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log_dir = getattr(logger, "log_dir", None)
    run_id = register_run(
        csv_path,
        ckpt_path,
        params_path,
        meta,
        log_dir=log_dir,
        epochs=max_epochs,
        batch_size=batch_size,
        limit_train_batches=limit_train_batches,
        set_active=True,
    )
    print(f"Зарегистрирована модель run_id={run_id}")
    return ckpt_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Обучение TFT для продаж АЗС")
    ap.add_argument("--data", choices=["5stations", "full"], default="5stations")
    ap.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Путь к своему CSV (колонки как у detailed_data); если задан, ключ --data игнорируется",
    )
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit-train-batches", type=int, default=0, help="0 = без ограничения")
    ap.add_argument(
        "--num-workers",
        type=int,
        default=min(4, max(0, (os.cpu_count() or 4) - 1)),
        help="Число воркеров DataLoader (0 = только главный процесс; на GPU обычно 2–8 быстрее)",
    )
    args = ap.parse_args()
    if args.csv is not None:
        csv_path = args.csv.expanduser().resolve()
        if not csv_path.is_file():
            raise SystemExit(f"Файл не найден: {csv_path}")
    else:
        csv_path = default_csv(args.data)
    lim = args.limit_train_batches or None
    ckpt = train(
        csv_path,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        limit_train_batches=lim,
        num_workers=args.num_workers,
    )
    print(f"Сохранено: {ckpt}")


if __name__ == "__main__":
    main()
