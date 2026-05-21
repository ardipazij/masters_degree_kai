from __future__ import annotations

from pathlib import Path

import pandas as pd

from azs_tft.analytics import load_meta, promotion_effect_stats, read_training_metrics_csv
from azs_tft.preprocess import load_panel
from azs_tft.recommendations import recommend


def build_report_markdown(csv_path: Path) -> str:
    df = load_panel(csv_path)
    meta = load_meta()
    lines = [
        "# Аналитический отчёт — сеть АЗС",
        "",
        f"- Строк в данных: **{len(df):,}**",
        f"- АЗС: **{df['station_id'].nunique()}**",
        f"- Период: **{df['timestamp'].min()}** — **{df['timestamp'].max()}**",
        "",
        "## Модель",
    ]
    if meta:
        lines.append(f"- Горизонт прогноза: **{meta.get('max_prediction_length')}** ч.")
        lines.append(f"- Encoder: **{meta.get('max_encoder_length')}** ч.")
        lines.append(f"- CSV обучения: `{meta.get('csv_path')}`")
    m = read_training_metrics_csv()
    if m is not None and "val_loss" in m.columns:
        best = m["val_loss"].dropna().min()
        lines.append(f"- Лучший val_loss (лог): **{best:.4f}**")
    lines.extend(["", "## Операционные рекомендации (14 суток)"])
    tail = df[df["timestamp"] > df["timestamp"].max() - pd.Timedelta(days=14)]
    base = float(tail["total_fuel_sales"].mean()) if len(tail) else 0.0
    for r in recommend(tail, forecast_fuel=base, baseline_fuel=base):
        lines.append(f"- {r}")
    promo = promotion_effect_stats(df)
    if not promo.empty:
        lines.extend(["", "## Эффект акции на топливо (средние по всему ряду)"])
        for _, row in promo.iterrows():
            lines.append(f"- {row['режим']}: {row['среднее']:.2f}")
    lines.append("")
    lines.append("_Сгенерировано из дашборда Streamlit._")
    return "\n".join(lines)
