from __future__ import annotations

import pandas as pd


def _mean_spread(df: pd.DataFrame, our: str, comp: str) -> float:
    if our not in df.columns or comp not in df.columns:
        return 0.0
    return float((df[comp] - df[our]).mean())


def recommend(df_tail: pd.DataFrame, forecast_fuel: float | None, baseline_fuel: float | None) -> list[str]:
    """
    Эвристические рекомендации по хвосту истории и (опционально) среднему прогнозу топлива.
    forecast_fuel / baseline_fuel — средние значения по горизонту (условные единицы/час).
    """
    out: list[str] = []
    if df_tail.empty:
        return ["Недостаточно данных для рекомендаций."]

    d92 = _mean_spread(df_tail, "price_AI92", "competitor_price_AI92")
    ddt = _mean_spread(df_tail, "price_DT_EURO", "competitor_price_DT")

    if d92 > 0.4:
        out.append(
            "Средняя цена АИ-92 заметно выше конкурентов: оценить эластичность спроса и пакеты с кафе/магазином."
        )
    elif d92 < -0.4:
        out.append("Конкуренты дороже по АИ-92: можно усилить рекламу данного маркет-микса без агрессивного демпинга.")

    if ddt > 0.5:
        out.append("ДТ: ценовое давление конкурентов — проверить акции на дизель и долю грузового трафика.")

    promo_f = float(df_tail["promotion_fuel_active"].mean()) if "promotion_fuel_active" in df_tail else 0.0
    ad = float(df_tail["ad_active"].mean()) if "ad_active" in df_tail else 0.0
    if promo_f < 0.15:
        out.append("Низкая доля часов с топливной акцией: протестировать короткие промо в часы пикового трафика.")

    if ad < 0.1:
        out.append("Реклама активна редко — синхронизировать каналы (ТВ/наружка/цифра) с пиками total_traffic.")

    tr = df_tail["total_traffic"].mean() if "total_traffic" in df_tail else None
    shop_rev = df_tail["shop_total_revenue"].mean() if "shop_total_revenue" in df_tail else None
    if tr and tr > 0 and shop_rev is not None and shop_rev / tr < 0.5:
        out.append("Выручка магазина слабо коррелирует с трафиком: развить cross-sell (напитки/кофе) в часы грузового потока.")

    if forecast_fuel is not None and baseline_fuel is not None and baseline_fuel > 0:
        rel = (forecast_fuel - baseline_fuel) / baseline_fuel
        if rel < -0.08:
            out.append(
                "Прогноз топлива ниже недавнего базиса: усилить промо/проверить цены конкурентов и погодные риски снабжения."
            )
        elif rel > 0.08:
            out.append("Прогноз топлива выше базиса: подготовить запасы и персонал пика, рассмотреть upsell в магазине.")

    if not out:
        out.append("Явных отклонений по правилам не найдено — ориентироваться на важность признаков TFT и локальные KPI.")
    return out
