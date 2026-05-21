"""Интерактивные графики Plotly для дашборда."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Segoe UI, Roboto, sans-serif", size=12),
    margin=dict(l=48, r=24, t=48, b=40),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

def _linear_trend_coeffs(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    """k, b для y ≈ k*x + b; None если цена почти не меняется (мало точек по оси X)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.unique(x).size < 2 or np.std(x) < 1e-4:
        return None
    design = np.column_stack([x, np.ones(len(x), dtype=np.float64)])
    coef, _, rank, _ = np.linalg.lstsq(design, y, rcond=1e-10)
    if rank < 2:
        return None
    return float(coef[0]), float(coef[1])


METRIC_LABELS = {
    "total_fuel_sales": "Продажи топлива, л/ч",
    "shop_total_revenue": "Выручка магазина, руб/ч",
    "total_traffic": "Суммарный трафик",
    "price_AI92": "Цена АИ-92, руб",
    "competitor_price_AI92": "Цена конкурента АИ-92",
    "temperature": "Температура, °C",
}


def _apply(fig: go.Figure, title: str, y_title: str | None = None) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT, title=title)
    if y_title:
        fig.update_yaxes(title_text=y_title)
    fig.update_xaxes(title_text="Дата")
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), rangeselector=dict(
        buttons=[
            dict(count=7, label="7д", step="day", stepmode="backward"),
            dict(count=30, label="30д", step="day", stepmode="backward"),
            dict(count=90, label="90д", step="day", stepmode="backward"),
            dict(step="all", label="Все"),
        ]
    )))
    return fig


def filter_period(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    m = (df["timestamp"] >= start) & (df["timestamp"] <= end)
    return df.loc[m].copy()


def daily_series(
    df: pd.DataFrame,
    value_cols: Iterable[str],
    agg: str = "sum",
) -> pd.DataFrame:
    g = df.set_index("timestamp").groupby(pd.Grouper(freq="1D"))
    if agg == "mean":
        out = g[list(value_cols)].mean()
    else:
        out = g[list(value_cols)].sum()
    return out.reset_index()


def chart_daily_multiline(df: pd.DataFrame, cols: list[str], title: str, agg: str = "sum") -> go.Figure:
    daily = daily_series(df, cols, agg=agg)
    long = daily.melt("timestamp", cols, "показатель", "значение")
    long["показатель"] = long["показатель"].map(lambda c: METRIC_LABELS.get(c, c))
    fig = px.line(long, x="timestamp", y="значение", color="показатель")
    return _apply(fig, title, "Значение")


def chart_hourly_profile(df: pd.DataFrame, metric: str) -> go.Figure:
    if "hour" not in df.columns:
        df = df.copy()
        df["hour"] = df["timestamp"].dt.hour
    prof = df.groupby("hour")[metric].mean().reset_index()
    fig = px.bar(
        prof,
        x="hour",
        y=metric,
        labels={"hour": "Час суток", metric: METRIC_LABELS.get(metric, metric)},
    )
    fig.update_layout(**PLOTLY_LAYOUT, title="Профиль по часам (среднее)", bargap=0.15)
    fig.update_xaxes(dtick=1)
    return fig


def chart_heatmap_dow_hour(df: pd.DataFrame, metric: str) -> go.Figure:
    d = df.copy()
    d["hour"] = d["timestamp"].dt.hour
    d["dow"] = d["timestamp"].dt.dayofweek
    dow_names = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    pivot = d.pivot_table(index="dow", columns="hour", values=metric, aggfunc="mean")
    pivot.index = [dow_names[i] for i in pivot.index]
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(x="Час", y="День недели", color=METRIC_LABELS.get(metric, metric)),
    )
    fig.update_layout(**PLOTLY_LAYOUT, title="Тепловая карта: день недели × час")
    return fig


def chart_fuel_structure_pie(df: pd.DataFrame) -> go.Figure | None:
    fuel_cols = [c for c in df.columns if c.startswith("sales_")]
    if not fuel_cols:
        return None
    s = df[fuel_cols].sum()
    fig = px.pie(
        values=s.values,
        names=[c.replace("sales_", "") for c in s.index],
        title="Структура продаж по маркам",
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def chart_traffic_area(df: pd.DataFrame) -> go.Figure | None:
    tr_cols = [c for c in df.columns if c.startswith("traffic_") and c != "total_traffic"]
    if not tr_cols:
        return None
    daily = daily_series(df, tr_cols, agg="mean")
    long = daily.melt("timestamp", tr_cols, "тип", "интенсивность")
    fig = px.area(long, x="timestamp", y="интенсивность", color="тип", line_group="тип")
    return _apply(fig, "Трафик по типам ТС (суточное среднее)", "Интенсивность")


def chart_price_vs_sales(df: pd.DataFrame) -> go.Figure | None:
    if "price_AI92" not in df.columns or "total_fuel_sales" not in df.columns:
        return None
    sample = df[["price_AI92", "total_fuel_sales", "competitor_price_AI92"]].dropna()
    if len(sample) > 8000:
        sample = sample.sample(8000, random_state=42)
    fig = px.scatter(
        sample,
        x="price_AI92",
        y="total_fuel_sales",
        opacity=0.35,
        labels={
            "price_AI92": METRIC_LABELS["price_AI92"],
            "total_fuel_sales": METRIC_LABELS["total_fuel_sales"],
        },
    )
    x = sample["price_AI92"].to_numpy()
    y = sample["total_fuel_sales"].to_numpy()
    trend = _linear_trend_coeffs(x, y)
    if trend is not None:
        k, b = trend
        xs = np.linspace(float(x.min()), float(x.max()), 80)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=k * xs + b,
                mode="lines",
                name="Линейный тренд",
                line=dict(color="#dc2626", width=2),
            )
        )
    fig.update_layout(**PLOTLY_LAYOUT, title="Зависимость продаж от цены АИ-92")
    return fig


def chart_promo_overlay(df: pd.DataFrame, metric: str) -> go.Figure:
    daily = daily_series(df, [metric], agg="sum")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily["timestamp"],
            y=daily[metric],
            mode="lines",
            name=METRIC_LABELS.get(metric, metric),
            line=dict(color="#2563eb", width=2),
        )
    )
    if "promotion_fuel_active" in df.columns:
        promo_days = (
            df.assign(day=df["timestamp"].dt.floor("D"))
            .groupby("day")["promotion_fuel_active"]
            .max()
            .reset_index()
        )
        promo_days.columns = ["timestamp", "promo"]
        active = promo_days[promo_days["promo"] > 0]["timestamp"]
        for ts in active:
            fig.add_vrect(
                x0=ts,
                x1=ts + pd.Timedelta(days=1),
                fillcolor="rgba(234, 88, 12, 0.15)",
                line_width=0,
            )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color="rgba(234, 88, 12, 0.5)", size=12),
                name="Дни с акцией на топливо",
            )
        )
    return _apply(fig, "Динамика продаж и периоды акций", METRIC_LABELS.get(metric, metric))


def chart_calendar_factors(df: pd.DataFrame, metric: str) -> go.Figure:
    frames = []
    if "is_weekend" in df.columns:
        w = df.groupby("is_weekend")[metric].mean().reset_index()
        w["фактор"] = w["is_weekend"].map({0: "Будни", 1: "Выходные"})
        w = w.rename(columns={metric: "среднее"})
        frames.append(w[["фактор", "среднее"]])
    if "is_holiday" in df.columns:
        h = df.groupby("is_holiday")[metric].mean().reset_index()
        h["фактор"] = h["is_holiday"].map({0: "Не праздник", 1: "Праздник"})
        h = h.rename(columns={metric: "среднее"})
        frames.append(h[["фактор", "среднее"]])
    if not frames:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT, title="Нет календарных признаков")
        return fig
    cal = pd.concat(frames, ignore_index=True)
    fig = px.bar(cal, x="фактор", y="среднее", color="фактор", text_auto=".1f")
    fig.update_layout(**PLOTLY_LAYOUT, title="Средние продажи: календарные факторы", showlegend=False)
    return fig


def chart_weather(df: pd.DataFrame, metric: str) -> go.Figure | None:
    if "weather_condition" not in df.columns:
        return None
    w = df.groupby("weather_condition")[metric].mean().reset_index().sort_values(metric, ascending=True)
    fig = px.bar(
        w,
        x=metric,
        y="weather_condition",
        orientation="h",
        labels={metric: METRIC_LABELS.get(metric, metric), "weather_condition": "Погода"},
    )
    fig.update_layout(**PLOTLY_LAYOUT, title="Продажи по типу погоды")
    return fig


def chart_network_comparison(df: pd.DataFrame, metric: str) -> go.Figure:
    if "station_name" in df.columns:
        g = df.groupby(["station_id", "station_name"])[metric].mean().reset_index()
        g["подпись"] = g["station_name"]
    else:
        g = df.groupby("station_id")[metric].mean().reset_index()
        g["подпись"] = g["station_id"].astype(str)
    g = g.sort_values(metric, ascending=True)
    fig = px.bar(
        g,
        x=metric,
        y="подпись",
        orientation="h",
        labels={metric: METRIC_LABELS.get(metric, metric), "подпись": "АЗС"},
    )
    fig.update_layout(**PLOTLY_LAYOUT, title="Сравнение АЗС (среднее за период)")
    return fig


def chart_dual_axis(df: pd.DataFrame, left: str, right: str) -> go.Figure:
    daily = daily_series(df, [left, right], agg="sum" if left == "total_fuel_sales" else "mean")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=daily["timestamp"], y=daily[left], name=METRIC_LABELS.get(left, left), line=dict(color="#2563eb")),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=daily["timestamp"], y=daily[right], name=METRIC_LABELS.get(right, right), line=dict(color="#dc2626")),
        secondary_y=True,
    )
    fig.update_layout(**PLOTLY_LAYOUT, title=f"{METRIC_LABELS.get(left, left)} и {METRIC_LABELS.get(right, right)}")
    fig.update_yaxes(title_text=METRIC_LABELS.get(left, left), secondary_y=False)
    fig.update_yaxes(title_text=METRIC_LABELS.get(right, right), secondary_y=True)
    fig.update_xaxes(title_text="Дата", rangeslider=dict(visible=True))
    return fig


def overview_kpi_row(df: pd.DataFrame) -> dict[str, str]:
    return {
        "stations": str(df["station_id"].nunique()),
        "hours": f"{len(df):,}",
        "fuel_sum": f"{df['total_fuel_sales'].sum():,.0f}",
        "shop_sum": f"{df['shop_total_revenue'].sum():,.0f}" if "shop_total_revenue" in df.columns else "—",
    }
