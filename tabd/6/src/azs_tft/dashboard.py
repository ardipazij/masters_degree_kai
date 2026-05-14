from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from azs_tft.paths import artifacts_dir, project_root  # noqa: E402
from azs_tft.preprocess import load_data_dictionary, load_panel, persist_uploaded_csv  # noqa: E402
from azs_tft.predict import (  # noqa: E402
    forecast_steps_table,
    plot_prediction_example,
    variable_importance_tft,
)
from azs_tft.recommendations import recommend  # noqa: E402


def _resolve_csv_path(upload_key: str, radio_key: str) -> Path | None:
    """Сначала загрузка своего CSV, иначе — встроенный файл из корня проекта."""
    up = st.file_uploader(
        "Свой CSV (те же колонки, что в `detailed_data.csv`)",
        type=["csv"],
        key=upload_key,
    )
    if up is not None:
        return persist_uploaded_csv(up.getvalue(), up.name)
    choice = st.radio(
        "Встроенный набор",
        ["5stations_data.csv", "detailed_data.csv"],
        horizontal=True,
        key=radio_key,
    )
    root = project_root()
    name = "5stations_data.csv" if choice.startswith("5") else "detailed_data.csv"
    p = root / name
    return p if p.exists() else None


def main() -> None:
    app = Path(__file__).resolve()
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app)], check=True)


def _artifact_ok() -> bool:
    a = artifacts_dir()
    return (a / "tft_azs.ckpt").exists() and (a / "tft_dataset_params.pt").exists()


st.set_page_config(page_title="Татнефть АЗС — TFT", layout="wide")
st.title("Татнефть АЗС — TFT аналитика")

(
    tab_meta,
    tab_overview,
    tab_station,
    tab_train,
    tab_fore,
    tab_table,
    tab_rec,
) = st.tabs(
    [
        "Описание данных",
        "Обзор сети",
        "Анализ АЗС",
        "Данные и обучение",
        "Прогноз TFT",
        "Прогноз — таблица",
        "Рекомендации",
    ]
)

with tab_meta:
    dd = load_data_dictionary()
    st.subheader("Источники и переменные")
    for s in dd.get("sources", []):
        st.markdown(f"- **{s['file']}**: {s['role']}")
    for v in dd.get("variables", []):
        st.markdown(f"- `{v['name']}` — *{v['type']}*: {v['description']}")
    with st.expander("Роли признаков в TFT (json)"):
        st.json(dd.get("tft_roles", {}))
    with st.expander("О проекте / стек"):
        st.markdown(
            "- **Модель:** Temporal Fusion Transformer (`pytorch-forecasting`), цели: суммарное топливо и выручка магазина.\n"
            "- **Обучение:** `python -m azs_tft.train` (или кнопка на вкладке «Данные и обучение»).\n"
            "- **Интерфейс:** Streamlit + Plotly.\n"
            "- Прогноз по шагам — из сырого выхода `predict`; при смене версии PF форма тензоров может отличаться."
        )

with tab_overview:
    st.subheader("Сводка по сети")
    p = _resolve_csv_path("ov_upload", "ov_builtin")
    if p is None or not p.exists():
        st.error("Нет CSV для обзора.")
    else:
        df = load_panel(p)
        tmin, tmax = df["timestamp"].min(), df["timestamp"].max()
        c1, c2, c3 = st.columns(3)
        c1.metric("АЗС", df["station_id"].nunique())
        c2.metric("Строк (часов)", f"{len(df):,}")
        c3.metric("Период", f"{tmin.date()} — {tmax.date()}")
        fuel_cols = [c for c in df.columns if c.startswith("sales_") and c not in ("sales_total",)]
        if fuel_cols:
            daily = (
                df.set_index("timestamp")
                .groupby([pd.Grouper(freq="1D")])[fuel_cols]
                .sum()
                .reset_index()
            )
            dm = daily.melt(id_vars=["timestamp"], var_name="марка", value_name="литры")
            st.plotly_chart(
                px.line(dm, x="timestamp", y="литры", color="марка", title="Динамика продаж по маркам (сумма по сети, л/сутки)"),
                width="stretch",
            )
            struct = df[fuel_cols].sum()
            st.plotly_chart(
                px.pie(values=struct.values, names=struct.index, title="Структура продаж (средняя по всему ряду)"),
                width="stretch",
            )
        tr_cols = [c for c in df.columns if c.startswith("traffic_") and c != "total_traffic"]
        if tr_cols and "timestamp" in df.columns:
            td = (
                df.set_index("timestamp")
                .groupby(pd.Grouper(freq="1D"))[tr_cols]
                .mean()
                .reset_index()
            )
            tm = td.melt(id_vars=["timestamp"], var_name="тип", value_name="инт.")
            st.plotly_chart(
                px.area(tm, x="timestamp", y="инт.", color="тип", title="Средний трафик по типам (по дням)"),
                width="stretch",
            )
        shop_cols = [c for c in ("shop_напитки", "shop_закуски", "shop_автотовары", "shop_кофе", "shop_табак") if c in df.columns]
        if shop_cols:
            sd = (
                df.set_index("timestamp")
                .groupby(pd.Grouper(freq="1D"))[shop_cols]
                .sum()
                .reset_index()
            )
            sm = sd.melt(id_vars=["timestamp"], var_name="категория", value_name="руб")
            st.plotly_chart(
                px.bar(sm, x="timestamp", y="руб", color="категория", title="Магазин: сумма по категориям (по дням)"),
                width="stretch",
            )

with tab_station:
    st.subheader("Анализ одной АЗС")
    p = _resolve_csv_path("st_upload", "st_builtin")
    if p is None or not p.exists():
        st.error("Нет CSV.")
    else:
        df = load_panel(p)
        sid = st.selectbox("АЗС", sorted(df["station_id"].unique()), key="st_sid")
        d = df[df["station_id"] == sid].copy()
        daily = (
            d.set_index("timestamp")
            .groupby(pd.Grouper(freq="1D"))[
                [c for c in ["sales_AI92", "sales_AI95", "sales_AI98", "sales_DT_EURO", "sales_DT_TANEKO", "sales_DT_SUMMER", "sales_DT_WINTER"] if c in d.columns]
            ]
            .sum()
            .reset_index()
        )
        if not daily.empty and len(daily.columns) > 1:
            dm = daily.melt(id_vars=["timestamp"], var_name="марка", value_name="л")
            st.plotly_chart(
                px.line(dm, x="timestamp", y="л", color="марка", title=f"Ежедневные продажи по маркам — station_id={sid}"),
                width="stretch",
            )
        if "hour" in d.columns and "total_fuel_sales" in d.columns:
            hp = d.groupby("hour")["total_fuel_sales"].mean().reset_index()
            st.plotly_chart(px.bar(hp, x="hour", y="total_fuel_sales", title="Суточный паттерн (ср. топливо/час)"), width="stretch")
        if "day_of_week" in d.columns:
            dp = d.groupby("day_of_week")["total_fuel_sales"].mean().reset_index()
            st.plotly_chart(px.bar(dp, x="day_of_week", y="total_fuel_sales", title="Недельный паттерн"), width="stretch")

with tab_train:
    st.subheader("Загрузка CSV и обучение TFT")
    st.markdown(
        "Файл должен совпадать по колонкам с `detailed_data.csv`. Обучение запускается **в отдельном процессе** (может занять минуты)."
    )
    up = st.file_uploader("CSV для обучения", type=["csv"], key="train_csv")
    use_builtin = st.radio(
        "Если файл не загружен — использовать из корня проекта",
        ["5stations_data.csv", "detailed_data.csv"],
        horizontal=True,
        key="train_builtin",
    )
    epochs = st.number_input("Эпохи", min_value=1, max_value=500, value=15)
    batch = st.number_input("Размер батча", min_value=8, max_value=512, value=64)
    lim = st.number_input("Лимит train-батчей (0 = без лимита)", min_value=0, value=0)
    nw = st.number_input("num_workers DataLoader", min_value=0, max_value=16, value=min(4, max(0, (os.cpu_count() or 4) - 1)))
    if st.button("Запустить обучение", type="primary"):
        if up is not None:
            csv_p = persist_uploaded_csv(up.getvalue(), up.name)
            cmd = [
                sys.executable,
                "-m",
                "azs_tft.train",
                "--csv",
                str(csv_p),
                "--epochs",
                str(int(epochs)),
                "--batch-size",
                str(int(batch)),
                "--num-workers",
                str(int(nw)),
            ]
        else:
            root = project_root()
            name = "5stations_data.csv" if use_builtin.startswith("5") else "detailed_data.csv"
            csv_p = root / name
            if not csv_p.exists():
                st.error(f"Нет файла {csv_p}")
                csv_p = None
            if csv_p is not None:
                cmd = [
                    sys.executable,
                    "-m",
                    "azs_tft.train",
                    "--csv",
                    str(csv_p),
                    "--epochs",
                    str(int(epochs)),
                    "--batch-size",
                    str(int(batch)),
                    "--num-workers",
                    str(int(nw)),
                ]
        if up is not None or (up is None and csv_p is not None):
            if int(lim) > 0:
                cmd.extend(["--limit-train-batches", str(int(lim))])
            env = {**os.environ, "PYTHONPATH": str(SRC)}
            with st.spinner("Идёт обучение…"):
                r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=86400)
            st.subheader("Вывод процесса")
            st.code((r.stdout or "") + "\n" + (r.stderr or ""), language="text")
            if r.returncode == 0:
                st.success("Обучение завершено. Чекпоинт: `artifacts/tft_azs.ckpt`")
            else:
                st.error(f"Код выхода: {r.returncode}")

with tab_fore:
    st.caption("Артефакты: `artifacts/tft_azs.ckpt`, `tft_dataset_params.pt`.")
    if not _artifact_ok():
        st.warning("Сначала обучите модель (вкладка «Данные и обучение»).")
    else:
        meta_path = artifacts_dir() / "tft_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            st.json(meta)
            st.caption(
                f"Горизонт TFT: **{meta.get('max_prediction_length', '?')}** ч. "
                "Для прогноза используйте CSV той же схемы, что при обучении."
            )
        else:
            st.caption("Нет `tft_meta.json` — горизонт по умолчанию задаётся в коде обучения (24 ч).")
        p = _resolve_csv_path("fore_upload", "fore_builtin")
        if p is None or not p.exists():
            st.error("Нет CSV.")
        else:
            idx = st.number_input("Индекс ряда в первом батче валидации", 0, 9999, 0, key="fore_idx")
            if st.button("График прогноза (matplotlib)"):
                try:
                    fig = plot_prediction_example(p, sample_idx=int(idx))
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.error(f"{type(e).__name__}: {e}")

with tab_table:
    st.subheader("Численный прогноз по шагам")
    if not _artifact_ok():
        st.warning("Нужен обученный чекпоинт.")
    else:
        p = _resolve_csv_path("tbl_upload", "tbl_builtin")
        if p is None or not p.exists():
            st.error("Нет CSV.")
        else:
            idx = st.number_input("Индекс ряда в первом батче", 0, 9999, 0, key="tbl_idx")
            if st.button("Сформировать таблицу"):
                try:
                    tbl = forecast_steps_table(p, sample_idx=int(idx))
                    st.dataframe(tbl, width="stretch", hide_index=True)
                except Exception as e:
                    st.error(f"{type(e).__name__}: {e}")

with tab_rec:
    p = _resolve_csv_path("rec_upload", "rec_builtin")
    if p is None or not p.exists():
        st.error("Нет CSV.")
    else:
        df = load_panel(p)
        tail = df[df["timestamp"] > df["timestamp"].max() - pd.Timedelta(days=14)]
        base = float(tail["total_fuel_sales"].mean()) if len(tail) else 0.0
        st.write("Базис топлива (14 дней):", round(base, 2))
        for r in recommend(tail, forecast_fuel=base, baseline_fuel=base):
            st.markdown(f"- {r}")
        st.subheader("Важность признаков")
        try:
            imp = variable_importance_tft(p)
            st.dataframe(imp, width="stretch")
        except Exception as e:
            st.info(str(e))
