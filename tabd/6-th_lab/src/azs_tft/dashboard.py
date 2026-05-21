from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*The given NumPy array is not writable.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*LeafSpec.*")
warnings.filterwarnings("ignore", message=".*already saved during checkpointing.*", category=UserWarning)

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

from azs_tft.analytics import (  # noqa: E402
    list_runs_table,
    load_meta,
    promotion_effect_stats,
    read_training_metrics_csv,
    station_risk_table,
    validate_csv_schema,
)
from azs_tft.dashboard_charts import (  # noqa: E402
    METRIC_LABELS,
    chart_calendar_factors,
    chart_daily_multiline,
    chart_dual_axis,
    chart_fuel_structure_pie,
    chart_heatmap_dow_hour,
    chart_hourly_profile,
    chart_network_comparison,
    chart_price_vs_sales,
    chart_promo_overlay,
    chart_traffic_area,
    chart_weather,
    filter_period,
    overview_kpi_row,
)
from azs_tft.model_registry import (  # noqa: E402
    get_run,
    import_legacy_checkpoint,
    list_runs,
    resolve_active_paths,
    set_active,
)
from azs_tft.export_report import build_report_markdown  # noqa: E402
from azs_tft.forecast_viz import (  # noqa: E402
    evaluate_validation_metrics,
    plot_forecast_plotly,
    plot_fuel_grades_forecast_plotly,
    scenario_promo_compare,
    target_options_for_csv,
)
from azs_tft.dashboard_cache import (  # noqa: E402
    get_cached_panel,
    get_cached_station_map,
)
from azs_tft.dashboard_jobs import (  # noqa: E402
    clear_job,
    get_job,
    job_running,
    render_global_job_banner,
    render_job_result_for_page,
    submit_training_subprocess,
    try_submit_or_warn,
)
from azs_tft.targets_config import FUEL_GRADE_TARGETS  # noqa: E402
from azs_tft.paths import artifacts_dir, project_root  # noqa: E402
from azs_tft.predict import (  # noqa: E402
    forecast_steps_table,
    plot_prediction_example,
    variable_importance_tft,
)
from azs_tft.preprocess import load_data_dictionary, persist_uploaded_csv  # noqa: E402
from azs_tft.recommendations import recommend  # noqa: E402

NAV_PAGES: list[tuple[str, str]] = [
    ("dict", "Справочник"),
    ("network", "Сеть"),
    ("analytics", "Аналитика"),
    ("train", "Обучение"),
    ("model", "Модель"),
    ("fore", "Прогноз"),
    ("table", "Таблица"),
    ("scenario", "Сценарии"),
    ("interp", "Интерпретация"),
    ("rec", "Рекомендации"),
    ("export", "Экспорт"),
]
NAV_LABEL_TO_ID = {label: pid for pid, label in NAV_PAGES}

ANALYTICS_METRICS = [
    "total_fuel_sales",
    "shop_total_revenue",
    "total_traffic",
    "price_AI92",
    "competitor_price_AI92",
]


def _apply_theme() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; max-width: 1400px; }
        [data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.65rem 0.85rem;
        }
        [data-testid="stMetricLabel"] { font-size: 0.8rem; color: #475569; }
        [data-testid="stMetricValue"] { font-size: 1.35rem; color: #0f172a; }
        div[data-testid="stTabs"] button { font-weight: 500; }
        .stAlert { border-radius: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _sidebar_data_csv() -> Path | None:
    up = st.file_uploader("Загрузка CSV", type=["csv"], key="global_csv_upload")
    if up is not None:
        return persist_uploaded_csv(up.getvalue(), up.name)
    choice = st.radio(
        "Набор данных",
        ["5stations_data.csv", "detailed_data.csv"],
        horizontal=True,
        key="global_csv_radio",
        label_visibility="collapsed",
    )
    name = "5stations_data.csv" if choice.startswith("5") else "detailed_data.csv"
    p = project_root() / name
    return p if p.exists() else None


def _station_select(df: pd.DataFrame, key: str) -> int:
    if "station_name" in df.columns:
        m = df.groupby("station_id", as_index=False)["station_name"].first()
        options = {f"{r.station_name}": int(r.station_id) for _, r in m.iterrows()}
    else:
        options = {f"АЗС {int(s)}": int(s) for s in sorted(df["station_id"].unique())}
    label = st.selectbox("АЗС", list(options.keys()), key=key, label_visibility="visible")
    return options[label]


def _period_filter(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    tmin, tmax = df["timestamp"].min(), df["timestamp"].max()
    c1, c2 = st.columns(2)
    start = c1.date_input("Начало периода", tmin.date(), min_value=tmin.date(), max_value=tmax.date(), key=f"{key_prefix}_start")
    end = c2.date_input("Конец периода", tmax.date(), min_value=tmin.date(), max_value=tmax.date(), key=f"{key_prefix}_end")
    ts_start = pd.Timestamp(start)
    ts_end = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return filter_period(df, ts_start, ts_end)


def _plotly(fig, height: int | None = None) -> None:
    kw: dict = {"width": "stretch"}
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, **kw)


def main() -> None:
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())], check=True)


def _artifact_ok() -> bool:
    paths = resolve_active_paths()
    return paths["checkpoint"].exists() and paths["dataset_params"].exists()


def _page_network() -> None:
    if PANEL_DF is None:
        st.error("Файл данных не найден")
        return
    df = PANEL_DF
    kpi = overview_kpi_row(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("АЗС", kpi["stations"])
    c2.metric("Наблюдений", kpi["hours"])
    c3.metric("Топливо, л (сумма)", kpi["fuel_sum"])
    c4.metric("Магазин, руб (сумма)", kpi["shop_sum"])
    fuel_cols = [c for c in df.columns if c.startswith("sales_")]
    if fuel_cols:
        _plotly(chart_daily_multiline(df, fuel_cols, "Продажи по маркам топлива (сутки)"), 420)
        pie = chart_fuel_structure_pie(df)
        if pie:
            col_a, col_b = st.columns([2, 1])
            with col_a:
                _plotly(chart_daily_multiline(df, ["total_fuel_sales"], "Суммарные продажи топлива"), 380)
            with col_b:
                _plotly(pie, 380)
    tr = chart_traffic_area(df)
    if tr:
        _plotly(tr, 400)
    _plotly(chart_network_comparison(df, "total_fuel_sales"), 360)


@st.fragment
def _page_analytics() -> None:
    if PANEL_DF is None:
        st.error("Файл данных не найден")
        return
    df_all = PANEL_DF
    sid = _station_select(df_all, "an_sid")
    df_st = df_all[df_all["station_id"] == sid]
    st_name = (
        df_st["station_name"].iloc[0]
        if "station_name" in df_st.columns and len(df_st)
        else f"ID {sid}"
    )
    st.markdown(f"#### {st_name}")
    df = _period_filter(df_st, "an")
    metric = st.selectbox(
        "Основной показатель",
        ANALYTICS_METRICS,
        format_func=lambda m: METRIC_LABELS.get(m, m),
        key="an_metric",
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Среднее / ч", f"{df[metric].mean():,.1f}")
    m2.metric("Макс. / ч", f"{df[metric].max():,.1f}")
    m3.metric("Мин. / ч", f"{df[metric].min():,.1f}")
    m4.metric("Строк в периоде", f"{len(df):,}")
    sub_dyn, sub_traf, sub_cal, sub_cmp = st.tabs(
        ["Динамика", "Трафик и цены", "Календарь", "Сравнение с сетью"]
    )
    with sub_dyn:
        _plotly(chart_promo_overlay(df, metric), 440)
        extra = st.multiselect(
            "Дополнительные ряды (суточные)",
            [c for c in ANALYTICS_METRICS if c in df.columns and c != metric],
            format_func=lambda m: METRIC_LABELS.get(m, m),
            key="an_extra",
        )
        if extra:
            _plotly(chart_daily_multiline(df, [metric, *extra], "Совмещённые показатели", agg="sum"), 400)
        c1, c2 = st.columns(2)
        with c1:
            _plotly(chart_hourly_profile(df, metric), 360)
        with c2:
            _plotly(chart_heatmap_dow_hour(df, metric), 360)
    with sub_traf:
        if "total_traffic" in df.columns and metric != "total_traffic":
            _plotly(chart_dual_axis(df, metric, "total_traffic"), 420)
        tr = chart_traffic_area(df)
        if tr:
            _plotly(tr, 400)
        price_fig = chart_price_vs_sales(df)
        if price_fig:
            _plotly(price_fig, 400)
        if "competitor_price_AI92" in df.columns and "price_AI92" in df.columns:
            _plotly(
                chart_daily_multiline(
                    df, ["price_AI92", "competitor_price_AI92"], "Цены: сеть и конкурент", agg="mean"
                ),
                380,
            )
    with sub_cal:
        c1, c2 = st.columns(2)
        with c1:
            _plotly(chart_calendar_factors(df, metric), 360)
        w = chart_weather(df, metric)
        with c2:
            if w:
                _plotly(w, 360)
            elif "temperature" in df.columns:
                _plotly(
                    chart_daily_multiline(df, ["temperature"], "Температура", agg="mean"),
                    360,
                )
        if "promotion_fuel_active" in df.columns:
            promo = promotion_effect_stats(df, metric)
            if not promo.empty:
                _plotly(px.bar(promo, x="режим", y="среднее", title="Акция на топливо"), 320)
    with sub_cmp:
        net = chart_network_comparison(df_all, metric)
        st.plotly_chart(net, width="stretch")
        st.caption("Средние значения по всем АЗС за полный период наблюдений")


def _page_fore() -> None:
    render_job_result_for_page("fore", plotly_chart=_plotly)
    if not _artifact_ok():
        st.warning("Требуется обученная модель")
        return
    if PANEL_DF is None:
        st.error("Файл данных не найден")
        return
    df = PANEL_DF
    sid = _station_select(df, "fore_sid")
    tgt_opts = target_options_for_csv()
    fuel_in_model = [c for c in FUEL_GRADE_TARGETS if c in tgt_opts.values()]
    if len(tgt_opts) <= 2:
        st.info(
            "Активная модель обучена на 2 цели. Для прогноза по маркам топлива — "
            "повторное обучение в разделе «Обучение» (9 целей: сумма + 7 марок + магазин)."
        )
    target_lbl = st.selectbox("Показатель", list(tgt_opts.keys()), key="fore_tgt")
    hist_h = st.slider("Глубина истории, ч", 24, 168, 72)
    c1, c2 = st.columns(2)
    if c1.button("Один показатель", type="primary", key="fore_one_target", disabled=job_running()):

        def _one(report):
            return plot_forecast_plotly(
                DATA_PATH, sid, target_lbl, hist_h, tgt_opts, on_progress=report
            )

        try_submit_or_warn("Прогноз TFT", "fore", _one, result_type="plotly", plot_height=480)
    if c2.button("Все марки топлива", key="fore_all_fuels", disabled=job_running()):
        if fuel_in_model:

            def _fuels(report):
                return plot_fuel_grades_forecast_plotly(DATA_PATH, sid, hist_h, on_progress=report)

            try_submit_or_warn("Прогноз по маркам", "fore", _fuels, result_type="plotly", plot_height=900)
        else:
            st.warning("Марки топлива не входят в цели текущей модели.")
    with st.expander("Диагностический график (matplotlib)"):
        mapping = get_cached_station_map(DATA_PATH)
        idx = mapping.get(sid, 0)
        if st.button("Открыть", key="fore_mpl_diag", disabled=job_running()):

            def _mpl(report):
                return plot_prediction_example(DATA_PATH, sample_idx=idx, on_progress=report)

            try_submit_or_warn("Диагностический график", "fore", _mpl, result_type="mpl")


def _page_table() -> None:
    render_job_result_for_page("table", plotly_chart=_plotly)
    if not _artifact_ok():
        st.warning("Требуется обученная модель")
        return
    if PANEL_DF is None:
        return
    sid = _station_select(PANEL_DF, "tbl_sid")
    if st.button("Сформировать таблицу", type="primary", disabled=job_running()):
        idx = get_cached_station_map(DATA_PATH)[sid]

        def _tbl(report):
            return forecast_steps_table(DATA_PATH, sample_idx=idx, on_progress=report)

        try_submit_or_warn("Таблица прогноза", "table", _tbl, result_type="dataframe")


def _page_scenario() -> None:
    render_job_result_for_page("scenario", plotly_chart=_plotly)
    st.caption("Сравнение прогноза при изменении признака акции на горизонте decoder")
    if not _artifact_ok():
        st.warning("Требуется обученная модель")
        return
    if PANEL_DF is None:
        return
    sid = _station_select(PANEL_DF, "sc_sid")
    promo = st.toggle("Акция на топливо на горизонте прогноза", value=True)
    if st.button("Расчёт сценариев", type="primary", disabled=job_running()):

        def _scen(report):
            return scenario_promo_compare(DATA_PATH, sid, promo_on=promo, on_progress=report)

        try_submit_or_warn("Сценарии", "scenario", _scen, result_type="dataframe")


def _page_rec() -> None:
    render_job_result_for_page("rec", plotly_chart=_plotly)
    if PANEL_DF is None:
        st.error("Файл данных не найден")
        return
    df = PANEL_DF
    tail = df[df["timestamp"] > df["timestamp"].max() - pd.Timedelta(days=14)]
    base = float(tail["total_fuel_sales"].mean()) if len(tail) else 0.0
    st.markdown("##### Сигналы по последним 14 суткам")
    for r in recommend(tail, forecast_fuel=base, baseline_fuel=base):
        st.markdown(f"- {r}")
    promo = promotion_effect_stats(df)
    if not promo.empty:
        _plotly(px.bar(promo, x="режим", y="среднее", title="Влияние акции на топливо"), 320)
    if _artifact_ok() and st.button("Матрица риска: прогноз vs базис", disabled=job_running()):

        def _risk(report):
            n_st = len(get_cached_station_map(DATA_PATH))
            report(1, max(n_st, 1), f"Прогноз по {n_st} АЗС")
            risk_df = station_risk_table(DATA_PATH, top_n=15)
            report(max(n_st, 1), max(n_st, 1), "Готово")
            return risk_df

        try_submit_or_warn("Матрица риска", "rec", _risk, result_type="dataframe")


def _page_interp() -> None:
    if not _artifact_ok():
        st.warning("Требуется обученная модель")
        return
    if not DATA_PATH or not DATA_PATH.exists():
        return
    job = get_job()
    if job and job.get("page") == "interp" and job.get("status") == "done":
        res = job.get("result")
        if isinstance(res, dict):
            if res.get("empty"):
                st.info("Данные недоступны")
            else:
                if res.get("chart") is not None:
                    _plotly(res["chart"], 480)
                if res.get("table") is not None:
                    st.dataframe(res["table"], width="stretch", hide_index=True)
        if st.button("Сбросить результат", key="job_clear_interp"):
            clear_job()
            st.rerun()
    elif job and job.get("page") == "interp" and job.get("status") == "error":
        st.error(job.get("error"))
    if st.button("Важность признаков", type="primary", disabled=job_running()):

        def _imp(report):
            imp = variable_importance_tft(DATA_PATH, on_progress=report)
            if imp.empty:
                return {"empty": True}
            return {
                "empty": False,
                "chart": px.bar(
                    imp.head(20),
                    x="importance",
                    y="variable",
                    orientation="h",
                    title="Важность признаков encoder",
                ),
                "table": imp,
            }

        try_submit_or_warn("Важность признаков", "interp", _imp, result_type="generic")


if "registry_ready" not in st.session_state:
    import_legacy_checkpoint()
    st.session_state.registry_ready = True

st.set_page_config(
    page_title="АЗС — аналитика и прогноз",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded",
)
_apply_theme()

st.markdown("## Сеть АЗС — аналитика и прогнозирование")
st.caption("Temporal Fusion Transformer · продажи топлива и магазина")

with st.sidebar:
    st.markdown("### Модель прогноза")
    runs = list_runs()
    if not runs:
        st.info("Модель не обучена")
    else:
        labels, id_by_label = [], {}
        for r in runs:
            vl = r.get("val_loss_min")
            vl_s = f"{vl:.3f}" if vl is not None and vl == vl else "—"
            lbl = f"{r['run_id']} · {r.get('dataset_label', '?')} · val {vl_s}"
            if r.get("is_active"):
                lbl = "● " + lbl
            labels.append(lbl)
            id_by_label[lbl] = r["run_id"]
        pick = st.selectbox("Версия модели", labels, key="active_model_pick")
        if st.button("Применить", width="stretch"):
            set_active(id_by_label[pick])
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        active = resolve_active_paths()
        st.caption(f"Активная: {active.get('run_id') or 'legacy'}")

    st.divider()
    st.markdown("### Источник данных")
    DATA_PATH = _sidebar_data_csv()

    st.divider()
    st.markdown("### Раздел")
    nav_label = st.radio(
        "Раздел",
        [label for _, label in NAV_PAGES],
        key="main_nav",
        label_visibility="collapsed",
    )
    PAGE_ID = NAV_LABEL_TO_ID[nav_label]

PANEL_DF = get_cached_panel(DATA_PATH)

if _artifact_ok():
    meta = load_meta()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Статус модели", "Готова")
    c2.metric("Обучающий набор", meta.get("dataset_label") or Path(str(meta.get("csv_path", ""))).name)
    c3.metric("Мин. val_loss", f"{meta.get('val_loss_min', '—')}")
    c4.metric("Горизонт прогноза", f"{meta.get('max_prediction_length', 24)} ч")
else:
    st.warning("Модель прогноза недоступна — требуется обучение в разделе «Обучение».")

render_global_job_banner()

# —— Справочник ——
if PAGE_ID == "dict":
    dd = load_data_dictionary()
    st.markdown("#### Переменные и роли в модели")
    for s in dd.get("sources", []):
        st.markdown(f"**{s['file']}** — {s['role']}")
    st.dataframe(
        pd.DataFrame(dd.get("variables", []))[["name", "type", "description"]]
        if dd.get("variables")
        else pd.DataFrame(),
        width="stretch",
        hide_index=True,
    )
    with st.expander("Конфигурация признаков TFT"):
        st.json(dd.get("tft_roles", {}))

elif PAGE_ID == "network":
    _page_network()

elif PAGE_ID == "analytics":
    _page_analytics()

elif PAGE_ID == "train":
    # —— Обучение ——
    st.markdown("#### Обучение модели TFT")
    up = st.file_uploader("CSV для обучения", type=["csv"], key="train_csv")
    use_builtin = st.radio(
        "Набор по умолчанию",
        ["5stations_data.csv", "detailed_data.csv"],
        horizontal=True,
        key="train_builtin",
    )
    csv_check: Path | None = None
    if up is not None:
        csv_check = persist_uploaded_csv(up.getvalue(), up.name)
    else:
        name = "5stations_data.csv" if use_builtin.startswith("5") else "detailed_data.csv"
        p = project_root() / name
        csv_check = p if p.exists() else None

    if csv_check:
        st.dataframe(validate_csv_schema(csv_check), width="stretch", hide_index=True)

    c1, c2, c3, c4 = st.columns(4)
    epochs = c1.number_input("Эпохи", 1, 500, 15)
    batch = c2.number_input("Размер батча", 8, 512, 64)
    lim = c3.number_input("Лимит батчей (0 — без лимита)", 0, value=0)
    nw = c4.number_input("Потоки загрузки", 0, 16, min(4, max(0, (os.cpu_count() or 4) - 1)))

    render_job_result_for_page("train", plotly_chart=_plotly)

    if st.button("Запуск обучения", type="primary", disabled=job_running()):
        if csv_check is None:
            st.error("Не указан файл данных")
        else:
            cmd = [
                sys.executable,
                "-m",
                "azs_tft.train",
                "--csv",
                str(csv_check),
                "--epochs",
                str(int(epochs)),
                "--batch-size",
                str(int(batch)),
                "--num-workers",
                str(int(nw)),
            ]
            if int(lim) > 0:
                cmd.extend(["--limit-train-batches", str(int(lim))])
            if submit_training_subprocess(
                "Обучение TFT",
                "train",
                cmd,
                cwd=str(ROOT),
                env={**os.environ, "PYTHONPATH": str(SRC)},
            ):
                st.toast("Обучение запущено в фоне", icon="⏳")
                st.rerun()
            else:
                st.warning("Дождитесь завершения текущей фоновой задачи.")

elif PAGE_ID == "model":
    # —— Модель ——
    st.markdown("#### Реестр версий модели")
    runs_df = list_runs_table()
    if runs_df.empty:
        st.info("Записей в реестре нет")
    else:
        st.dataframe(runs_df, width="stretch", hide_index=True)
        sel = st.selectbox("Параметры запуска", runs_df["run_id"].tolist(), key="model_detail_id")
        detail = get_run(sel)
        if detail:
            with st.expander("Метаданные запуска"):
                st.json(detail)
            metrics = read_training_metrics_csv(sel)
            if metrics is not None:
                cols = [c for c in ("epoch", "train_loss", "train_loss_epoch", "val_loss") if c in metrics.columns]
                if cols:
                    st.markdown("##### Кривые потерь")
                    st.line_chart(metrics[cols].dropna(how="all"), width="stretch")
        if st.button("Активировать выбранную версию", key="model_activate_btn"):
            set_active(sel)
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    if _artifact_ok() and DATA_PATH and DATA_PATH.exists():
        st.divider()
        st.markdown("##### Метрики на валидации")
        st.caption("MAE и R² в нормализованном масштабе. Для прогноза по маркам — модель с 9 целями (переобучение).")
        render_job_result_for_page("model", plotly_chart=_plotly)
        if st.button("Расчёт MAE и R²", disabled=job_running()):

            def _metrics(report):
                return evaluate_validation_metrics(
                    DATA_PATH, max_batches=50, on_progress=report
                )

            try_submit_or_warn("Метрики MAE и R²", "model", _metrics, result_type="dataframe")

elif PAGE_ID == "fore":
    _page_fore()

elif PAGE_ID == "table":
    _page_table()

elif PAGE_ID == "scenario":
    _page_scenario()

elif PAGE_ID == "interp":
    _page_interp()

elif PAGE_ID == "rec":
    _page_rec()

elif PAGE_ID == "export":
    if DATA_PATH and DATA_PATH.exists():
        render_job_result_for_page("export", plotly_chart=_plotly)
        if st.button("Собрать отчёт", disabled=job_running()):

            def _md(_report):
                _report(1, 1, "Формирование markdown")
                return build_report_markdown(DATA_PATH)

            try_submit_or_warn("Экспорт отчёта", "export", _md, result_type="markdown")
