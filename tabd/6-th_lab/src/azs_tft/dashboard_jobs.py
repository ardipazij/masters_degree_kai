"""Фоновые задачи дашборда: не блокируют UI и переключение разделов."""

from __future__ import annotations

import subprocess
import threading
from typing import Any, Callable

import streamlit as st

from azs_tft.dashboard_progress import ProgressReporter

JOB_KEY = "bg_job"

JobFn = Callable[[ProgressReporter], Any]


def get_job() -> dict[str, Any] | None:
    return st.session_state.get(JOB_KEY)


def job_running() -> bool:
    j = get_job()
    return j is not None and j.get("status") == "running"


def clear_job() -> None:
    st.session_state.pop(JOB_KEY, None)


def submit_job(
    title: str,
    page: str,
    fn: JobFn,
    *,
    result_type: str = "generic",
    plot_height: int | None = None,
) -> bool:
    """Запуск в daemon-потоке. False, если уже есть активная задача."""
    if job_running():
        return False
    job: dict[str, Any] = {
        "title": title,
        "page": page,
        "status": "running",
        "message": "Запуск…",
        "progress": 0.0,
        "log_lines": [],
        "result": None,
        "error": None,
        "result_type": result_type,
        "plot_height": plot_height,
    }
    st.session_state[JOB_KEY] = job

    def report(step: int, total: int, message: str) -> None:
        total = max(int(total), 1)
        step = max(0, min(int(step), total))
        job["progress"] = step / total
        job["message"] = message

    def worker() -> None:
        try:
            job["result"] = fn(report)
            job["status"] = "done"
            job["message"] = "Готово"
            job["progress"] = 1.0
        except Exception as exc:
            job["status"] = "error"
            job["error"] = f"{type(exc).__name__}: {exc}"
            job["message"] = str(exc)
            job["progress"] = 1.0

    threading.Thread(target=worker, daemon=True).start()
    return True


def submit_training_subprocess(
    title: str,
    page: str,
    cmd: list[str],
    *,
    cwd: str,
    env: dict[str, str],
) -> bool:
    if job_running():
        return False
    job: dict[str, Any] = {
        "title": title,
        "page": page,
        "status": "running",
        "message": "Обучение…",
        "progress": 0.05,
        "log_lines": [],
        "result": None,
        "error": None,
        "result_type": "train",
    }
    st.session_state[JOB_KEY] = job

    def worker() -> None:
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            lines: list[str] = []
            if proc.stdout:
                for line in proc.stdout:
                    lines.append(line.rstrip())
                    job["log_lines"] = lines[-40:]
                    if "Epoch" in line:
                        job["message"] = line.strip()[:100]
            rc = proc.wait(timeout=86400)
            job["result"] = {"returncode": rc, "log": "\n".join(lines)}
            if rc == 0:
                job["status"] = "done"
                job["message"] = "Обучение завершено"
                job["progress"] = 1.0
            else:
                job["status"] = "error"
                job["error"] = f"Код выхода {rc}"
                job["message"] = f"Ошибка обучения (код {rc})"
        except Exception as exc:
            job["status"] = "error"
            job["error"] = f"{type(exc).__name__}: {exc}"
            job["message"] = str(exc)

    threading.Thread(target=worker, daemon=True).start()
    return True


def render_global_job_banner() -> None:
    """Панель статуса вверху главной области (лёгкая, без блокировки)."""
    job = get_job()
    if not job:
        return
    status = job.get("status")
    title = job.get("title", "Задача")
    if status == "running":
        st.info(f"Выполняется в фоне: **{title}** — {job.get('message', '')}. Можно переключать разделы.")
        st.progress(float(job.get("progress", 0.0)), text=job.get("message", ""))
        if job.get("log_lines"):
            st.code("\n".join(job["log_lines"][-25:]))
        if st.button("Обновить статус", key="job_refresh_status"):
            st.rerun()
        return
    if status == "error":
        st.error(f"{title}: {job.get('error') or job.get('message')}")
        if job.get("log_lines"):
            st.code("\n".join(job["log_lines"][-25:]))
        if st.button("Закрыть", key="job_dismiss_err"):
            clear_job()
            st.rerun()
        return
    if status == "done":
        page = job.get("page", "")
        st.success(f"{title} — завершено. Откройте раздел «{page}» для просмотра результата.")
        if st.button("Закрыть уведомление", key="job_dismiss_ok"):
            clear_job()
            st.rerun()


def render_job_result_for_page(
    page_id: str,
    *,
    plotly_chart: Callable[[Any, int | None], None],
) -> None:
    """Показ результата на странице, где задача была запущена."""
    job = get_job()
    if not job or job.get("page") != page_id:
        return
    if job.get("status") == "error":
        st.error(job.get("error") or job.get("message"))
        if st.button("Сбросить", key=f"job_clear_err_{page_id}"):
            clear_job()
            st.rerun()
        return
    if job.get("status") != "done":
        return
    res = job.get("result")
    rt = job.get("result_type")
    if rt == "plotly" and res is not None:
        plotly_chart(res, job.get("plot_height"))
    elif rt == "dataframe" and res is not None:
        import pandas as pd

        st.dataframe(res, width="stretch", hide_index=True)
    elif rt == "mpl" and res is not None:
        import matplotlib.pyplot as plt

        st.pyplot(res)
        plt.close(res)
    elif rt == "markdown" and res:
        st.download_button(
            "Скачать отчёт (.md)",
            str(res),
            file_name="azs_tft_report.md",
            mime="text/markdown",
        )
        with st.expander("Предпросмотр"):
            st.markdown(str(res))
    elif rt == "train" and isinstance(res, dict):
        with st.expander("Журнал обучения", expanded=True):
            st.code(res.get("log") or "")
        if res.get("returncode") == 0:
            from azs_tft.analytics import list_runs_table
            from azs_tft.model_registry import import_legacy_checkpoint

            import_legacy_checkpoint()
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Модель сохранена в реестр")
            st.dataframe(list_runs_table(), width="stretch", hide_index=True)
    if st.button("Сбросить результат", key=f"job_clear_done_{page_id}"):
        clear_job()
        st.rerun()


def try_submit_or_warn(
    title: str,
    page: str,
    fn: JobFn,
    *,
    result_type: str = "generic",
    plot_height: int | None = None,
) -> None:
    if submit_job(title, page, fn, result_type=result_type, plot_height=plot_height):
        st.toast(f"Запущено: {title}", icon="⏳")
        st.rerun()
    else:
        busy = get_job()
        st.warning(
            f"Уже выполняется: **{busy.get('title') if busy else '?'}**. "
            "Дождитесь завершения или нажмите «Обновить статус» в панели выше."
        )
