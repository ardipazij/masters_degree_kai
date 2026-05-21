"""Индикаторы прогресса Streamlit для долгих операций TFT."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Generator

import streamlit as st

# step: номер шага (1..total), total: всего шагов, message: подпись
ProgressReporter = Callable[[int, int, str], None]


@contextmanager
def progress_task(title: str) -> Generator[ProgressReporter, None, None]:
    """
    Статус + progress bar. Вызывайте report(step, total, message) после каждого этапа.
    step=total означает 100%.
    """
    with st.status(title, expanded=True) as status:
        bar = st.progress(0.0, text="Запуск…")

        def report(step: int, total: int, message: str) -> None:
            total = max(int(total), 1)
            step = max(0, min(int(step), total))
            frac = step / total
            bar.progress(frac, text=message)
            status.update(label=f"{title} — {message}")

        try:
            yield report
            bar.progress(1.0, text="Готово")
            status.update(label=f"{title} — готово", state="complete")
        except Exception:
            bar.progress(1.0, text="Ошибка")
            status.update(label=f"{title} — ошибка", state="error")
            raise


def noop_progress(_step: int, _total: int, _message: str) -> None:
    pass
