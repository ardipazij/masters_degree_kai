# Руководство разработчика — azs-tft-dashboard

Техническая документация для сопровождения, расширения и отладки пайплайна TFT + Streamlit.

**См. также:** [README.md](README.md) — руководство пользователя.

---

## Содержание

1. [Архитектура](#архитектура)
2. [Стек и версии](#стек-и-версии)
3. [Среда разработки](#среда-разработки)
4. [Слои приложения](#слои-приложения)
5. [Поток данных](#поток-данных)
6. [Модуль `preprocess`](#модуль-preprocess)
7. [Модуль `train`](#модуль-train)
8. [Модуль `targets_config`](#модуль-targets_config)
9. [Модуль `model_registry`](#модуль-model_registry)
10. [Модуль `predict`](#модуль-predict)
11. [Модуль `forecast_viz`](#модуль-forecast_viz)
12. [Интеграция pytorch-forecasting 1.7](#интеграция-pytorch-forecasting-17)
13. [Модуль `dashboard_cache`](#модуль-dashboard_cache)
14. [Модуль `dashboard_jobs`](#модуль-dashboard_jobs)
15. [Дашборд и UI-слой](#дашборд-и-ui-слой)
16. [Контракты и инварианты](#контракты-и-инварианты)
17. [Расширение функциональности](#расширение-функциональности)
18. [Тестирование и отладка](#тестирование-и-отладка)
19. [Производительность](#производительность)
20. [Соглашения по коду](#соглашения-по-коду)
21. [Чеклист перед PR / сдачей](#чеклист-перед-pr--сдачей)

---

## Архитектура

```
┌──────────────────────────────────────────────────────────────────────────┐
│  dashboard.py — sidebar (модель, CSV, PAGE_ID), одна страница за rerun   │
│  dashboard_jobs (фон) │ dashboard_cache (@st.cache_data) │ charts       │
└────────────┬─────────────────────────────┬───────────────────────────────┘
             │                             │
             ▼                             ▼
┌────────────────────────┐       ┌──────────────────────────────┐
│   forecast_viz.py      │       │   train.py (CLI / subprocess) │
│   Plotly, MAE/R²       │       │   Lightning Trainer + TFT     │
└────────────┬───────────┘       └──────────────┬───────────────┘
             │                                   │
             ▼                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  predict.py — load_model (in-process cache), build_validation_dataloader │
│  model.predict(loader) → Prediction.output → extract_prediction_array    │
└────────────┬───────────────────────────────┬───────────────────────────────┘
             │                               │
             ▼                               ▼
┌────────────────────────┐       ┌──────────────────────────────┐
│   preprocess.py        │       │   model_registry + artifacts │
│   load_panel, split    │       │   .ckpt, dataset_params.pt   │
└────────────────────────┘       └──────────────────────────────┘
```

**Принципы:**

- **Один пакет** `azs_tft` в `src/` (setuptools `where = ["src"]`).
- **Разделение:** `train` и `predict` не импортируют Streamlit; `dashboard_charts` не импортирует `torch`.
- **Активная модель** — `model_registry.resolve_active_paths()`.
- **Тяжёлый UI** — только через `dashboard_jobs` (thread) или subprocess (обучение), не блокировать rerun.
- **Ленивая навигация** — рендерится только `PAGE_ID`, не все 11 экранов (раньше `st.tabs` выполнял всё).
- **Legacy:** `import_legacy_checkpoint()` один раз за сессию (`st.session_state.registry_ready`).

---

## Стек и версии

| Компонент | Пакет | Роль |
|-----------|--------|------|
| ML | `torch>=2.1` | Тензоры, CUDA |
| Обучение | `lightning>=2.1` | `Trainer`, callbacks, чекпоинты |
| TFT | `pytorch-forecasting>=1.0` | Фактически **1.7.x** — API `Prediction(output, x)` |
| Данные | `pandas`, `numpy` | ETL |
| UI | `streamlit>=1.28` | Дашборд (`@st.fragment` на Аналитике; `st.cache_data` в cache-модуле) |
| Визуализация | `plotly`, `matplotlib` | Интерактив + PF `plot_prediction` |

`requires-python = ">=3.11,<3.14"` — CI/локально целиться на **3.13**.

**Lightning 2:** в `train.py` только `train_dataloaders` / `val_dataloaders` (не `train_dataloader`).

---

## Среда разработки

### Клонирование и editable install

```bash
cd 6-th_lab
python3.13 -m venv .venv
source .venv/bin/activate
uv pip install -e . --python .venv/bin/python
export PYTHONPATH=src   # если не ставите -e
```

### Запуск без Streamlit

```bash
PYTHONPATH=src python -m azs_tft.train --csv 5stations_data.csv --epochs 2 --limit-train-batches 3
PYTHONPATH=src python -c "
from pathlib import Path
from azs_tft.predict import forecast_steps_table
print(forecast_steps_table(Path('5stations_data.csv'), 0).head())
"
```

### Запуск дашборда с hot-reload

```bash
streamlit run src/azs_tft/dashboard.py --server.runOnSave true
```

`dashboard.py` при `python -m azs_tft.dashboard` вызывает subprocess streamlit — для разработки предпочтительна команда выше.

### Entry points (`pyproject.toml`)

```toml
[project.scripts]
azs-train-tft = "azs_tft.train:main"
azs-dashboard = "azs_tft.dashboard:main"
```

### Что в `.gitignore`

- `artifacts/`, `lightning_logs/`, `*.ckpt`, `.venv/`, `.env`
- Не коммитить чекпоинты и загруженные пользователем CSV из `/tmp`.

---

## Слои приложения

| Модуль | Зависимости | Назначение |
|--------|-------------|------------|
| `paths.py` | — | `project_root()`, `artifacts_dir()` |
| `preprocess.py` | paths | CSV → panel, `time_idx`, split |
| `targets_config.py` | — | Список targets, подписи UI |
| `train.py` | preprocess, registry | `build_datasets`, `train()` |
| `model_registry.py` | paths | Версии моделей, active |
| `predict.py` | registry, preprocess | Inference, таблицы, plot PF |
| `forecast_viz.py` | predict | Plotly, метрики, сценарии |
| `analytics.py` | registry, predict | CSV validate, risk table |
| `dashboard_charts.py` | — | Чистые Plotly-функции (без ST) |
| `dashboard_cache.py` | streamlit, preprocess | `@st.cache_data`: panel, targets, station map |
| `dashboard_jobs.py` | streamlit, threading | Фоновые задачи, `bg_job` в session_state |
| `dashboard_progress.py` | streamlit | `ProgressReporter` (callback для jobs) |
| `dashboard.py` | всё выше | Sidebar, `PAGE_ID`, функции `_page_*` |
| `recommendations.py` | — | Эвристики текста |
| `export_report.py` | analytics | Markdown-отчёт |
| `data_dictionary.json` | — | Package data, справочник |

**Правило зависимостей:** `dashboard_charts` и `targets_config` не импортируют `torch`. `train` не импортирует `streamlit`.

---

## Поток данных

### 1. Загрузка CSV

`load_panel(path)`:

- парсит `timestamp`;
- заполняет категории `none` для строковых полей;
- сортирует `station_id`, `timestamp`;
- добавляет **`time_idx`** — счётчик внутри группы АЗС (обязателен для `TimeSeriesDataSet`);
- `_ensure_writable_numeric_arrays` — обход `UserWarning` read-only numpy в PF encoders.

### 2. Train / validation split

`time_split(df, val_hours=336)` — **глобальный** cutoff по `timestamp.max()`:

- train: `timestamp <= cutoff`
- val: `timestamp > cutoff`

Не per-station. При изменении логики split переобучите все модели.

### 3. TimeSeriesDataSet

`train.build_datasets(train_df, val_df)`:

- создаёт `training` на train;
- `validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=True)`.

Параметры сохраняются: `training.get_parameters()` → `tft_dataset_params.pt`.

### 4. Inference

`TimeSeriesDataSet.from_parameters(params, val_df, predict=True, stop_randomization=True)`:

- те же энкодеры/нормализаторы, что при обучении;
- `val_df` должен содержать достаточно истории для encoder length.

### 5. Predict output

`model.predict(loader, mode="raw", return_x=True)` → объект **`Prediction`**:

- `.output` — forward dict / `OutputMixIn` (ключ `prediction`);
- `.x` — dict входов (`decoder_target`, `encoder_target`, …).

Далее `extract_prediction_array` → `[batch, time, n_targets]`.

---

## Модуль `preprocess`

### Публичный API

```python
load_panel(csv_path: Path | str) -> pd.DataFrame
time_split(df, val_hours: int = 336) -> tuple[pd.DataFrame, pd.DataFrame]
default_csv(which: str = "5stations") -> Path
persist_uploaded_csv(data: bytes | BinaryIO, name: str) -> Path
load_data_dictionary() -> dict
```

### `project_root()`

Ищет вверх от `cwd` и от расположения пакета файл `detailed_data.csv` или `5stations_data.csv`. Иначе fallback `parents[2]` от `paths.py`.

**Важно:** Streamlit часто стартует с CWD = корень проекта — тогда всё ок. При запуске из другой директории явно `cd` в `6-th_lab`.

### Расширение

- Per-station split: заменить `time_split` и прокинуть в `train` / `build_validation_dataloader`.
- Доп. фичи: добавить колонки в CSV и в списки `train.build_datasets`.

---

## Модуль `train`

### `build_datasets(train_df, val_df, ...)`

Центральная конфигурация TFT:

| Параметр | Значение | Где менять |
|----------|----------|------------|
| `targets` | `default_model_targets()` — 9 целей | `targets_config.py` |
| `group_ids` | `["station_id"]` | `train.py` |
| `max_encoder_length` | 168 | аргумент функции / вызов в `train()` |
| `max_prediction_length` | 24 | то же |
| `time_varying_unknown_reals` | `[]` | марки только в `target`, не дублировать |
| `target_normalizer` | `MultiNormalizer([GroupNormalizer×N])` | N = len(targets) |
| `loss` / `output_size` | `MultiLoss([MAE()×N])`, `[1]*N` | `TemporalFusionTransformer.from_dataset` |

### `train(csv_path, ...) -> Path`

Последовательность:

1. `load_panel` + `time_split`
2. `build_datasets`
3. DataLoaders (`num_workers`, `persistent_workers` if > 0)
4. `TemporalFusionTransformer.from_dataset(training, ...)`
5. `pl.Trainer` + `EarlyStopping`, `ModelCheckpoint`, `CSVLogger`
6. `trainer.fit(tft, train_dataloaders=..., val_dataloaders=...)`
7. Копирование best ckpt → `artifacts/tft_azs.ckpt`
8. `torch.save(training.get_parameters(), tft_dataset_params.pt)`
9. `tft_meta.json` + `register_run(...)`

### Метаданные `tft_meta.json`

Пишутся в корень `artifacts/` и копируются в `runs/<id>/meta.json`. Поля: `csv_path`, `targets`, `fuel_grade_targets`, длины encoder/decoder, путь к ckpt.

### Изменение loss

Пример QuantileLoss (требует согласования `output_size` и UI):

```python
from pytorch_forecasting.metrics import QuantileLoss, MultiLoss

loss = MultiLoss([QuantileLoss()] * n_targets)
output_size = [7] * n_targets  # пример; сверить с документацией PF
```

После смены loss **обязательно** переобучение и обновление `forecast_viz` (извлечение квантилей).

---

## Модуль `targets_config`

Единый источник правды для целей:

```python
FUEL_GRADE_TARGETS  # 7 sales_* колонок
default_model_targets(include_total=True)  # total + fuels + shop
target_label(col) -> str  # подпись для UI
target_select_options(cols) -> dict[str, str]  # label -> column
```

**Порядок targets** задаёт индекс в тензоре `prediction[..., j]`. Не переставляйте список без переобучения и без обновления сценариев, где используется индекс `total_fuel_sales`.

---

## Модуль `model_registry`

### Файлы

```
artifacts/
  models_registry.json   # { "runs": [...], "active_run_id": "..." }
  active_model.json      # { "run_id": "..." }
  tft_azs.ckpt           # legacy mirror активного run
  tft_dataset_params.pt
  tft_meta.json
  runs/
    YYYYMMDD_HHMMSS/
      tft.ckpt
      dataset_params.pt
      meta.json
      metrics.csv          # опционально, копия из Lightning
```

### API

| Функция | Описание |
|---------|----------|
| `register_run(...)` | Новый run, опционально `set_active=True` |
| `list_runs()` | Список с флагом `is_active` |
| `get_run(run_id)` | meta.json |
| `set_active(run_id)` | Копирует ckpt/params/meta в корень artifacts |
| `resolve_active_paths()` | `dict` с Path: checkpoint, dataset_params, meta, run_id |
| `import_legacy_checkpoint()` | Миграция старых артефактов без реестра |

При первом заходе в дашборд: `import_legacy_checkpoint()` если `not st.session_state.registry_ready`.

### Добавление полей в реестр

Расширить `run_meta` в `register_run` и таблицу в `analytics.list_runs_table()`.

---

## Модуль `predict`

### Кэш `load_model`

```python
_model_by_key: dict[tuple[str, int], TemporalFusionTransformer] = {}
```

Ключ `(resolve_path, mtime_ns)`. Повторный predict в той же сессии Streamlit не делает `load_from_checkpoint`. Сброс при смене файла или `st.cache_data.clear()` в UI после `set_active` / обучения.

### Тихий inference

`_quiet_inference_logs()` — один раз за процесс: фильтры warnings, уровень логов Lightning, `torch.set_float32_matmul_precision("high")`.

### Разбор выхода PF (критично для сопровождения)

| Функция | Назначение |
|---------|------------|
| `unpack_predict_raw_x(pred_out)` | `Prediction` → `(output, x)` |
| `forward_output_as_dict(raw)` | namedtuple → dict |
| `tensors_to_numpy(value)` | tensor или `list[tensor]` (MultiLoss) → ndarray |
| `extract_prediction_array(raw)` | `[B, T, n_targets]` |
| `extract_decoder_target_array(x)` | факты decoder |
| `restore_network_output(model, raw)` | для `plot_prediction` + `.iget()` |

**Ошибка `'dict' object has no attribute 'iget'`:** в `plot_prediction` передавать `restore_network_output`, не голый dict.

**Нельзя:** `model.predict(batch)` по одному батчу — только полный `DataLoader`.

### `build_validation_dataloader`

```python
def build_validation_dataloader(
    csv_path: Path,
    batch_size: int = 128,
    val_hours: int = 336,
    num_workers: int | None = None,
) -> tuple[TemporalFusionTransformer, DataLoader, TimeSeriesDataSet]
```

Всегда грузит **val**-срез через `time_split`. Для train-метрик нужен отдельный loader или параметр `split=`.

### `forecast_steps_table(csv_path, sample_idx, on_progress=...)`

Один полный `predict` по loader → таблица с колонками `прогноз_<target>`, `факт_<target>`.

`sample_idx` — индекс ряда в батче (см. `station_index_map`).

### `station_index_map`

Сопоставление `station_id → индекс` в dataset predict mode. При расхождении длины — fallback по sorted `station_id`.

### `ProgressReporter`

```python
ProgressReporter = Callable[[int, int, str], None]
# (step, total, message)
```

Опциональный callback для Streamlit; в CLI передавать `None` или `dashboard_progress.noop_progress`.

---

## Модуль `forecast_viz`

### `target_options_for_csv`

```python
def target_options_for_csv(csv_path: Path | None = None) -> dict[str, str]:
    return target_select_options(active_targets())  # из dashboard_cache
```

Не вызывает `build_validation_dataloader` / `load_model`. Параметр `csv_path` оставлен для совместимости вызовов.

### Зависимости от predict

- `forecast_series_for_station` — может принять готовый `forecast_tbl` (избегает повторного predict).
- `plot_fuel_grades_forecast_plotly` — **один** `forecast_steps_table`, затем цикл по маркам (не 7× predict).

### Метрики

```python
evaluate_validation_metrics(csv_path, max_batches=50, on_progress=None) -> pd.DataFrame
```

- `trainer_kwargs={"limit_predict_batches": max_batches}`
- MAE, R² в **нормализованном** пространстве
- `evaluate_validation_mae` — алиас для обратной совместимости

### Сценарии

`scenario_promo_compare` — два полных predict на разных `val_df` (базовый / с подменой `promotion_fuel_active` на хвосте decoder для одной АЗС). Индекс топлива: `tlist.index("total_fuel_sales")`.

### Plotly vs matplotlib

- UI прогноз: `plot_forecast_plotly` — тёмный template, псевдо-P10/P90 из std остатков истории.
- PF native: `plot_prediction_example` → `model.plot_prediction(x, fwd, plot_attention=...)`.

---

## Интеграция pytorch-forecasting 1.7

### Объект `Prediction`

```python
pred_out = model.predict(loader, mode="raw", return_x=True)
# type: Prediction (namedtuple)
# .output  — сырой forward
# .x        — dict тензоров входа
```

Режимы `mode`: `"prediction"`, `"quantiles"`, `"raw"`, или `("raw", "attention")`.

### Multi-target + MultiLoss

Forward возвращает `prediction` как **список** тензоров (по одному на target). `tensors_to_numpy` конкатенирует по последней оси.

### GroupNormalizer

Нормализация по `station_id`. Метрики в UI без inverse transform — см. [расширение денормализации](#денормализация-прогнозов).

### Известные предупреждения

| Warning | Действие |
|---------|----------|
| NumPy not writable | `preprocess._ensure_writable_numeric_arrays`, filter в predict/dashboard |
| LeafSpec / checkpointing | filters в train/predict |
| Polyfit poorly conditioned | заменено на `lstsq` в `dashboard_charts` |

---

## Модуль `dashboard_cache`

Streamlit-специфичный слой кэша. Не импортируется из `train` / `predict` (кроме `active_targets` через `forecast_viz`).

| Функция | Кэш-ключ | Назначение |
|---------|----------|------------|
| `cached_load_panel(path_str, mtime_ns)` | путь + mtime файла | `load_panel` один раз на CSV |
| `get_cached_panel(path)` | обёртка | → `PANEL_DF` в `dashboard.py` |
| `cached_targets_list(run_key, pm, mm)` | run_id + mtime meta/params | список `targets` без TFT |
| `active_targets()` | обёртка | для `target_options_for_csv` |
| `cached_station_map(...)` | csv + params mtime + run | `station_index_map` |

Инвалидация: смена CSV (mtime), `set_active` / обучение → `st.cache_data.clear()` и `st.cache_resource.clear()` в `dashboard.py`.

---

## Модуль `dashboard_jobs`

Фоновые задачи в **daemon thread**. Состояние: `st.session_state["bg_job"]`.

### Структура `bg_job`

```python
{
    "title": str,           # «Прогноз TFT»
    "page": str,            # PAGE_ID: fore, train, model, ...
    "status": "running" | "done" | "error",
    "message": str,
    "progress": float,      # 0.0 .. 1.0
    "log_lines": list[str], # обучение
    "result": Any,
    "error": str | None,
    "result_type": str,     # plotly | dataframe | train | mpl | markdown | generic
    "plot_height": int | None,
}
```

### API

| Функция | Описание |
|---------|----------|
| `job_running()` | Есть ли активная задача |
| `submit_job(title, page, fn, result_type=..., plot_height=...)` | `fn(report) -> result` в потоке |
| `submit_training_subprocess(title, page, cmd, cwd, env)` | Отдельный процесс train |
| `try_submit_or_warn(...)` | submit + `st.toast` + `st.rerun` или warning |
| `render_global_job_banner()` | Панель вверху главной области |
| `render_job_result_for_page(page_id, plotly_chart=...)` | Вывод result на странице |
| `clear_job()` | Сброс session_state |

### Ограничения (намеренные)

- **Одна** активная задача — вторая `submit_*` вернёт `False`.
- PyTorch/Lightning в потоке: для типичного predict работает; не запускать два predict параллельно.
- `session_state` из потока: обновления `job["progress"]` — best-effort; UI обновляется при следующем rerun.
- Обучение — **subprocess**, не `trainer.fit` в Streamlit-процессе.

### `ProgressReporter`

Тот же контракт, что в `predict` / `forecast_viz`:

```python
def report(step: int, total: int, message: str) -> None: ...
```

`progress_task` (модуль `dashboard_progress`) в UI для синхронных операций **больше не используется** для predict/train — только callback в jobs.

---

## Дашборд и UI-слой

### Жизненный цикл `dashboard.py`

1. `st.set_page_config`, тема CSS.
2. Sidebar: модель, `DATA_PATH`, **`PAGE_ID`** (`NAV_PAGES` + `st.radio` key=`main_nav`).
3. `PANEL_DF = get_cached_panel(DATA_PATH)` — один раз за rerun.
4. Метрики модели (4 columns).
5. `render_global_job_banner()`.
6. `if PAGE_ID == "network": _page_network()` … только **одна** ветка.

`import_legacy_checkpoint()` — один раз: `st.session_state.registry_ready`.

### Константа навигации

```python
NAV_PAGES: list[tuple[str, str]] = [
    ("dict", "Справочник"),
    ("network", "Сеть"),
    ...
]
NAV_LABEL_TO_ID = {label: pid for pid, label in NAV_PAGES}
```

### Функции страниц

| PAGE_ID | Функция | Fragment |
|---------|---------|----------|
| `dict` | inline | нет |
| `network` | `_page_network()` | нет |
| `analytics` | `_page_analytics()` | `@st.fragment` |
| `train` | inline + `render_job_result_for_page("train")` | нет |
| `model` | inline | нет |
| `fore` | `_page_fore()` | нет |
| `table` | `_page_table()` | нет |
| `scenario` | `_page_scenario()` | нет |
| `interp` | `_page_interp()` | нет |
| `rec` | `_page_rec()` | нет |
| `export` | inline | нет |

### Паттерн: тяжёлая кнопка

```python
if st.button("...", disabled=job_running()):
    def work(report: ProgressReporter):
        return some_heavy_fn(DATA_PATH, on_progress=report)
    try_submit_or_warn("Заголовок", PAGE_ID, work, result_type="dataframe")
```

Результат: `render_job_result_for_page(PAGE_ID, plotly_chart=_plotly)` в начале страницы.

### Обучение

`submit_training_subprocess` → worker читает stdout → `job["log_lines"]`. По `done` с `returncode==0`: `import_legacy_checkpoint()`, clear caches, `list_runs_table()` в `render_job_result_for_page`.

### `dashboard_charts`

Без Streamlit/torch — unit-test / notebook friendly.

### Добавление нового раздела

1. Добавить `("id", "Подпись")` в `NAV_PAGES`.
2. Реализовать `_page_*()` или ветку `elif PAGE_ID == "id":`.
3. Тяжёлые кнопки — через `try_submit_or_warn`, не inline `model.predict`.
4. Не возвращать к `st.tabs` для верхнего уровня — ломает ленивый рендер.

### `width="stretch"`

Вместо устаревшего `use_container_width` для `st.dataframe`, `st.plotly_chart`, кнопок sidebar.

---

## Контракты и инварианты

1. **`dataset_params.pt` и `checkpoint` обучены вместе** — нельзя смешивать от разных run без ошибок размерности.
2. **Список `targets` в params должен совпадать** с архитектурой головы модели.
3. **`time_idx` непрерывен внутри station** — пропуски часов: `allow_missing_timesteps=False` упадёт.
4. **`predict=True` на validation dataset** — для inference и dashboard.
5. **Индексация батча** — `station_index_map` зависит от порядка в `TimeSeriesDataSet`; после смены `batch_size` / данных перепроверять.
6. **Сценарии меняют только known reals** на decoder horizon — не полноценный counterfactual по всем факторам.

---

## Расширение функциональности

### Денормализация прогнозов

В батче `x` есть `target_scale` (если `add_target_scales=True` при обучении). Псевдокод:

```python
# после extract_prediction_array
scales = x["target_scale"]  # форма зависит от PF версии
# применить inverse transform encoders из dataset_parameters
```

Реализацию смотреть в `GroupNormalizer` / `dataset.transform_values` в документации PF.

### Квантили P10/P50/P90

1. Заменить `MAE()` на `QuantileLoss` в `train.py`.
2. Обновить `output_size`.
3. В `extract_prediction_array` обрабатывать размерность квантилей.
4. В `plot_forecast_plotly` рисовать band из квантилей, убрать псевдо-P10/P90.

### Кэш predict в Streamlit

Уже есть: `dashboard_cache` (panel, targets, station map), `predict._model_by_key`. Дополнительно можно кэшировать таблицу прогноза:

```python
@st.cache_data(ttl=600, show_spinner=False)
def cached_forecast_table(path_str: str, mtime: int, run_id: str, sample_idx: int) -> pd.DataFrame:
    return forecast_steps_table(Path(path_str), sample_idx=sample_idx)
```

Ключ **обязан** включать `run_id` и mtime CSV.

### Новая цель (например `sales_LPG`)

1. Колонка в CSV.
2. `FUEL_GRADE_TARGETS` или отдельный список в `targets_config`.
3. `default_model_targets()` +1 normalizer +1 MAE.
4. Переобучение.
5. `TARGET_DISPLAY`, `data_dictionary.json`.

### API / REST поверх predict

Вынести ядро в функцию:

```python
def predict_station(csv_path: Path, station_id: int) -> dict[str, pd.DataFrame]:
    ...
```

FastAPI-обёртка без Streamlit — отдельный модуль, не импортировать `dashboard`.

### Матрица риска быстрее

`analytics.station_risk_table` вызывает `forecast_steps_table` на **каждую** АЗС → N× predict.

Оптимизация: один predict на весь loader, индексация всех `station_id` из `station_index_map` за один проход.

---

## Тестирование и отладка

### Минимальные smoke-тесты (ручные)

```bash
# 1. Датасет
PYTHONPATH=src python -c "
from azs_tft.preprocess import load_panel, time_split, default_csv
df=load_panel(default_csv()); tr,va=time_split(df,48); print(len(tr),len(va))
"

# 2. Обучение 1 эпоха
PYTHONPATH=src python -m azs_tft.train --csv 5stations_data.csv --epochs 1 --limit-train-batches 2

# 3. Predict
PYTHONPATH=src python -c "
from pathlib import Path
from azs_tft.forecast_viz import evaluate_validation_metrics
print(evaluate_validation_metrics(Path('5stations_data.csv'), max_batches=2))
"
```

### Отладка predict в REPL

```python
from pathlib import Path
from azs_tft.predict import build_validation_dataloader, unpack_predict_raw_x, extract_prediction_array

model, loader, ds = build_validation_dataloader(Path("5stations_data.csv"), batch_size=32)
out = model.predict(loader, mode="raw", return_x=True, trainer_kwargs={"limit_predict_batches": 1})
raw, x = unpack_predict_raw_x(out)
pred = extract_prediction_array(raw)
print(pred.shape, ds.target)
```

### Логи Lightning

`artifacts/logs/` или `artifacts/runs/<id>/metrics.csv`. Dashboard читает через `analytics.read_training_metrics_csv(run_id)`.

### Типичные исключения

| Exception | Причина |
|-----------|---------|
| `KeyError: prediction` | Неверный разбор `Prediction`; обновить `unpack_*` |
| `size mismatch` | Чекпоинт ≠ dataset_params |
| `StreamlitDuplicateElementId` | Дубли `st.button` без `key=` |
| CUDA OOM | Уменьшить `batch_size` |

---

## Производительность

### Уже реализовано

| Проблема | Решение |
|----------|---------|
| Все `st.tabs` рендерились за rerun | `PAGE_ID` + одна ветка `elif` |
| `load_panel` 5–7× за rerun | `get_cached_panel` / `@st.cache_data` |
| Список целей грузил TFT | `active_targets()` из meta / `dataset_params.pt` |
| Повторный `load_from_checkpoint` | `_model_by_key` в `predict.py` |
| UI блокировался на predict/train | `dashboard_jobs` (thread / subprocess) |
| `station_index_map` каждый раз | `cached_station_map` |

### Узкие места (остаются)

| Операция | Узкое место | Рекомендация |
|----------|-------------|--------------|
| `model.predict(loader)` | Полный val loader, Lightning | GPU; `5stations` для dev |
| Метрики MAE/R² | До 50 val batches | Фон; уменьшить `max_batches` в коде |
| Матрица риска | `station_risk_table`: N× `forecast_steps_table` | Один predict на loader (TODO в `analytics.py`) |
| Аналитика sub-tabs | 4× Plotly за rerun fragment | Приемлемо; можно radio вместо tabs |
| Поток + PyTorch | GIL, не два predict | Одна `bg_job` — by design |

### Streamlit rerun

Любой виджет → полный rerun скрипта. Фоновая задача **не отменяет** rerun, но rerun завершается быстро (нет блокировки на predict). Прогресс в UI — при следующем rerun (кнопка «Обновить статус», смена раздела).

### Сброс кэша

Вызывать при: `set_active`, успешном обучении, смене чекпоинта вручную:

```python
st.cache_data.clear()
st.cache_resource.clear()
```

---


## Соглашения по коду

- **Язык UI и docstring:** русский.
- **Типы:** `from __future__ import annotations`, по возможности явные `Path`.
- **Импорты:** stdlib → third-party → `azs_tft`.
- **Минимальный diff:** не тянуть torch в chart-модули.
- **Секреты:** только `.env`, не в репозиторий.
- **Коммиты:** не включать `artifacts/*.ckpt`.

### Форматирование (рекомендуется)

```bash
ruff check src/azs_tft
ruff format src/azs_tft
```

Ruff в проекте не обязателен — при добавлении положить в `pyproject.toml` `[tool.ruff]`.

---

## Чеклист перед PR / сдачей

- [ ] `PYTHONPATH=src python -m azs_tft.train` на `5stations_data.csv` завершается без ошибок
- [ ] `streamlit run src/azs_tft/dashboard.py` — разделы Прогноз / Модель / Аналитика; переключение без минутной паузы (на `5stations`)
- [ ] Фоновая задача: запустить прогноз → перейти в Сеть → вернуться → результат на Прогноз
- [ ] После смены `targets` — переобучена модель, обновлены `data_dictionary.json` и README
- [ ] Нет ключей/API в коде
- [ ] Новые кнопки Streamlit с уникальным `key=`
- [ ] Долгие операции через `dashboard_jobs` (не блокирующий `progress_task` в main thread)
- [ ] README.md / DEVELOPER.md обновлены при изменении контрактов

---

## Связанные файлы документации

| Файл | Аудитория |
|------|-----------|
| [README.md](README.md) | Пользователь, эксплуатация |
| [DEVELOPER.md](DEVELOPER.md) | Разработчик (этот документ) |
| [data_dictionary.json](src/azs_tft/data_dictionary.json) | Схема данных + TFT roles |
| `описание данных.docx` | Бизнес-смысл полей |
| `TFT_анализ.pdf` | Методология TFT-анализа |

---

*Версия: 9 targets, `Prediction.output`, registry runs, `dashboard_jobs` + `dashboard_cache`, навигация `PAGE_ID`, in-process cache `load_model`.*
