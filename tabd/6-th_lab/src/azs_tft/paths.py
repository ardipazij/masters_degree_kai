from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Корень проекта: текущая рабочая директория с CSV или поиск вверх от пакета."""
    cwd = Path.cwd()
    for p in (cwd, *cwd.parents):
        if (p / "detailed_data.csv").exists() or (p / "5stations_data.csv").exists():
            return p
    here = Path(__file__).resolve()
    for par in here.parents:
        if (par / "detailed_data.csv").exists() or (par / "5stations_data.csv").exists():
            return par
    return here.parents[2]


def artifacts_dir() -> Path:
    p = project_root() / "artifacts"
    p.mkdir(parents=True, exist_ok=True)
    return p
