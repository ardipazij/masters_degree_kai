"""Цели TFT: марки топлива, сумма, магазин."""

from __future__ import annotations

FUEL_GRADE_TARGETS: tuple[str, ...] = (
    "sales_AI92",
    "sales_AI95",
    "sales_AI98",
    "sales_DT_EURO",
    "sales_DT_TANEKO",
    "sales_DT_SUMMER",
    "sales_DT_WINTER",
)

TARGET_DISPLAY: dict[str, str] = {
    "total_fuel_sales": "Сумма топливо",
    "shop_total_revenue": "Магазин",
    "sales_AI92": "АИ-92",
    "sales_AI95": "АИ-95",
    "sales_AI98": "АИ-98",
    "sales_DT_EURO": "ДТ Euro",
    "sales_DT_TANEKO": "ДТ Taneko",
    "sales_DT_SUMMER": "ДТ летнее",
    "sales_DT_WINTER": "ДТ зимнее",
}


def default_model_targets(include_total: bool = True) -> list[str]:
    """Порядок целей при обучении (совпадает с индексами выхода MultiLoss)."""
    out: list[str] = []
    if include_total:
        out.append("total_fuel_sales")
    out.extend(FUEL_GRADE_TARGETS)
    out.append("shop_total_revenue")
    return out


def fuel_targets_only() -> list[str]:
    return list(FUEL_GRADE_TARGETS)


def target_label(col: str) -> str:
    return TARGET_DISPLAY.get(col, col)


def target_select_options(cols: list[str] | None = None) -> dict[str, str]:
    """Подпись в UI → имя колонки."""
    names = cols if cols is not None else default_model_targets()
    return {target_label(c): c for c in names}
