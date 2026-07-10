from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd


IDENTIFIER_KEYWORDS = (
    "id",
    "idx",
    "index",
    "no",
    "number",
    "seq",
    "code",
    "cd",
    "key",
    "serial",
    "주소",
    "address",
    "지역",
    "region",
    "시도",
    "시군구",
    "법정동",
    "행정동",
    "코드",
    "번호",
    "식별",
)


def _missing_stats(dataframe: pd.DataFrame) -> dict[str, dict[str, Any]]:
    row_count = len(dataframe)
    stats = {}
    for column in dataframe.columns:
        count = int(dataframe[column].isna().sum())
        stats[str(column)] = {
            "count": count,
            "ratio": float(count / row_count) if row_count else 0.0,
        }
    return stats


def _is_identifier_like(column: str, series: pd.Series) -> bool:
    normalized = str(column).strip().lower().replace("_", "").replace("-", "")
    if any(keyword in normalized for keyword in IDENTIFIER_KEYWORDS):
        return True
    non_null = series.dropna()
    if non_null.empty:
        return False
    unique_ratio = non_null.astype(str).nunique(dropna=True) / max(len(non_null), 1)
    return unique_ratio >= 0.95 and len(non_null) >= 20


def _looks_datetime(series: pd.Series) -> tuple[bool, pd.Series]:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True, series
    converted = pd.to_datetime(series, errors="coerce")
    non_null_count = max(int(series.notna().sum()), 1)
    return (converted.notna().sum() / non_null_count) >= 0.6, converted


def clean_missing_values(
    dataframe: pd.DataFrame,
    target_column: str = "",
    drop_threshold: float = 0.7,
    required_columns: list[str] | None = None,
    fill_dates: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required_columns = [column for column in (required_columns or []) if column]
    missing_required = [column for column in required_columns if column not in dataframe.columns]
    if missing_required:
        raise ValueError(f"필수 컬럼이 없습니다: {', '.join(missing_required)}")

    cleaned = dataframe.copy()
    rows_before = len(cleaned)
    columns_before = len(cleaned.columns)
    missing_before = _missing_stats(cleaned)
    actions: list[dict[str, Any]] = []

    if target_column:
        if target_column not in cleaned.columns:
            raise ValueError(f"target_column '{target_column}' 컬럼이 없습니다.")
        before_drop = len(cleaned)
        cleaned = cleaned.dropna(subset=[target_column]).copy()
        dropped = before_drop - len(cleaned)
        if dropped:
            actions.append(
                {
                    "column": target_column,
                    "action": "drop_rows",
                    "rows": int(dropped),
                    "reason": "target_column 결측 행 제거",
                }
            )

    dropped_columns: list[str] = []
    for column in list(cleaned.columns):
        if column == target_column:
            continue
        ratio = float(cleaned[column].isna().mean()) if len(cleaned) else 0.0
        if ratio >= drop_threshold and cleaned[column].isna().sum() > 0:
            cleaned = cleaned.drop(columns=[column])
            dropped_columns.append(str(column))
            actions.append(
                {
                    "column": str(column),
                    "action": "drop_column",
                    "missing_ratio": ratio,
                    "threshold": drop_threshold,
                    "reason": "결측률 임계값 이상",
                }
            )

    for column in list(cleaned.columns):
        missing_count = int(cleaned[column].isna().sum())
        if missing_count == 0:
            continue

        series = cleaned[column]
        is_datetime, datetime_series = _looks_datetime(series)
        if is_datetime:
            cleaned[column] = datetime_series
            if fill_dates:
                cleaned[column] = cleaned[column].ffill().bfill()
                method = "datetime_ffill_bfill"
            else:
                method = "datetime_parse_only"
            actions.append(
                {
                    "column": str(column),
                    "action": method,
                    "missing_before": missing_count,
                    "missing_after": int(cleaned[column].isna().sum()),
                }
            )
            continue

        if _is_identifier_like(str(column), series):
            cleaned[column] = series.astype("object").fillna("Unknown")
            actions.append(
                {
                    "column": str(column),
                    "action": "fill_identifier_unknown",
                    "missing_before": missing_count,
                    "value": "Unknown",
                    "reason": "ID/코드/주소/지역명 후보는 중앙값 대체 제외",
                }
            )
            continue

        numeric_series = pd.to_numeric(series, errors="coerce")
        numeric_ratio = numeric_series.notna().sum() / max(series.notna().sum(), 1)
        if pd.api.types.is_numeric_dtype(series) or numeric_ratio >= 0.8:
            median_value = numeric_series.median()
            if pd.isna(median_value):
                cleaned[column] = numeric_series
                fill_value = None
            else:
                cleaned[column] = numeric_series.fillna(median_value)
                fill_value = float(median_value)
            actions.append(
                {
                    "column": str(column),
                    "action": "fill_numeric_median",
                    "missing_before": missing_count,
                    "value": fill_value,
                }
            )
            continue

        mode_values = series.dropna().mode()
        fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"
        cleaned[column] = series.fillna(fill_value)
        actions.append(
            {
                "column": str(column),
                "action": "fill_categorical_mode",
                "missing_before": missing_count,
                "value": str(fill_value),
            }
        )

    missing_after = _missing_stats(cleaned)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows_before": rows_before,
        "rows_after": len(cleaned),
        "columns_before": columns_before,
        "columns_after": len(cleaned.columns),
        "missing_before": missing_before,
        "missing_after": missing_after,
        "missing_total_before": int(sum(item["count"] for item in missing_before.values())),
        "missing_total_after": int(sum(item["count"] for item in missing_after.values())),
        "dropped_rows": rows_before - len(cleaned),
        "dropped_columns": dropped_columns,
        "actions": actions,
    }
    return cleaned, report
