from __future__ import annotations

import csv
import io
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from missing_value_pipeline import clean_missing_values
from storage import build_storage_metadata, maybe_upload_to_object_storage


load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "public_data"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "public_data"


class PublicDataPipelineError(RuntimeError):
    pass


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _safe_slug(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    return cleaned.strip("_") or "public_statistics"


def _parse_json_env(name: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default or {}
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError as error:
        raise PublicDataPipelineError(f"{name}는 JSON 문자열이어야 합니다: {error}") from error
    if not isinstance(parsed, dict):
        raise PublicDataPipelineError(f"{name}는 JSON object 형식이어야 합니다.")
    return parsed


def _split_csv_env(name: str) -> list[str]:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def public_data_config() -> dict[str, Any]:
    api_url = os.getenv("PUBLIC_DATA_API_URL", "").strip()
    if not api_url:
        raise PublicDataPipelineError("PUBLIC_DATA_API_URL 환경변수가 설정되어 있지 않습니다.")

    params = _parse_json_env("PUBLIC_DATA_API_PARAMS", {})
    api_key = os.getenv("PUBLIC_DATA_API_KEY", "").strip()
    api_key_param = os.getenv("PUBLIC_DATA_API_KEY_PARAM", "serviceKey").strip()
    if api_key and api_key_param and api_key_param not in params:
        params[api_key_param] = api_key

    response_format = os.getenv("PUBLIC_DATA_API_FORMAT", "json").strip().lower()
    if response_format not in {"json", "xml"}:
        raise PublicDataPipelineError("PUBLIC_DATA_API_FORMAT은 json 또는 xml이어야 합니다.")

    return {
        "api_url": api_url,
        "api_key_present": bool(api_key),
        "params": params,
        "format": response_format,
        "dataset_name": os.getenv("PUBLIC_DATA_DATASET_NAME", "public-statistics").strip() or "public-statistics",
        "category_name": os.getenv("PUBLIC_DATA_CATEGORY_NAME", "공공데이터 자동 수집").strip() or "공공데이터 자동 수집",
        "target_column": os.getenv("PUBLIC_DATA_TARGET_COLUMN", "").strip(),
        "required_columns": _split_csv_env("PUBLIC_DATA_REQUIRED_COLUMNS"),
        "drop_threshold": float(os.getenv("PUBLIC_DATA_MISSING_DROP_THRESHOLD", "0.7") or 0.7),
        "timeout": float(os.getenv("PUBLIC_DATA_HTTP_TIMEOUT", "30") or 30),
    }


def _fetch_public_data(config: dict[str, Any]) -> tuple[bytes, str]:
    query = urllib.parse.urlencode(config["params"], doseq=True)
    separator = "&" if urllib.parse.urlparse(config["api_url"]).query else "?"
    url = f"{config['api_url']}{separator}{query}" if query else config["api_url"]
    request = urllib.request.Request(url, headers={"User-Agent": "AI_Cloud public data batch"})
    try:
        with urllib.request.urlopen(request, timeout=config["timeout"]) as response:
            status = getattr(response, "status", 200)
            body = response.read()
    except urllib.error.HTTPError as error:
        raise PublicDataPipelineError(f"공공데이터 API HTTP 오류 status={error.code}") from error
    except urllib.error.URLError as error:
        raise PublicDataPipelineError(f"공공데이터 API 요청 실패: {error.reason}") from error

    if status >= 400:
        raise PublicDataPipelineError(f"공공데이터 API HTTP 오류 status={status}")
    if not body.strip():
        raise PublicDataPipelineError("공공데이터 API 응답이 비어 있습니다.")
    return body, url


def _flatten_record(record: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened = {}
    for key, value in record.items():
        column = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_record(value, column))
        elif isinstance(value, list):
            flattened[column] = json.dumps(value, ensure_ascii=False)
        else:
            flattened[column] = value
    return flattened


def _find_records_json(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        if all(isinstance(item, dict) for item in payload):
            return payload
        return []
    if not isinstance(payload, dict):
        return []

    preferred_keys = ("item", "items", "row", "rows", "data", "list", "records", "result")
    for key in preferred_keys:
        value = payload.get(key)
        records = _find_records_json(value)
        if records:
            return records
    for value in payload.values():
        records = _find_records_json(value)
        if records:
            return records
    return []


def _parse_json_dataframe(raw_body: bytes) -> pd.DataFrame:
    try:
        payload = json.loads(raw_body.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicDataPipelineError(f"JSON 응답을 파싱할 수 없습니다: {error}") from error
    records = [_flatten_record(record) for record in _find_records_json(payload)]
    if not records:
        raise PublicDataPipelineError("JSON 응답에서 행 데이터 목록을 찾지 못했습니다.")
    return pd.DataFrame(records)


def _strip_xml_namespace(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _xml_leaf_dict(element: ET.Element) -> dict[str, Any]:
    children = list(element)
    if not children:
        return {_strip_xml_namespace(element.tag): (element.text or "").strip()}
    record = {}
    for child in children:
        child_name = _strip_xml_namespace(child.tag)
        grand_children = list(child)
        if grand_children:
            record[child_name] = json.dumps(_xml_leaf_dict(child), ensure_ascii=False)
        else:
            record[child_name] = (child.text or "").strip()
    return record


def _parse_xml_dataframe(raw_body: bytes) -> pd.DataFrame:
    try:
        root = ET.fromstring(raw_body)
    except ET.ParseError as error:
        raise PublicDataPipelineError(f"XML 응답을 파싱할 수 없습니다: {error}") from error
    candidates = []
    for element in root.iter():
        children = list(element)
        if children and all(not list(child) for child in children):
            candidates.append(_xml_leaf_dict(element))
    if not candidates:
        raise PublicDataPipelineError("XML 응답에서 행 데이터 목록을 찾지 못했습니다.")
    max_width = max(len(record) for record in candidates)
    records = [record for record in candidates if len(record) == max_width]
    return pd.DataFrame(records)


def response_to_dataframe(raw_body: bytes, response_format: str) -> pd.DataFrame:
    if response_format == "json":
        dataframe = _parse_json_dataframe(raw_body)
    else:
        dataframe = _parse_xml_dataframe(raw_body)
    if dataframe.empty:
        raise PublicDataPipelineError("공공데이터 API 결과가 비어 있어 저장하지 않습니다.")
    return dataframe


def _dataframe_to_csv_text(dataframe: pd.DataFrame) -> str:
    buffer = io.StringIO()
    dataframe.to_csv(buffer, index=False, quoting=csv.QUOTE_MINIMAL)
    return buffer.getvalue()


def _save_pipeline_files(
    dataset_slug: str,
    timestamp: str,
    response_format: str,
    raw_body: bytes,
    raw_dataframe: pd.DataFrame,
    processed_dataframe: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, str]:
    raw_dir = RAW_ROOT / dataset_slug
    processed_dir = PROCESSED_ROOT / dataset_slug
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_response_path = raw_dir / f"{dataset_slug}_{timestamp}_response.{response_format}"
    raw_csv_path = raw_dir / f"{dataset_slug}_{timestamp}_raw.csv"
    processed_csv_path = processed_dir / f"{dataset_slug}_{timestamp}_processed.csv"
    metadata_path = processed_dir / f"{dataset_slug}_{timestamp}_metadata.json"

    raw_response_path.write_bytes(raw_body)
    raw_dataframe.to_csv(raw_csv_path, index=False)
    processed_dataframe.to_csv(processed_csv_path, index=False)

    raw_response_uri = maybe_upload_to_object_storage(raw_response_path)
    raw_csv_uri = maybe_upload_to_object_storage(raw_csv_path)
    processed_uri = maybe_upload_to_object_storage(processed_csv_path)

    metadata.update(
        {
            "raw_response_file": os.path.relpath(raw_response_path, PROJECT_ROOT),
            "raw_file": os.path.relpath(raw_csv_path, PROJECT_ROOT),
            "processed_file": os.path.relpath(processed_csv_path, PROJECT_ROOT),
            "raw_response_storage": build_storage_metadata(raw_response_path, raw_response_uri),
            "raw_storage": build_storage_metadata(raw_csv_path, raw_csv_uri),
            "processed_storage": build_storage_metadata(processed_csv_path, processed_uri),
        }
    )
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata_uri = maybe_upload_to_object_storage(metadata_path)
    metadata["metadata_storage"] = build_storage_metadata(metadata_path, metadata_uri)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if metadata_uri:
        maybe_upload_to_object_storage(metadata_path)

    return {
        "raw_response_path": str(raw_response_path),
        "raw_csv_path": str(raw_csv_path),
        "processed_path": str(processed_csv_path),
        "metadata_path": str(metadata_path),
        "storage_uri": processed_uri,
        "metadata_storage_uri": metadata_uri,
    }


def build_public_data_rag_content(metadata: dict[str, Any], dataframe: pd.DataFrame) -> str:
    report = metadata.get("missing_report", {})
    columns = dataframe.columns.tolist()
    preview_stats = dataframe.describe(include="all").fillna("").head(8).to_dict()
    return (
        f"공공데이터 자동 수집 요약\n"
        f"데이터셋: {metadata.get('dataset_name')}\n"
        f"수집 시각: {metadata.get('collected_at')}\n"
        f"출처 URL: {metadata.get('source_url')}\n"
        f"행/열: {report.get('rows_before')}행 {report.get('columns_before')}열 -> "
        f"{report.get('rows_after')}행 {report.get('columns_after')}열\n"
        f"결측치: {report.get('missing_total_before')}개 -> {report.get('missing_total_after')}개\n"
        f"제거 컬럼: {', '.join(report.get('dropped_columns') or []) or '없음'}\n"
        f"컬럼 목록: {', '.join(map(str, columns))}\n"
        f"처리 내역: {json.dumps(report.get('actions', []), ensure_ascii=False)}\n"
        f"요약 통계: {json.dumps(preview_stats, ensure_ascii=False)}"
    )


def collect_public_data() -> dict[str, Any]:
    config = public_data_config()
    timestamp = _timestamp()
    dataset_slug = _safe_slug(config["dataset_name"])

    raw_body, request_url = _fetch_public_data(config)
    raw_dataframe = response_to_dataframe(raw_body, config["format"])
    if raw_dataframe.empty:
        raise PublicDataPipelineError("빈 데이터는 저장하지 않습니다.")

    cleaned_dataframe, missing_report = clean_missing_values(
        raw_dataframe,
        target_column=config["target_column"],
        drop_threshold=config["drop_threshold"],
        required_columns=config["required_columns"],
    )
    if cleaned_dataframe.empty:
        raise PublicDataPipelineError("결측치 처리 후 데이터가 비어 있어 저장하지 않습니다.")

    metadata = {
        "source": "public_data",
        "dataset_name": config["dataset_name"],
        "category_name": config["category_name"],
        "source_url": config["api_url"],
        "request_url": request_url,
        "response_format": config["format"],
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "lineage": {
            "raw": "data/raw/public_data",
            "processed": "data/processed/public_data",
            "metadata": "documents.metadata_path",
        },
        "required_columns": config["required_columns"],
        "target_column": config["target_column"],
        "missing_drop_threshold": config["drop_threshold"],
        "missing_report": missing_report,
    }
    paths = _save_pipeline_files(
        dataset_slug,
        timestamp,
        config["format"],
        raw_body,
        raw_dataframe,
        cleaned_dataframe,
        metadata,
    )
    metadata_path = Path(paths["metadata_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "dataset_name": config["dataset_name"],
        "category_name": config["category_name"],
        "document_title": f"{dataset_slug}_processed.csv",
        "display_name": f"{config['dataset_name']} 공공데이터",
        "source_type": "public_data",
        "source_url": config["api_url"],
        "file_path": paths["raw_csv_path"],
        "processed_path": paths["processed_path"],
        "metadata_path": paths["metadata_path"],
        "storage_uri": paths["storage_uri"],
        "processed_csv": _dataframe_to_csv_text(cleaned_dataframe),
        "rag_content": build_public_data_rag_content(metadata, cleaned_dataframe),
        "metadata": metadata,
        "rows_before": missing_report["rows_before"],
        "rows_after": missing_report["rows_after"],
    }
