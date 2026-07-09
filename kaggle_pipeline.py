import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from storage import build_storage_metadata, maybe_upload_to_object_storage


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "kaggle"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "kaggle"



def get_kaggle_bin() -> str:
    kaggle_bin = shutil.which("kaggle")
    if not kaggle_bin:
        raise RuntimeError("현재 가상환경에서 kaggle CLI를 찾을 수 없습니다. pip install kaggle을 실행하세요.")
    return kaggle_bin


def normalize_dataset_id(dataset_id: str) -> str:
    cleaned = dataset_id.strip().lower()
    cleaned = cleaned.replace("https://www.kaggle.com/datasets/", "")
    cleaned = cleaned.replace("http://www.kaggle.com/datasets/", "")
    cleaned = cleaned.replace("https://kaggle.com/datasets/", "")
    cleaned = cleaned.replace("http://kaggle.com/datasets/", "")
    cleaned = cleaned.strip("/")
    return cleaned


def slugify_dataset_id(dataset_id: str) -> str:
    cleaned = normalize_dataset_id(dataset_id)
    return cleaned.replace("/", "__")


def _run_kaggle_download(dataset_id: str, raw_dir: Path) -> None:
    command = [
        get_kaggle_bin(),
        "datasets",
        "download",
        "-d",
        dataset_id,
        "-p",
        str(raw_dir),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as error:
        raise RuntimeError(
            "Kaggle CLI를 실행할 수 없습니다. 현재 가상환경에 kaggle 패키지가 설치되어 있는지 확인하세요."
        ) from error
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        if "No module named kaggle" in message:
            message = "현재 가상환경에 kaggle 패키지가 설치되어 있지 않습니다. requirements.txt 설치를 먼저 실행하세요."
        if "Could not find kaggle.json" in message or "kaggle.json" in message:
            message = (
                f"{message}\n"
                "확인할 것: Kaggle API Token(kaggle.json)을 C:\\Users\\<사용자>\\.kaggle\\kaggle.json 위치에 저장하거나 "
                "KAGGLE_USERNAME, KAGGLE_KEY 환경변수를 설정하세요."
            )
        if "403" in message or "Forbidden" in message:
            message = (
                f"{message}\n"
                "확인할 것: dataset id가 owner/dataset 형식인지, "
                "비공개 데이터셋이 아닌지, Kaggle 페이지에서 데이터셋 접근 권한이나 약관 동의가 필요한지, "
                "노출된 API 토큰을 폐기한 뒤 새 토큰을 저장했는지 확인하세요."
            )
        raise RuntimeError(f"Kaggle 다운로드에 실패했습니다: {message}")


def _format_kaggle_error(message: str, action: str) -> str:
    if "No module named kaggle" in message:
        message = "현재 가상환경에 kaggle 패키지가 설치되어 있지 않습니다. requirements.txt 설치를 먼저 실행하세요."
    if "Could not find kaggle.json" in message or "kaggle.json" in message:
        message = (
            f"{message}\n"
            "확인할 것: Kaggle API Token(kaggle.json)을 C:\\Users\\<사용자>\\.kaggle\\kaggle.json 위치에 저장하거나 "
            "KAGGLE_USERNAME, KAGGLE_KEY 환경변수를 설정하세요."
        )
    if "403" in message or "Forbidden" in message:
        message = (
            f"{message}\n"
            "확인할 것: Kaggle 페이지에서 데이터셋 접근 권한이나 약관 동의가 필요한지, "
            "노출된 API 토큰을 폐기한 뒤 새 토큰을 저장했는지 확인하세요."
        )
    return f"Kaggle {action}에 실패했습니다: {message}"


def search_kaggle_datasets(keyword: str, limit: int = 10) -> list[dict]:
    cleaned_keyword = keyword.strip()
    if not cleaned_keyword:
        return []

    command = [
        get_kaggle_bin(),
        "datasets",
        "list",
        "--search",
        cleaned_keyword,
        "--file-type",
        "csv",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    except FileNotFoundError as error:
        raise RuntimeError(
            "Kaggle CLI를 실행할 수 없습니다. 현재 가상환경에 kaggle 패키지가 설치되어 있는지 확인하세요."
        ) from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Kaggle 검색 응답이 지연되고 있습니다. 잠시 뒤 다시 시도하세요.") from error

    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(_format_kaggle_error(message, "검색"))

    import re

    datasets = []
    for line in (result.stdout or "").splitlines():
        line = line.strip()
        if not line or line.lower().startswith("ref"):
            continue

        parts = re.split(r"\s{2,}", line)
        if len(parts) < 2 or "/" not in parts[0]:
            continue

        dataset_ref = parts[0].strip()
        title = parts[1].strip() if len(parts) > 1 else dataset_ref
        size = parts[2].strip() if len(parts) > 2 else ""
        last_updated = parts[3].strip() if len(parts) > 3 else ""
        download_count = parts[4].strip() if len(parts) > 4 else 0
        vote_count = parts[5].strip() if len(parts) > 5 else 0
        usability_rating = parts[6].strip() if len(parts) > 6 else ""

        datasets.append(
            {
                "ref": dataset_ref,
                "title": title,
                "size": size,
                "last_updated": last_updated,
                "download_count": download_count,
                "vote_count": vote_count,
                "usability_rating": usability_rating,
                "url": f"https://www.kaggle.com/datasets/{dataset_ref}",
            }
        )

        if len(datasets) >= limit:
            break

    if not datasets:
        raise RuntimeError("Kaggle 검색 결과가 없습니다.")

    return datasets


def _unzip_downloads(raw_dir: Path) -> None:
    for zip_path in raw_dir.glob("*.zip"):
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(raw_dir)


def _find_first_csv(raw_dir: Path) -> Path:
    csv_files = sorted(raw_dir.rglob("*.csv"))
    if not csv_files:
        raise RuntimeError("다운로드한 Kaggle 데이터셋에서 CSV 파일을 찾지 못했습니다.")
    return csv_files[0]


def preprocess_csv(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    before_rows = len(dataframe)
    before_columns = len(dataframe.columns)
    before_missing = int(dataframe.isna().sum().sum())

    processed = dataframe.copy()
    processed.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in processed.columns
    ]
    processed = processed.drop_duplicates()

    zero_as_missing = {}
    normalized_columns = {column.lower().replace("_", "") for column in processed.columns}
    is_heart_dataset = {"age", "cholesterol", "maxhr", "heartdisease"}.issubset(normalized_columns)
    if is_heart_dataset:
        for column in ("cholesterol", "restingbp"):
            if column in processed.columns:
                numeric_series = pd.to_numeric(processed[column], errors="coerce")
                zero_count = int((numeric_series == 0).sum())
                if zero_count:
                    processed[column] = numeric_series.mask(numeric_series == 0)
                    zero_as_missing[column] = zero_count

    for column in processed.columns:
        if pd.api.types.is_numeric_dtype(processed[column]):
            processed[column] = processed[column].fillna(processed[column].median())
            continue
        processed[column] = processed[column].fillna("missing")
        processed[column] = processed[column].astype(str).str.strip()

    after_missing = int(processed.isna().sum().sum())
    summary = {
        "before_rows": before_rows,
        "after_rows": len(processed),
        "before_columns": before_columns,
        "after_columns": len(processed.columns),
        "before_missing": before_missing,
        "after_missing": after_missing,
        "dropped_duplicates": before_rows - len(processed),
        "zero_as_missing": zero_as_missing,
        "numeric_columns": processed.select_dtypes(include="number").columns.tolist(),
        "categorical_columns": processed.select_dtypes(exclude="number").columns.tolist(),
    }
    return processed, summary


def download_and_preprocess_dataset(dataset_id: str) -> dict:
    dataset_id = normalize_dataset_id(dataset_id)
    if "/" not in dataset_id:
        raise RuntimeError("Kaggle dataset id는 예: uciml/iris 형식이어야 합니다.")

    dataset_slug = slugify_dataset_id(dataset_id)
    raw_dir = RAW_ROOT / dataset_slug
    processed_dir = PROCESSED_ROOT / dataset_slug

    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    if processed_dir.exists():
        shutil.rmtree(processed_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    _run_kaggle_download(dataset_id, raw_dir)
    _unzip_downloads(raw_dir)

    raw_csv_path = _find_first_csv(raw_dir)
    dataframe = pd.read_csv(raw_csv_path)
    processed_dataframe, preprocess_summary = preprocess_csv(dataframe)

    processed_csv_path = processed_dir / f"{dataset_slug}_processed.csv"
    metadata_path = processed_dir / "metadata.json"
    processed_dataframe.to_csv(processed_csv_path, index=False)
    raw_storage_uri = maybe_upload_to_object_storage(raw_csv_path)
    processed_storage_uri = maybe_upload_to_object_storage(processed_csv_path)

    metadata = {
        "source": "kaggle",
        "dataset_id": dataset_id,
        "url": f"https://www.kaggle.com/datasets/{dataset_id}",
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "raw_file": os.path.relpath(raw_csv_path, PROJECT_ROOT),
        "processed_file": os.path.relpath(processed_csv_path, PROJECT_ROOT),
        "raw_storage": build_storage_metadata(raw_csv_path, raw_storage_uri),
        "processed_storage": build_storage_metadata(processed_csv_path, processed_storage_uri),
        "preprocess": preprocess_summary,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata_storage_uri = maybe_upload_to_object_storage(metadata_path)
    metadata["metadata_storage"] = build_storage_metadata(metadata_path, metadata_storage_uri)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if metadata_storage_uri:
        maybe_upload_to_object_storage(metadata_path)

    return {
        "dataset_id": dataset_id,
        "document_title": processed_csv_path.name,
        "processed_csv": processed_dataframe.to_csv(index=False),
        "processed_path": str(processed_csv_path),
        "metadata_path": str(metadata_path),
        "storage_uri": processed_storage_uri,
        "metadata": metadata,
    }
