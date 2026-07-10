import os
from datetime import datetime
from types import SimpleNamespace

from database import Base, SessionLocal, engine
from kaggle_pipeline import download_and_preprocess_dataset, normalize_dataset_id
from models import Category, Document
from public_data_pipeline import PublicDataPipelineError, collect_public_data
from rag import upsert_document
from sqlalchemy import inspect, text


DEFAULT_DATASETS = [
    "blastchar/telco-customer-churn",
]


def configured_datasets() -> list[str]:
    raw_value = os.getenv("BATCH_KAGGLE_DATASETS", "").strip()
    if not raw_value:
        return DEFAULT_DATASETS
    return [dataset.strip() for dataset in raw_value.split(",") if dataset.strip()]


def get_or_create_category(db, name: str):
    category = db.query(Category).filter(Category.name == name).first()
    if category:
        return category
    category = Category(name=name)
    db.add(category)
    db.commit()
    db.refresh(category)
    return category


def ensure_pipeline_schema():
    inspector = inspect(engine)
    if "documents" not in inspector.get_table_names():
        return

    existing_columns = {column["name"] for column in inspector.get_columns("documents")}
    document_columns = {
        "display_name": "VARCHAR(200) NULL",
        "source_type": "VARCHAR(50) NULL",
        "source_url": "VARCHAR(500) NULL",
        "file_path": "VARCHAR(500) NULL",
        "processed_path": "VARCHAR(500) NULL",
        "metadata_path": "VARCHAR(500) NULL",
        "storage_uri": "VARCHAR(700) NULL",
    }
    with engine.begin() as connection:
        for column_name, column_type in document_columns.items():
            if column_name not in existing_columns:
                connection.execute(text(f"ALTER TABLE documents ADD COLUMN {column_name} {column_type}"))


def upsert_kaggle_document(db, dataset_id: str, category_id: int):
    result = download_and_preprocess_dataset(dataset_id)
    document = (
        db.query(Document)
        .filter(Document.source_type == "kaggle")
        .filter(Document.source_url == f"https://www.kaggle.com/datasets/{result['dataset_id']}")
        .first()
    )
    if document is None:
        document = Document(category_id=category_id)
        db.add(document)

    document.title = result["document_title"]
    document.content = result["processed_csv"]
    document.source_type = "kaggle"
    document.source_url = f"https://www.kaggle.com/datasets/{result['dataset_id']}"
    document.processed_path = result["processed_path"]
    document.metadata_path = result["metadata_path"]
    document.storage_uri = result.get("storage_uri") or ""
    db.commit()
    db.refresh(document)
    upsert_document(document)
    return document


def public_data_enabled() -> bool:
    return bool(os.getenv("PUBLIC_DATA_API_URL", "").strip())


def upsert_public_data_document(db):
    result = collect_public_data()
    category = get_or_create_category(db, result["category_name"])
    document = (
        db.query(Document)
        .filter(Document.source_type == "public_data")
        .filter(Document.source_url == result["source_url"])
        .filter(Document.title == result["document_title"])
        .first()
    )
    if document is None:
        document = Document(category_id=category.id)
        db.add(document)

    document.title = result["document_title"]
    document.display_name = result.get("display_name") or result["dataset_name"]
    document.content = result["processed_csv"]
    document.category_id = category.id
    document.source_type = "public_data"
    document.source_url = result["source_url"]
    document.file_path = result["file_path"]
    document.processed_path = result["processed_path"]
    document.metadata_path = result["metadata_path"]
    document.storage_uri = result.get("storage_uri") or ""
    db.commit()
    db.refresh(document)

    if os.getenv("PUBLIC_DATA_ENABLE_RAG", "false").strip().lower() in {"1", "true", "yes", "y"}:
        rag_document = SimpleNamespace(
            id=document.id,
            title=document.title,
            content=result["rag_content"],
            category_id=document.category_id,
        )
        upsert_document(rag_document)

    return document, result


def _log(message: str):
    print(message, flush=True)


def run_batch():
    Base.metadata.create_all(bind=engine)
    ensure_pipeline_schema()
    db = SessionLocal()
    success_count = 0
    failed_count = 0
    _log(f"[BATCH START] time={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        category = get_or_create_category(db, os.getenv("BATCH_CATEGORY_NAME", "배치 수집 데이터"))
        for dataset_id in configured_datasets():
            normalized_dataset_id = normalize_dataset_id(dataset_id)
            _log(f"[START] source=kaggle dataset={normalized_dataset_id}")
            try:
                document = upsert_kaggle_document(db, normalized_dataset_id, category.id)
                success_count += 1
                _log(
                    "[SUCCESS] "
                    f"source=kaggle dataset={normalized_dataset_id} "
                    f"document_id={document.id} path={document.processed_path}"
                )
            except Exception as error:
                db.rollback()
                failed_count += 1
                _log(f"[FAILED] source=kaggle dataset={normalized_dataset_id} error={error}")

        if public_data_enabled():
            dataset_name = os.getenv("PUBLIC_DATA_DATASET_NAME", "public-statistics").strip() or "public-statistics"
            _log(f"[START] source=public_data dataset={dataset_name}")
            try:
                document, result = upsert_public_data_document(db)
                success_count += 1
                _log(
                    "[SUCCESS] "
                    f"source=public_data dataset={dataset_name} document_id={document.id} "
                    f"rows_before={result['rows_before']} rows_after={result['rows_after']} "
                    f"path={document.processed_path}"
                )
            except (PublicDataPipelineError, Exception) as error:
                db.rollback()
                failed_count += 1
                _log(f"[FAILED] source=public_data dataset={dataset_name} error={error}")
        else:
            _log("[SKIP] source=public_data reason=PUBLIC_DATA_API_URL is not set")
    finally:
        _log(
            "[BATCH END] "
            f"time={datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            f"success={success_count} failed={failed_count}"
        )
        db.close()


if __name__ == "__main__":
    run_batch()
