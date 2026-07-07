import os

from database import Base, SessionLocal, engine
from kaggle_pipeline import download_and_preprocess_dataset, normalize_dataset_id
from models import Category, Document
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


def run_batch():
    Base.metadata.create_all(bind=engine)
    ensure_pipeline_schema()
    db = SessionLocal()
    try:
        category = get_or_create_category(db, os.getenv("BATCH_CATEGORY_NAME", "배치 수집 데이터"))
        for dataset_id in configured_datasets():
            document = upsert_kaggle_document(db, normalize_dataset_id(dataset_id), category.id)
            print(f"processed dataset={dataset_id} document_id={document.id} path={document.processed_path}")
    finally:
        db.close()


if __name__ == "__main__":
    run_batch()
