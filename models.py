from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import relationship

from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    created_at = Column(DateTime, server_default=func.now())


class Category(Base):
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    documents = relationship("Document", back_populates="category")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    display_name = Column(String(200), nullable=True)
    content = Column(Text, nullable=False)
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=True)
    source_type = Column(String(50), nullable=True)
    source_url = Column(String(500), nullable=True)
    file_path = Column(String(500), nullable=True)
    processed_path = Column(String(500), nullable=True)
    metadata_path = Column(String(500), nullable=True)
    storage_uri = Column(String(700), nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    category = relationship("Category", back_populates="documents")


class ModelResult(Base):
    __tablename__ = "model_results"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    target_column = Column(String(100), nullable=False)
    task_type = Column(String(50), nullable=True)
    best_model = Column(String(100), nullable=True)
    primary_metric = Column(String(100), nullable=True)
    valid_score = Column(String(50), nullable=True)
    metrics_json = Column(LONGTEXT, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    document = relationship("Document")
