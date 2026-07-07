import os
import base64
import json
import re
import warnings
from io import BytesIO, StringIO
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from database import Base, engine, get_db
from crawler_pipeline import crawl_web_page
from kaggle_pipeline import download_and_preprocess_dataset, search_kaggle_datasets
from models import Category, Document, ModelResult, User
from rag import ask_rag, upsert_document
from storage import maybe_upload_to_object_storage

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

Base.metadata.create_all(bind=engine)


def ensure_pipeline_schema():
    inspector = inspect(engine)
    if "documents" in inspector.get_table_names():
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


ensure_pipeline_schema()

HEART_DATASET_COLUMNS = {
    "age",
    "sex",
    "chestpaintype",
    "restingbp",
    "cholesterol",
    "fastingbs",
    "restingecg",
    "maxhr",
    "exerciseangina",
    "oldpeak",
    "st_slope",
    "heartdisease",
}

CHURN_DATASET_COLUMNS = {
    "age",
    "gender",
    "calls",
    "internetusage",
    "monthlycharge",
    "customersupportcalls",
    "contractlength",
    "datausage",
    "smsusage",
    "paymentmethod",
    "region",
    "churn",
}

CHURN_ONE_HOT_BASELINES = {
    "payment_method": "Bank Transfer",
    "region": "East",
}

CHURN_GENDER_ENCODING = {
    "0": "Female",
    "1": "Male",
}

DEFAULT_CHURN_DATASET_TITLE = "customer_churn.csv"
DEFAULT_CHURN_DATASET_PATH = Path(__file__).resolve().parent / "data" / DEFAULT_CHURN_DATASET_TITLE
DEFAULT_CHURN_CATEGORY_NAME = "기본 고객이탈 데이터"
LEGACY_DEFAULT_CHURN_DATASET_TITLES = {"default_customer_churn.csv"}


def normalized_column_name(column):
    return str(column).lower().replace("_", "").replace(" ", "")


def parse_datetime_series(series, pd):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(series, errors="coerce")


def is_heart_dataset(dataframe=None, title=""):
    title_text = (title or "").lower()
    title_match = any(keyword in title_text for keyword in ("heart", "cardio", "심장"))
    if dataframe is None:
        return title_match

    normalized_columns = {column.lower().replace("_", "") for column in dataframe.columns}
    column_hits = len(HEART_DATASET_COLUMNS & normalized_columns)
    return title_match or column_hits >= 6


def is_churn_dataset(dataframe=None, title=""):
    title_text = (title or "").lower()
    title_match = any(keyword in title_text for keyword in ("churn", "customer_churn", "고객", "이탈"))
    if dataframe is None:
        return title_match

    normalized_columns = {normalized_column_name(column) for column in dataframe.columns}
    column_hits = len(CHURN_DATASET_COLUMNS & normalized_columns)
    return title_match or column_hits >= 6


def dataset_column(dataframe, normalized_name):
    target = normalized_column_name(normalized_name)
    for column in dataframe.columns:
        if normalized_column_name(column) == target:
            return column
    return None


def humanize_filename_title(title):
    stem = Path(str(title or "데이터")).stem
    stem = re.sub(r"_processed$", "", stem, flags=re.IGNORECASE)
    stem = stem.replace("__", "_").replace("-", " ").replace("_", " ")
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem or str(title or "데이터")


def infer_display_name(title="", content="", dataframe=None):
    title_text = str(title or "")
    normalized_title = normalized_column_name(title_text)
    columns_text = ""
    if dataframe is not None:
        columns_text = " ".join(str(column) for column in dataframe.columns)
    normalized_columns = normalized_column_name(columns_text)
    combined = f"{normalized_title} {normalized_columns}"

    if "churn" in combined or "customerchurn" in combined or "고객이탈" in combined:
        return "고객 이탈 데이터"
    if "heart" in combined or "cardio" in combined or "heartdisease" in combined or "심장" in combined:
        return "심장 질환 데이터"
    if (
        ("child" in combined or "children" in combined or "아동" in combined)
        and ("mother" in combined or "mom" in combined or "어머니" in combined)
        and "iq" in combined
    ):
        return "아동-어머니 IQ 데이터"

    readable = humanize_filename_title(title_text)
    readable = re.sub(r"\b(csv|data|dataset|processed)\b", "", readable, flags=re.IGNORECASE)
    readable = re.sub(r"\s+", " ", readable).strip()
    if readable:
        return f"{readable} 데이터"
    return "CSV 데이터"


def display_document_title(document):
    if not document:
        return ""
    display_name = getattr(document, "display_name", None)
    if display_name:
        return display_name

    dataframe = None
    if str(document.title or "").lower().endswith(".csv") and document.content:
        try:
            import pandas as pd

            dataframe = pd.read_csv(StringIO(document.content), nrows=20)
        except Exception:
            dataframe = None
    return infer_display_name(document.title, document.content or "", dataframe)


def attach_display_titles(documents):
    for document in documents or []:
        try:
            document.display_title = display_document_title(document)
        except Exception:
            document.display_title = getattr(document, "display_name", None) or document.title
    return documents


def infer_display_name_from_csv_content(title, content):
    dataframe = None
    if str(title or "").lower().endswith(".csv") and content:
        try:
            import pandas as pd

            dataframe = pd.read_csv(StringIO(content), nrows=50)
        except Exception:
            dataframe = None
    return infer_display_name(title, content or "", dataframe)


def heart_column(dataframe, normalized_name):
    return dataset_column(dataframe, normalized_name)


def churn_column(dataframe, normalized_name):
    return dataset_column(dataframe, normalized_name)


def one_hot_category_series(dataframe, prefix, pd, baseline_label="기준 범주"):
    matching_columns = [
        column
        for column in dataframe.columns
        if normalized_column_name(column).startswith(normalized_column_name(prefix))
    ]
    if len(matching_columns) < 2:
        return None

    one_hot_frame = dataframe[matching_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    labels = []
    normalized_prefix = normalized_column_name(prefix)
    for _, row in one_hot_frame.iterrows():
        if row.max() <= 0:
            labels.append(baseline_label)
            continue
        selected_column = row.idxmax()
        selected_label = selected_column
        if normalized_column_name(selected_column).startswith(normalized_prefix):
            selected_label = selected_column[len(prefix):].lstrip("_ ")
        labels.append(selected_label or selected_column)
    return pd.Series(labels, index=dataframe.index, name=prefix.rstrip("_"))


def heart_numeric_columns(dataframe):
    preferred_columns = ["Age", "MaxHR", "Oldpeak", "Cholesterol"]
    return [
        column
        for column in [heart_column(dataframe, preferred_column) for preferred_column in preferred_columns]
        if column and column in dataframe.columns
    ]


def build_heart_feature_guide(document):
    if not document:
        return []

    content_preview = (document.content or "")[:1000].lower().replace("_", "")
    looks_like_heart = is_heart_dataset(title=document.title) or (
        "heartdisease" in content_preview and "cholesterol" in content_preview
    )
    if not looks_like_heart:
        return []

    return [
        {"feature": "age", "meaning": "나이", "usage": "연령대별 심장질환 발생 경향 확인"},
        {"feature": "restingbp", "meaning": "안정 시 혈압", "usage": "혈압 분포와 이상치 확인"},
        {"feature": "cholesterol", "meaning": "콜레스테롤 수치", "usage": "0값 보정 및 질병 발생군/비발생군 비교"},
        {"feature": "fastingbs", "meaning": "공복 혈당 여부", "usage": "혈당 조건에 따른 발생 비율 비교"},
        {"feature": "restingecg", "meaning": "안정 심전도 결과", "usage": "심전도 그룹별 발생 차이 확인"},
        {"feature": "maxhr", "meaning": "최대 심박수", "usage": "운동 능력과 발생 여부 관계 확인"},
        {"feature": "exerciseangina", "meaning": "운동 유발 협심증 여부", "usage": "운동 중 흉통 여부별 발생 비율 비교"},
        {"feature": "oldpeak", "meaning": "운동 후 ST depression", "usage": "심전도 변화량 분포 비교"},
        {"feature": "chestpaintype", "meaning": "흉통 유형", "usage": "흉통 유형별 HeartDisease 발생 비율"},
        {"feature": "st_slope", "meaning": "ST 구간 기울기", "usage": "핵심 범주형 변수로 발생 비율 비교"},
        {"feature": "heartdisease", "meaning": "심장질환 발생 여부", "usage": "예측 target"},
    ]


def build_churn_feature_guide(document):
    if not document:
        return []

    content_preview = (document.content or "")[:1400].lower().replace("_", "")
    looks_like_churn = is_churn_dataset(title=document.title) or (
        "churn" in content_preview and "monthlycharge" in content_preview
    )
    if not looks_like_churn:
        return []

    return [
        {"feature": "age", "meaning": "고객 나이", "usage": "연령대별 이탈 경향 확인"},
        {"feature": "gender", "meaning": "고객 성별", "usage": "원본은 Male/Female 범주형이며 현재 CSV에서는 0=Female, 1=Male로 변환됨"},
        {"feature": "calls", "meaning": "월간 통화량", "usage": "서비스 사용량과 이탈 여부 관계 확인"},
        {"feature": "internet_usage", "meaning": "인터넷 사용 시간", "usage": "인터넷 이용 강도와 유지/이탈 패턴 비교"},
        {"feature": "monthly_charge", "meaning": "월 요금", "usage": "요금 부담이 이탈 가능성에 미치는 영향 확인"},
        {"feature": "customer_support_calls", "meaning": "고객센터 문의 횟수", "usage": "탐색적 비교 변수. 현재 데이터에서는 핵심 유의 변수로 단정하지 않음"},
        {"feature": "contract_length", "meaning": "계약 기간", "usage": "탐색적 비교 변수. 현재 데이터에서는 핵심 유의 변수로 단정하지 않음"},
        {"feature": "data_usage", "meaning": "데이터 사용량", "usage": "데이터 사용 패턴과 이탈 여부 비교"},
        {"feature": "sms_usage", "meaning": "문자 사용량", "usage": "부가 서비스 이용량 분석"},
        {"feature": "payment_method", "meaning": "결제 방식", "usage": "원본 범주는 Credit Card / Bank Transfer / PayPal, 현재 CSV는 drop-first 원-핫 인코딩"},
        {"feature": "region", "meaning": "고객 거주 지역", "usage": "원본 범주는 North / South / East / West, 현재 CSV는 drop-first 원-핫 인코딩"},
        {"feature": "churn", "meaning": "고객 이탈 여부", "usage": "이진 분류 모델의 target"},
    ]


def build_feature_guide(document):
    return build_churn_feature_guide(document) or build_heart_feature_guide(document)


def build_churn_schema_info(document):
    if not document or not is_churn_dataset(title=document.title):
        return {}

    original_schema = [
        {"column": "age", "type": "수치형", "meaning": "고객 나이"},
        {"column": "gender", "type": "범주형", "meaning": "Male / Female"},
        {"column": "calls", "type": "수치형", "meaning": "월간 통화량"},
        {"column": "internet_usage", "type": "수치형", "meaning": "인터넷 사용 시간"},
        {"column": "monthly_charge", "type": "수치형", "meaning": "월 요금"},
        {"column": "customer_support_calls", "type": "수치형", "meaning": "고객센터 문의 횟수"},
        {"column": "contract_length", "type": "수치형", "meaning": "계약 기간"},
        {"column": "data_usage", "type": "수치형", "meaning": "데이터 사용량"},
        {"column": "sms_usage", "type": "수치형", "meaning": "문자 사용량"},
        {"column": "payment_method", "type": "범주형", "meaning": "Credit Card / Bank Transfer / PayPal"},
        {"column": "region", "type": "범주형", "meaning": "North / South / East / West"},
        {"column": "churn", "type": "Target", "meaning": "0=유지, 1=이탈"},
    ]
    model_schema = [
        {"column": "gender", "type": "인코딩된 범주형", "meaning": "0=Female, 1=Male"},
        {"column": "payment_method_Credit Card", "type": "원-핫", "meaning": "1이면 Credit Card, 두 payment_method 컬럼이 모두 0이면 Bank Transfer"},
        {"column": "payment_method_PayPal", "type": "원-핫", "meaning": "1이면 PayPal, 두 payment_method 컬럼이 모두 0이면 Bank Transfer"},
        {"column": "region_North", "type": "원-핫", "meaning": "1이면 North, 세 region 컬럼이 모두 0이면 East"},
        {"column": "region_South", "type": "원-핫", "meaning": "1이면 South, 세 region 컬럼이 모두 0이면 East"},
        {"column": "region_West", "type": "원-핫", "meaning": "1이면 West, 세 region 컬럼이 모두 0이면 East"},
    ]
    encoding_maps = [
        {"variable": "gender", "encoded": "0", "original": "Female"},
        {"variable": "gender", "encoded": "1", "original": "Male"},
        {"variable": "payment_method", "encoded": "Credit Card=0, PayPal=0", "original": "Bank Transfer"},
        {"variable": "payment_method", "encoded": "payment_method_Credit Card=1", "original": "Credit Card"},
        {"variable": "payment_method", "encoded": "payment_method_PayPal=1", "original": "PayPal"},
        {"variable": "region", "encoded": "North=0, South=0, West=0", "original": "East"},
        {"variable": "region", "encoded": "region_North=1", "original": "North"},
        {"variable": "region", "encoded": "region_South=1", "original": "South"},
        {"variable": "region", "encoded": "region_West=1", "original": "West"},
    ]
    return {
        "original_schema": original_schema,
        "model_schema": model_schema,
        "encoding_maps": encoding_maps,
        "note": "수업 자료의 원본 스키마는 범주형 변수를 포함하지만, 현재 업로드된 CSV는 모델 학습용으로 gender 라벨 인코딩과 payment_method/region drop-first 원-핫 인코딩이 이미 적용된 파일입니다.",
    }


def build_churn_preprocess_mapping_rows():
    return [
        {
            "original": "gender",
            "processed": "gender",
            "method": "Label Encoding",
            "baseline": "0=Female, 1=Male",
        },
        {
            "original": "payment_method",
            "processed": "payment_method_Credit Card, payment_method_PayPal",
            "method": "One-Hot Encoding(drop-first)",
            "baseline": "기준범주=Bank Transfer",
        },
        {
            "original": "region",
            "processed": "region_North, region_South, region_West",
            "method": "One-Hot Encoding(drop-first)",
            "baseline": "기준범주=East",
        },
        {
            "original": "age, calls, internet_usage, monthly_charge, customer_support_calls, contract_length, data_usage, sms_usage",
            "processed": "동일 변수명 유지 후 모델 학습 단계에서 수치형 feature로 사용",
            "method": "Numeric Type Check + Standard Scaling",
            "baseline": "StandardScaler는 train 기준으로 fit, validation에는 transform만 적용",
        },
        {
            "original": "churn",
            "processed": "churn",
            "method": "Target 유지",
            "baseline": "0=유지, 1=이탈",
        },
    ]


def build_analysis_header(document, view_name):
    if not document:
        return {}
    display_title = display_document_title(document)

    if is_churn_dataset(title=document.title):
        view_titles = {
            "select": "고객 이탈 예측 데이터 분석 선택",
            "preprocess": "고객 이탈 데이터 전처리",
            "eda": "고객 이탈 데이터 EDA",
            "modeling": "고객 이탈 데이터 모델링",
        }
        return {
            "title": view_titles.get(view_name, "고객 이탈 예측 데이터 분석"),
            "subtitle": (
                "고객 이탈 데이터는 churn을 target으로 두고 나이, 요금, 사용량, 고객센터 문의, 계약 기간, 결제 방식, 지역 등의 "
                "feature를 활용해 서비스 유지/이탈 가능성을 분석하는 이진 분류 웹서비스 주제입니다."
            ),
        }

    if not is_heart_dataset(title=document.title):
        suffix_map = {
            "select": "분석 선택",
            "preprocess": "전처리",
            "eda": "EDA",
            "modeling": "모델링",
        }
        return {
            "title": f"{display_title} {suffix_map.get(view_name, '분석')}",
            "subtitle": "CSV 컬럼 구조를 자동으로 분석해 가능한 전처리, EDA, 모델링 항목을 생성합니다.",
        }

    view_titles = {
        "select": "심장질환 예측 데이터 분석 선택",
        "preprocess": "심장질환 예측 데이터 전처리",
        "eda": "심장질환 예측 데이터 EDA",
    }
    return {
        "title": view_titles.get(view_name, "심장질환 예측 데이터 분석"),
        "subtitle": (
            f"원본 데이터셋 파일명은 {document.title}이지만, 화면에는 {display_title}로 표시합니다. 실제 예측 target은 HeartDisease입니다. "
            "따라서 서비스 제목과 분석 설명은 심장질환 예측으로 통일합니다."
        ),
    }


def normalize_search_tokens(text):
    normalized = text.lower()
    normalized = re.sub(r"\.(csv|txt|md|pdf)\b", " ", normalized)
    tokens = re.split(r"[^0-9a-z가-힣]+", normalized)
    stop_words = {
        "csv",
        "txt",
        "md",
        "pdf",
        "file",
        "data",
        "dataset",
        "train",
        "test",
        "about",
        "explain",
        "설명",
        "데이터",
        "파일",
        "문서",
        "대해",
        "대한",
        "관련",
    }
    token_set = {
        token
        for token in tokens
        if len(token) >= 2 and token not in stop_words and not token.isdigit()
    }
    synonyms = {
        "지하철": "subway",
        "전철": "subway",
        "아파트": "apartment",
        "부동산": "apartment",
        "가격": "price",
        "기상": "weather",
        "날씨": "weather",
        "심장": "heart",
        "질환": "disease",
    }
    for token, synonym in synonyms.items():
        if token in token_set:
            token_set.add(synonym)
        if synonym in token_set:
            token_set.add(token)

    return token_set


def find_title_matched_document_ids(question, documents):
    question_tokens = normalize_search_tokens(question)
    if not question_tokens:
        return []

    scored_documents = []
    for document in documents:
        title_tokens = normalize_search_tokens(document.title)
        if not title_tokens:
            continue

        exact_overlap = question_tokens & title_tokens
        partial_overlap = {
            question_token
            for question_token in question_tokens
            for title_token in title_tokens
            if question_token in title_token or title_token in question_token
        }
        score = (len(exact_overlap) * 2) + len(partial_overlap)
        if score:
            scored_documents.append((score, document.id))

    scored_documents.sort(reverse=True)
    return [document_id for _, document_id in scored_documents[:5]]


def dashboard_context(
    username,
    documents,
    categories=None,
    active_view="home",
    selected_category_id=None,
    keyword="",
    search_scope="all",
    rag_question="",
    rag_answer="",
    rag_sources=None,
    rag_error=None,
    error=None,
    message=None,
    pipeline_stats=None,
    eda_charts=None,
    csv_profiles=None,
    csv_profile=None,
    csv_documents=None,
    selected_document=None,
    target_column="",
    target_candidates=None,
    ml_result=None,
    preprocess_summary=None,
    kaggle_dataset_id="",
    kaggle_search_keyword="",
    kaggle_search_results=None,
    crawl_url="",
    source_info=None,
    feature_guide=None,
    schema_info=None,
    analysis_header=None,
):
    attach_display_titles(documents)
    attach_display_titles(csv_documents or [])
    if selected_document:
        attach_display_titles([selected_document])
    return {
        "username": username,
        "documents": documents,
        "categories": categories or [],
        "active_view": active_view,
        "selected_category_id": selected_category_id,
        "keyword": keyword,
        "search_scope": search_scope,
        "rag_question": rag_question,
        "rag_answer": rag_answer,
        "rag_sources": rag_sources or [],
        "rag_error": rag_error or error,
        "error": error or rag_error,
        "message": message,
        "pipeline_stats": pipeline_stats or {},
        "eda_charts": eda_charts or {},
        "csv_profiles": csv_profiles or [],
        "csv_profile": csv_profile or {},
        "csv_documents": csv_documents or [],
        "selected_document": selected_document,
        "target_column": target_column,
        "target_candidates": target_candidates or [],
        "ml_result": ml_result,
        "preprocess_summary": preprocess_summary,
        "kaggle_dataset_id": kaggle_dataset_id,
        "kaggle_search_keyword": kaggle_search_keyword,
        "kaggle_search_results": kaggle_search_results or [],
        "crawl_url": crawl_url,
        "source_info": source_info or {},
        "feature_guide": feature_guide or [],
        "schema_info": schema_info or {},
        "analysis_header": analysis_header or {},
        "default_churn_document": next(
            (
                document
                for document in documents
                if document.title == DEFAULT_CHURN_DATASET_TITLE
            ),
            None,
        ),
    }


def decode_text_file(file_bytes: bytes):
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue

    raise RuntimeError("텍스트 파일 인코딩을 읽을 수 없습니다.")


def ensure_default_churn_document(db: Session):
    if not DEFAULT_CHURN_DATASET_PATH.exists():
        return None

    csv_content = DEFAULT_CHURN_DATASET_PATH.read_text(encoding="utf-8-sig").strip()
    if not csv_content:
        return None

    category = (
        db.query(Category)
        .filter(Category.name == DEFAULT_CHURN_CATEGORY_NAME)
        .first()
    )
    if category is None:
        category = Category(name=DEFAULT_CHURN_CATEGORY_NAME)
        db.add(category)
        db.commit()
        db.refresh(category)

    document = (
        db.query(Document)
        .filter(Document.title == DEFAULT_CHURN_DATASET_TITLE)
        .first()
    )
    if document is None:
        document = (
            db.query(Document)
            .filter(Document.title.in_(LEGACY_DEFAULT_CHURN_DATASET_TITLES))
            .first()
        )

    if document is None:
        document = Document(
            title=DEFAULT_CHURN_DATASET_TITLE,
            display_name="고객 이탈 데이터",
            content=csv_content,
            category_id=category.id,
            source_type="seed_csv",
            source_url=str(DEFAULT_CHURN_DATASET_PATH),
            file_path=str(DEFAULT_CHURN_DATASET_PATH),
            processed_path=str(DEFAULT_CHURN_DATASET_PATH),
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        try:
            upsert_document(document)
        except Exception:
            pass
        return document

    changed = False
    if document.title != DEFAULT_CHURN_DATASET_TITLE:
        document.title = DEFAULT_CHURN_DATASET_TITLE
        changed = True
    if document.content != csv_content:
        document.content = csv_content
        changed = True
    if document.category_id != category.id:
        document.category_id = category.id
        changed = True
    if not getattr(document, "display_name", None):
        document.display_name = "고객 이탈 데이터"
        changed = True
    default_lineage = {
        "source_type": "seed_csv",
        "source_url": str(DEFAULT_CHURN_DATASET_PATH),
        "file_path": str(DEFAULT_CHURN_DATASET_PATH),
        "processed_path": str(DEFAULT_CHURN_DATASET_PATH),
    }
    for field_name, field_value in default_lineage.items():
        if not getattr(document, field_name):
            setattr(document, field_name, field_value)
            changed = True

    if changed:
        db.commit()
        db.refresh(document)
        try:
            upsert_document(document)
        except Exception:
            pass

    return document


def dashboard_records(db: Session):
    ensure_default_churn_document(db)
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    categories = db.query(Category).order_by(Category.name).all()
    return documents, categories


def save_public_data_file(filename: str, file_bytes: bytes):
    safe_name = os.path.basename(filename).replace("\\", "_").replace("/", "_")
    storage_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_files")
    os.makedirs(storage_dir, exist_ok=True)

    path = os.path.join(storage_dir, safe_name)
    name, extension = os.path.splitext(safe_name)
    counter = 1
    while os.path.exists(path):
        path = os.path.join(storage_dir, f"{name}_{counter}{extension}")
        counter += 1

    with open(path, "wb") as saved_file:
        saved_file.write(file_bytes)

    storage_uri = maybe_upload_to_object_storage(path)
    return {"file_path": path, "storage_uri": storage_uri}


def extract_pdf_text(file_bytes: bytes):
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError(
            "PDF 파일을 읽으려면 pypdf가 필요합니다. requirements.txt 설치를 확인하세요."
        ) from error

    reader = PdfReader(BytesIO(file_bytes))
    page_texts = []

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            page_texts.append(f"[페이지 {page_num:02d}]\n{page_text.strip()}")

    extracted_text = "\n\n".join(page_texts).strip()
    if not extracted_text:
        raise RuntimeError("PDF에서 텍스트를 추출하지 못했습니다.")

    return extracted_text


async def extract_upload_text(upload_file: UploadFile):
    file_bytes = await upload_file.read()
    filename = upload_file.filename or "uploaded-file"
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if not file_bytes:
        raise RuntimeError(f"{filename} 파일이 비어 있습니다.")

    saved_file = save_public_data_file(filename, file_bytes)

    if suffix == "pdf":
        return {
            "content": extract_pdf_text(file_bytes),
            "file_path": saved_file["file_path"],
            "storage_uri": saved_file["storage_uri"],
        }

    if suffix in {"txt", "md"}:
        extracted_text = decode_text_file(file_bytes).strip()
        if not extracted_text:
            raise RuntimeError(f"{filename} 파일에서 저장할 텍스트를 찾지 못했습니다.")
        return {
            "content": extracted_text,
            "file_path": saved_file["file_path"],
            "storage_uri": saved_file["storage_uri"],
        }

    if suffix == "csv":
        extracted_text = decode_text_file(file_bytes).strip()
        if not extracted_text:
            raise RuntimeError(f"{filename} CSV 파일에서 저장할 텍스트를 찾지 못했습니다.")
        return {
            "content": extracted_text,
            "file_path": saved_file["file_path"],
            "processed_path": saved_file["file_path"],
            "storage_uri": saved_file["storage_uri"],
        }

    raise RuntimeError(f"{filename} 파일 형식은 지원하지 않습니다. PDF, TXT, MD, CSV만 가능합니다.")


def build_pipeline_stats(documents, categories=None):
    total_chars = sum(len(document.content or "") for document in documents)
    return {
        "source_count": len(documents),
        "category_count": len(categories or []),
        "vector_count": len(documents),
        "table_count": 1,
        "csv_count": sum(1 for document in documents if document.title.lower().endswith(".csv")),
        "total_chars": total_chars,
    }


def find_metadata_for_document(document):
    project_root = Path(__file__).resolve().parent
    metadata_paths = [
        *(project_root / "data" / "processed" / "web").glob("**/metadata.json"),
        *(project_root / "data" / "processed" / "kaggle").glob("**/metadata.json"),
    ]

    for metadata_path in metadata_paths:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        processed_file = metadata.get("processed_file", "")
        processed_name = os.path.basename(processed_file)
        metadata_title = metadata.get("title", "")
        document_title_candidates = {
            processed_name,
            f"{metadata_title}.csv",
            f"{metadata_title}.txt",
        }

        if document.title in document_title_candidates:
            return metadata

    return None


def build_document_source_info(document):
    if not document:
        return {}

    if document.source_type:
        source_labels = {
            "manual": "직접 입력",
            "upload": "파일 업로드",
            "kaggle": "Kaggle 데이터셋",
            "crawl": "웹 크롤링",
            "seed_csv": "서비스 기본 CSV",
        }
        category_name = document.category.name if document.category else "미분류"
        created_at = document.created_at.strftime("%Y-%m-%d %H:%M") if document.created_at else "알 수 없음"
        storage_detail = ""
        if document.storage_uri:
            storage_detail = f" Object Storage URI: {document.storage_uri}"
        return {
            "label": source_labels.get(document.source_type, document.source_type),
            "origin": document.source_url or document.file_path or document.title,
            "detail": (
                "DB Document 레코드에 source_type, source_url, file_path, processed_path, metadata_path를 저장해 "
                f"수집부터 가공 산출물까지 추적할 수 있습니다.{storage_detail}"
            ),
            "category": category_name,
            "saved_at": created_at,
            "processed_file": document.processed_path or document.file_path or "",
        }

    if document.title == DEFAULT_CHURN_DATASET_TITLE:
        category_name = document.category.name if document.category else DEFAULT_CHURN_CATEGORY_NAME
        created_at = document.created_at.strftime("%Y-%m-%d %H:%M") if document.created_at else "앱 시작 시 자동 등록"
        return {
            "label": "서비스 기본 CSV",
            "origin": str(DEFAULT_CHURN_DATASET_PATH),
            "detail": (
                "앱에 포함된 고객 이탈 기본 CSV입니다. 사용자가 파일을 업로드하지 않아도 로그인 후 "
                "전처리, EDA, 모델링, 오즈비 분석 페이지가 이 데이터셋을 기준으로 바로 제공됩니다."
            ),
            "category": category_name,
            "saved_at": created_at,
            "processed_file": str(DEFAULT_CHURN_DATASET_PATH),
        }

    metadata = find_metadata_for_document(document)
    category_name = document.category.name if document.category else "미분류"
    created_at = document.created_at.strftime("%Y-%m-%d %H:%M") if document.created_at else "알 수 없음"

    if metadata:
        source = metadata.get("source", "")
        if source == "web":
            return {
                "label": "웹 크롤링",
                "origin": metadata.get("url", "알 수 없음"),
                "detail": "웹 페이지에서 HTML 표 또는 본문을 추출해 저장한 데이터입니다.",
                "category": category_name,
                "saved_at": created_at,
                "processed_file": metadata.get("processed_file", ""),
            }
        if source == "kaggle":
            dataset_id = metadata.get("dataset_id", "")
            detail = f"{dataset_id} 데이터셋을 다운로드하고 CSV 전처리 후 저장한 데이터입니다."
            if is_heart_dataset(title=document.title):
                detail = (
                    f"원본 Kaggle 데이터셋 파일명은 {dataset_id}이지만, 실제 예측 target은 HeartDisease입니다. "
                    "이 서비스에서는 heart failure가 아니라 심장질환 발생 여부 예측 데이터로 해석합니다."
                )
            return {
                "label": "Kaggle 데이터셋",
                "origin": metadata.get("url") or f"https://www.kaggle.com/datasets/{dataset_id}",
                "detail": detail,
                "category": category_name,
                "saved_at": created_at,
                "processed_file": metadata.get("processed_file", ""),
            }

    if document.title.endswith("_processed.csv"):
        dataset_id = document.title.removesuffix("_processed.csv").replace("__", "/")
        detail = f"{dataset_id} 데이터셋을 전처리해 저장한 CSV입니다."
        if is_heart_dataset(title=document.title):
            detail = (
                f"원본 Kaggle 데이터셋 파일명은 {dataset_id}이지만, 실제 예측 target은 HeartDisease입니다. "
                "이 서비스에서는 심장질환 발생 여부 예측 데이터로 해석합니다."
            )
        return {
            "label": "Kaggle 데이터셋",
            "origin": f"https://www.kaggle.com/datasets/{dataset_id}",
            "detail": detail,
            "category": category_name,
            "saved_at": created_at,
            "processed_file": "",
        }

    return {
        "label": "직접 업로드",
        "origin": document.title,
        "detail": (
            "사용자가 업로드한 고객 이탈 CSV를 DB에 저장하고 VectorDB에 색인한 데이터입니다. "
            "수집 단계는 로컬 CSV 업로드, 저장 단계는 애플리케이션 DB 저장, 제공 단계는 전처리/EDA/모델 결과 웹 페이지입니다."
            if is_churn_dataset(title=document.title)
            else "사용자가 업로드한 파일을 DB에 저장하고 VectorDB에 색인한 데이터입니다."
        ),
        "category": category_name,
        "saved_at": created_at,
        "processed_file": "",
    }


def make_chart_uri(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buffer.seek(0)
    return "data:image/png;base64," + base64.b64encode(buffer.read()).decode("ascii")


def build_eda_tabs(charts):
    tab_specs = [
        ("churn_target", "churn-target", "클래스 분포"),
        ("churn_box_age", "churn-age", "age"),
        ("churn_box_calls", "churn-calls", "calls"),
        ("churn_box_monthlycharge", "churn-monthly-charge", "monthly_charge"),
        ("churn_rate_paymentmethod", "churn-payment", "payment_method"),
        ("churn_rate_region", "churn-region", "region"),
        ("churn_logit_rows", "churn-odds", "odds ratio"),
        ("heart_target", "heart-target", "target 분포"),
        ("heart_boxplot_by_target", "heart-boxplot", "수치형 분포"),
        ("heart_category_risk", "heart-category-risk", "범주형 비율"),
        ("heart_st_slope_risk", "heart-st-slope", "ST slope"),
        ("generic_target_distribution", "generic-target", "target 후보 분포"),
        ("generic_numeric_distribution", "generic-numeric", "수치형 분포"),
        ("generic_outlier_boxplot", "generic-outlier", "이상치 후보"),
        ("generic_category_frequency", "generic-category", "범주형 빈도"),
        ("generic_target_numeric_relation", "generic-target-relation", "target별 차이"),
        ("csv_numeric", "csv-numeric-summary", "컬럼 분포"),
        ("csv_time", "csv-time", "날짜 추이"),
        ("csv_weekday", "csv-weekday", "요일 패턴"),
        ("csv_corr", "csv-corr", "상관관계"),
    ]
    return [
        {"key": key, "panel": panel, "label": label}
        for key, panel, label in tab_specs
        if charts.get(key)
    ]


def build_churn_logit_rows(dataframe, pd):
    target_column = churn_column(dataframe, "churn")
    if not target_column:
        return []

    try:
        import numpy as np
        from scipy.optimize import minimize
        from scipy.special import expit
        from scipy.stats import norm
    except Exception:
        return []

    preferred_columns = ["age", "calls", "monthly_charge"]
    feature_columns = [
        churn_column(dataframe, column)
        for column in preferred_columns
        if churn_column(dataframe, column)
    ]
    if not feature_columns:
        return []

    logit_frame = dataframe[[target_column] + feature_columns].copy()
    for column in [target_column] + feature_columns:
        logit_frame[column] = pd.to_numeric(logit_frame[column], errors="coerce")
    logit_frame = logit_frame.dropna()
    if len(logit_frame) < 20 or logit_frame[target_column].nunique() != 2:
        return []

    y = logit_frame[target_column].astype(float).to_numpy()
    X_raw = logit_frame[feature_columns].astype(float)
    X_scaled = (X_raw - X_raw.mean()) / X_raw.std(ddof=0).replace(0, 1)
    X = np.column_stack([np.ones(len(X_scaled)), X_scaled.to_numpy()])

    def negative_log_likelihood(beta):
        linear = X @ beta
        return -np.sum(y * np.log(expit(linear) + 1e-12) + (1 - y) * np.log(1 - expit(linear) + 1e-12))

    result = minimize(negative_log_likelihood, np.zeros(X.shape[1]), method="BFGS", options={"maxiter": 1000})
    if not result.success:
        return []

    beta = result.x
    probabilities = expit(X @ beta)
    weights = probabilities * (1 - probabilities)
    hessian = X.T @ (X * weights[:, None])
    covariance = np.linalg.pinv(hessian)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0))

    rows = []
    for index, column in enumerate(feature_columns, start=1):
        coefficient = beta[index]
        standard_error = standard_errors[index]
        z_value = coefficient / standard_error if standard_error else 0
        p_value = 2 * (1 - norm.cdf(abs(z_value)))
        odds_ratio = float(np.exp(coefficient))
        rows.append(
            {
                "feature": column,
                "coefficient": round(float(coefficient), 4),
                "odds_ratio": round(odds_ratio, 4),
                "p_value": round(float(p_value), 4),
                "significant": bool(p_value < 0.05),
            }
        )

    rows.sort(key=lambda row: row["p_value"])
    return rows


def read_csv_documents(documents, pd):
    csv_frames = []

    for document in documents:
        if not document.title.lower().endswith(".csv"):
            continue

        try:
            dataframe = pd.read_csv(StringIO(document.content))
        except Exception:
            continue

        dataframe = normalize_csv_dataframe_for_analysis(dataframe, pd)

        if dataframe.empty:
            continue

        csv_frames.append(
            {
                "title": document.title,
                "document": document,
                "dataframe": dataframe,
            }
        )

    return csv_frames


def normalize_csv_dataframe_for_analysis(dataframe, pd):
    normalized = dataframe.copy()

    for column in normalized.columns:
        if pd.api.types.is_numeric_dtype(normalized[column]):
            continue

        text_series = normalized[column].astype(str).str.strip()
        numeric_series = (
            text_series
            .str.replace(r"\[[^\]]*\]", "", regex=True)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace("−", "-", regex=False)
            .str.replace(r"^\s*$", "", regex=True)
        )
        converted_numeric = pd.to_numeric(numeric_series, errors="coerce")
        numeric_ratio = converted_numeric.notna().mean()

        if numeric_ratio >= 0.6:
            normalized[column] = converted_numeric
            continue

        converted_date = parse_datetime_series(text_series, pd)
        date_ratio = converted_date.notna().mean()
        if date_ratio >= 0.6:
            normalized[column] = converted_date

    return normalized


def is_probable_identifier(column_name, series):
    normalized_name = normalized_column_name(column_name)
    if normalized_name in {"id", "idx", "index", "no", "number", "seq"}:
        return True
    if normalized_name.endswith("id") or normalized_name.endswith("no"):
        return series.nunique(dropna=True) >= max(10, len(series) * 0.7)
    return series.nunique(dropna=True) == len(series) and len(series) > 20


def classify_csv_columns(dataframe, pd):
    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
    date_columns = []
    categorical_columns = []
    identifier_columns = []
    constant_columns = []

    for column in dataframe.columns:
        series = dataframe[column]
        unique_count = int(series.nunique(dropna=True))
        if unique_count <= 1:
            constant_columns.append(column)
        if is_probable_identifier(column, series):
            identifier_columns.append(column)

        if column in numeric_columns:
            continue

        parsed = parse_datetime_series(series, pd)
        if parsed.notna().mean() >= 0.6:
            date_columns.append(column)
        else:
            categorical_columns.append(column)

    target_candidates = recommend_target_columns(
        dataframe,
        pd,
        numeric_columns,
        categorical_columns,
        date_columns,
        identifier_columns,
        constant_columns,
    )

    column_rows = []
    for column in dataframe.columns:
        if column in identifier_columns:
            detected_type = "식별자"
            reason = "행을 구분하는 고유값 성격이 강해 feature/target에서 제외 후보입니다."
        elif column in constant_columns:
            detected_type = "상수"
            reason = "값이 거의 하나뿐이라 모델 학습 정보가 부족합니다."
        elif column in date_columns:
            detected_type = "날짜형"
            reason = "datetime으로 변환 가능한 값 비율이 높습니다."
        elif column in numeric_columns:
            detected_type = "수치형"
            reason = "분포, 이상치, 상관관계 분석에 사용합니다."
        else:
            detected_type = "범주형"
            reason = "빈도, 비율, target별 차이 분석에 사용합니다."
        column_rows.append(
            {
                "column": column,
                "type": detected_type,
                "missing": int(dataframe[column].isna().sum()),
                "unique": int(dataframe[column].nunique(dropna=True)),
                "reason": reason,
            }
        )

    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "date_columns": date_columns,
        "identifier_columns": identifier_columns,
        "constant_columns": constant_columns,
        "target_candidates": target_candidates,
        "column_rows": column_rows,
    }


def recommend_target_columns(
    dataframe,
    pd,
    numeric_columns,
    categorical_columns,
    date_columns,
    identifier_columns,
    constant_columns,
):
    target_keywords = {
        "target",
        "label",
        "class",
        "outcome",
        "result",
        "status",
        "flag",
        "return",
        "returned",
        "churn",
        "survived",
        "disease",
        "default",
        "fraud",
        "success",
        "price",
        "amount",
        "sales",
        "revenue",
        "score",
        "rating",
        "value",
    }
    candidates = []
    row_count = max(len(dataframe), 1)

    for index, column in enumerate(dataframe.columns):
        if column in date_columns or column in identifier_columns or column in constant_columns:
            continue
        series = dataframe[column].dropna()
        if series.empty:
            continue

        normalized_name = normalized_column_name(column)
        unique_count = int(series.nunique())
        unique_ratio = unique_count / row_count
        is_numeric = column in numeric_columns
        task_type = "회귀" if is_numeric and (unique_count > 10 or unique_ratio > 0.2) else "분류"
        score = 0
        reasons = []

        matched_keywords = [keyword for keyword in target_keywords if keyword in normalized_name]
        if normalized_name in {"y", "targety"}:
            matched_keywords.append("y")
        if matched_keywords:
            score += 8
            reasons.append(f"컬럼명에 target 후보 키워드({', '.join(matched_keywords[:3])})가 포함됨")
        if unique_count == 2:
            score += 5
            reasons.append("이진 분류 target으로 쓰기 좋은 2개 값")
        elif task_type == "분류" and 3 <= unique_count <= min(20, max(3, row_count // 3)):
            score += 3
            reasons.append(f"{unique_count}개 범주를 가진 분류 후보")
        elif task_type == "회귀":
            score += 2
            reasons.append("연속형 수치 예측 후보")
        if index == len(dataframe.columns) - 1:
            score += 2
            reasons.append("CSV의 마지막 컬럼")
        if unique_ratio > 0.8 and not matched_keywords:
            score -= 4
            reasons.append("고유값 비율이 높아 식별자일 가능성 있음")
        if unique_count < 2:
            score -= 10

        if score <= 0:
            continue
        candidates.append(
            {
                "column": column,
                "score": score,
                "task_type": task_type,
                "unique_count": unique_count,
                "missing_count": int(dataframe[column].isna().sum()),
                "reason": " / ".join(reasons) if reasons else "모델링 target 후보로 사용할 수 있는 값 구조",
                "recommended": False,
            }
        )

    candidates.sort(key=lambda row: row["score"], reverse=True)
    for row in candidates[:5]:
        row["recommended"] = row is candidates[0]
    return candidates[:8]


def auto_target_column(dataframe, pd):
    profile = classify_csv_columns(dataframe, pd)
    candidates = profile["target_candidates"]
    return candidates[0]["column"] if candidates else ""


def build_csv_profile(document, dataframe, pd):
    title = display_document_title(document)
    original_title = document.title
    column_profile = classify_csv_columns(dataframe, pd)
    numeric_columns = column_profile["numeric_columns"]
    date_columns = column_profile["date_columns"]
    missing_counts = dataframe.isna().sum()
    top_missing = [
        {"column": column, "count": int(count)}
        for column, count in missing_counts.sort_values(ascending=False).head(5).items()
        if int(count) > 0
    ]
    categorical_columns = column_profile["categorical_columns"]
    total_cells = len(dataframe) * len(dataframe.columns)
    missing_total = int(missing_counts.sum())
    missing_ratio = (missing_total / total_cells * 100) if total_cells else 0

    lower_columns = {column.lower(): column for column in dataframe.columns}
    lower_title = original_title.lower()
    if is_churn_dataset(dataframe, title):
        data_domain = "고객 이탈 예측 데이터"
        data_character = "고객의 이용량, 요금, 문의 횟수, 계약 기간, 결제 방식, 지역 같은 feature로 churn 여부를 분석하고 예측하는 데 적합합니다."
    elif is_heart_dataset(dataframe, title):
        data_domain = "심장질환 발생 예측 데이터"
        data_character = "환자의 나이, 흉통 유형, 콜레스테롤, 최대 심박수, 운동성 협심증 같은 검진 feature로 HeartDisease 발생 여부를 분석하고 예측하는 데 적합합니다."
    elif any(keyword in lower_title for keyword in ("weather", "기상")) or {
        "temperature",
        "precipitation",
        "visibility",
    }.intersection(lower_columns):
        data_domain = "기상 관측 데이터"
        data_character = "날짜, 관측 지점, 날씨 관련 수치가 함께 들어 있어 시간·지역별 기상 변화를 살펴보는 데 적합합니다."
    elif any(keyword in lower_title for keyword in ("subway", "지하철")):
        data_domain = "지하철 이용 데이터"
        data_character = "역이나 시간 단위의 이용 패턴을 비교하고 수요 변화를 분석하는 데 적합합니다."
    elif any(keyword in lower_title for keyword in ("apartment", "apt", "아파트", "price")):
        data_domain = "부동산 가격 데이터"
        data_character = "가격과 위치·면적 같은 설명 변수를 함께 보며 가격 흐름이나 영향 요인을 탐색하는 데 적합합니다."
    else:
        data_domain = "CSV 기반 공공데이터"
        data_character = "여러 행의 관측값과 변수로 구성되어 전체 분포, 결측치, 변수 관계를 탐색하는 데 적합합니다."

    if is_churn_dataset(dataframe, title):
        period_text = "고객별 서비스 이용 레코드 기반 데이터로, 날짜 컬럼 대신 고객 속성과 churn target을 포함합니다."
    elif is_heart_dataset(dataframe, title):
        period_text = "개별 환자 검진 레코드 기반 데이터로, 날짜 컬럼 대신 환자별 feature와 HeartDisease target을 포함합니다."
    else:
        period_text = "명확한 날짜 범위는 감지되지 않았습니다."
    if date_columns:
        date_column = date_columns[0]
        parsed_dates = parse_datetime_series(dataframe[date_column], pd).dropna()
        if not parsed_dates.empty:
            start_date = parsed_dates.min().date()
            end_date = parsed_dates.max().date()
            period_text = f"{date_column} 기준 {start_date}부터 {end_date}까지의 기간이 포함됩니다."
    elif "year" in lower_columns:
        year_values = pd.to_numeric(dataframe[lower_columns["year"]], errors="coerce").dropna()
        if not year_values.empty:
            period_text = f"year 기준 {int(year_values.min())}년부터 {int(year_values.max())}년까지의 값이 포함됩니다."

    category_name = document.category.name if document.category else "미분류"
    created_at = document.created_at.strftime("%Y-%m-%d %H:%M") if document.created_at else "알 수 없음"
    representative_columns = dataframe.columns.tolist()[:8]
    representative_text = ", ".join(representative_columns)
    if len(dataframe.columns) > len(representative_columns):
        representative_text = f"{representative_text} 외 {len(dataframe.columns) - len(representative_columns)}개"

    if missing_total:
        quality_text = f"전체 셀 중 결측치는 {missing_total}개로 약 {missing_ratio:.1f}%입니다."
    else:
        quality_text = "감지된 결측치는 없어 기본적인 데이터 완성도는 좋은 편입니다."

    churn_target = churn_column(dataframe, "churn")
    heart_target = heart_column(dataframe, "HeartDisease")
    if is_churn_dataset(dataframe, title) and churn_target:
        target_counts = dataframe[churn_target].value_counts(dropna=False).to_dict()
        quality_text = f"{quality_text} Target churn 클래스 분포는 {target_counts}입니다."
    elif is_heart_dataset(dataframe, title) and heart_target:
        target_counts = dataframe[heart_target].value_counts(dropna=False).to_dict()
        quality_text = f"{quality_text} Target HeartDisease 클래스 분포는 {target_counts}입니다."
    elif column_profile["target_candidates"]:
        candidate = column_profile["target_candidates"][0]
        quality_text = f"{quality_text} 모델링 target 후보로는 '{candidate['column']}' 컬럼을 우선 추천합니다. 추천 이유: {candidate['reason']}."

    overview = {
        "source": f"{title} 파일에서 읽어온 데이터입니다. 원본 파일명은 {original_title}이며 앱에는 '{category_name}' 카테고리로 저장되어 있습니다. 저장 시각은 {created_at}입니다.",
        "summary": f"{data_domain}로 보이며, 총 {len(dataframe)}개의 관측 행과 {len(dataframe.columns)}개의 변수로 구성되어 있습니다. {data_character}",
        "period": period_text,
        "structure": f"수치형 변수 {len(numeric_columns)}개, 범주/문자형 변수 {len(categorical_columns)}개, 날짜형 변수 {len(date_columns)}개가 감지되었습니다.",
        "columns": f"대표 컬럼은 {representative_text}입니다.",
        "quality": quality_text,
    }

    return {
        "title": title,
        "row_count": len(dataframe),
        "column_count": len(dataframe.columns),
        "columns": dataframe.columns.tolist(),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "date_columns": date_columns,
        "identifier_columns": column_profile["identifier_columns"],
        "target_candidates": column_profile["target_candidates"],
        "top_missing": top_missing,
        "overview": overview,
    }


def build_eda_charts(documents, include_storage_charts=True):
    matplotlib_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".matplotlib_cache")
    os.makedirs(matplotlib_cache, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", matplotlib_cache)

    try:
        import pandas as pd
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("EDA 시각화를 사용하려면 pandas와 matplotlib 설치가 필요합니다.") from error

    csv_frames = read_csv_documents(documents, pd)
    csv_profiles = [
        build_csv_profile(csv_file["document"], csv_file["dataframe"], pd)
        for csv_file in csv_frames
    ]

    plt.rcParams["font.family"] = ["Malgun Gothic", "Arial", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    charts = {}

    if include_storage_charts:
        rows = [
            {
                "title": document.title,
                "category": document.category.name if document.category else "미분류",
                "content_length": len(document.content or ""),
                "created_at": document.created_at,
            }
            for document in documents
        ]

        if rows:
            dataframe = pd.DataFrame(rows)
            dataframe["created_date"] = pd.to_datetime(dataframe["created_at"]).dt.date

            category_counts = dataframe.groupby("category").size().sort_values(ascending=False)
            fig_category, ax_category = plt.subplots(figsize=(7.2, 4.2))
            category_counts.plot(kind="bar", ax=ax_category, color="#4a6fe3")
            ax_category.set_title("카테고리별 수집 데이터 수")
            ax_category.set_xlabel("카테고리")
            ax_category.set_ylabel("건수")
            ax_category.grid(axis="y", alpha=0.22)
            ax_category.tick_params(axis="x", rotation=20)
            charts["category"] = make_chart_uri(fig_category)
            plt.close(fig_category)

            date_counts = dataframe.groupby("created_date").size()
            fig_time, ax_time = plt.subplots(figsize=(7.2, 4.2))
            date_counts.plot(kind="line", marker="o", ax=ax_time, color="#50c878", linewidth=2.5)
            ax_time.set_title("날짜별 저장 추이")
            ax_time.set_xlabel("저장일")
            ax_time.set_ylabel("건수")
            ax_time.grid(alpha=0.24)
            charts["time"] = make_chart_uri(fig_time)
            plt.close(fig_time)

            fig_length, ax_length = plt.subplots(figsize=(7.2, 4.2))
            dataframe["content_length"].plot(kind="hist", bins=min(8, max(3, len(dataframe))), ax=ax_length, color="#2c5282")
            ax_length.set_title("텍스트 길이 분포")
            ax_length.set_xlabel("문자 수")
            ax_length.set_ylabel("문서 수")
            ax_length.grid(axis="y", alpha=0.22)
            charts["length"] = make_chart_uri(fig_length)
            plt.close(fig_length)

    if csv_frames:
        first_csv = csv_frames[0]
        csv_title = first_csv["title"]
        csv_dataframe = first_csv["dataframe"]
        numeric_dataframe = csv_dataframe.select_dtypes(include="number")
        helper_numeric_names = {
            "id",
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "weekday",
            "day_of_week",
            "week",
            "quarter",
        }
        analysis_numeric_columns = [
            column
            for column in numeric_dataframe.columns
            if column.lower() not in helper_numeric_names
        ]
        analysis_numeric_dataframe = numeric_dataframe[analysis_numeric_columns] if analysis_numeric_columns else numeric_dataframe
        churn_dataset = is_churn_dataset(csv_dataframe, csv_title)
        churn_target = churn_column(csv_dataframe, "churn")
        heart_dataset = is_heart_dataset(csv_dataframe, csv_title)
        heart_target = heart_column(csv_dataframe, "HeartDisease")
        column_profile = classify_csv_columns(csv_dataframe, pd)
        target_candidates = column_profile["target_candidates"]
        generic_target = target_candidates[0]["column"] if target_candidates else ""
        charts["target_candidates"] = target_candidates
        charts["column_type_rows"] = column_profile["column_rows"]

        if churn_dataset and churn_target:
            churn_frame = csv_dataframe.copy()
            churn_frame[churn_target] = pd.to_numeric(churn_frame[churn_target], errors="coerce")
            target_counts = churn_frame[churn_target].value_counts().sort_index()
            fig_target, ax_target = plt.subplots(figsize=(7.8, 4.5))
            target_counts.plot(kind="bar", ax=ax_target, color=["#50c878", "#ef4444"])
            ax_target.set_title("churn 클래스 분포")
            ax_target.set_xlabel("churn (0=유지, 1=이탈)")
            ax_target.set_ylabel("고객 수")
            ax_target.grid(axis="y", alpha=0.22)
            ax_target.tick_params(axis="x", rotation=0)
            charts["churn_target"] = make_chart_uri(fig_target)
            churn_count = int(target_counts.get(1, 0))
            total_count = int(target_counts.sum())
            churn_ratio = (churn_count / total_count * 100) if total_count else 0
            charts["churn_target_note"] = f"전체 {total_count}명 중 churn=1 고객은 {churn_count}명이며 이탈률은 {churn_ratio:.1f}%입니다. 모델 학습 전 클래스 불균형을 확인합니다."
            plt.close(fig_target)

            logit_rows = build_churn_logit_rows(churn_frame, pd)
            if logit_rows:
                charts["churn_logit_rows"] = logit_rows
                significant_features = [row["feature"] for row in logit_rows if row["significant"]]
                if significant_features:
                    charts["churn_logit_note"] = (
                        f"현재 CSV 기준 로지스틱 회귀의 p-value 0.05 미만 변수는 {', '.join(significant_features)}입니다. "
                        "계수와 odds ratio는 수치형 변수를 표준화한 뒤 계산했으므로, EDA 해석은 이 유의 변수와 탐색용 변수를 구분해서 읽어야 합니다."
                    )
                else:
                    charts["churn_logit_note"] = "현재 CSV 기준 p-value 0.05 미만 변수는 없습니다. 그래프 패턴은 탐색용으로 해석합니다."

            preferred_box_columns = ["age", "calls", "monthly_charge"]
            boxplot_columns = [
                churn_column(csv_dataframe, column)
                for column in preferred_box_columns
                if churn_column(csv_dataframe, column)
            ]
            for column in boxplot_columns:
                boxplot_frame = churn_frame[[churn_target, column]].copy()
                boxplot_frame[column] = pd.to_numeric(boxplot_frame[column], errors="coerce")
                fig_box, ax_box = plt.subplots(figsize=(7.8, 4.5))
                boxplot_frame.boxplot(column=column, by=churn_target, ax=ax_box, grid=False)
                ax_box.set_title(f"이탈 여부별 {column} boxplot")
                ax_box.set_xlabel("churn")
                ax_box.set_ylabel(column)
                ax_box.grid(axis="y", alpha=0.22)
                fig_box.suptitle("")
                key = f"churn_box_{normalized_column_name(column)}"
                charts[key] = make_chart_uri(fig_box)
                if normalized_column_name(column) == "age":
                    charts[f"{key}_note"] = "age는 고객 특성별 이탈 차이를 확인하는 기본 변수입니다. 이탈 여부별 연령 분포가 한쪽으로 치우치는지 먼저 점검합니다."
                elif normalized_column_name(column) == "calls":
                    charts[f"{key}_note"] = "calls는 서비스 사용량을 나타내며, 로지스틱 회귀에서 오즈비 해석으로 연결하기 좋은 핵심 변수입니다."
                elif normalized_column_name(column) == "monthlycharge":
                    charts[f"{key}_note"] = "현재 데이터의 유의성 해석에서 monthly_charge는 핵심 확인 변수입니다. 박스플롯은 이탈 여부별 월 요금 분포 차이를 시각적으로 점검합니다."
                else:
                    charts[f"{key}_note"] = f"{column}는 이탈 여부별 분포 차이를 탐색적으로 확인하는 변수입니다."
                plt.close(fig_box)

            for preferred_category in ["payment_method", "region"]:
                category_column = churn_column(csv_dataframe, preferred_category)
                if category_column:
                    category_series = churn_frame[category_column]
                    category_label = category_column
                else:
                    category_series = one_hot_category_series(
                        churn_frame,
                        f"{preferred_category}_",
                        pd,
                        baseline_label=CHURN_ONE_HOT_BASELINES.get(preferred_category, "기준 범주"),
                    )
                    category_label = preferred_category
                if category_series is not None:
                    risk_frame = pd.DataFrame({
                        category_label: category_series,
                        churn_target: pd.to_numeric(churn_frame[churn_target], errors="coerce"),
                    })
                    risk_summary = risk_frame.groupby(category_label)[churn_target].mean().sort_values(ascending=False)
                    fig_risk, ax_risk = plt.subplots(figsize=(7.8, 4.5))
                    risk_summary.plot(kind="bar", ax=ax_risk, color="#4a6fe3")
                    ax_risk.set_title(f"{category_label}별 이탈률")
                    ax_risk.set_xlabel(category_label)
                    ax_risk.set_ylabel("이탈률")
                    ax_risk.set_ylim(0, 1)
                    ax_risk.grid(axis="y", alpha=0.22)
                    ax_risk.tick_params(axis="x", rotation=20)
                    key = f"churn_rate_{normalized_column_name(category_label)}"
                    charts[key] = make_chart_uri(fig_risk)
                    top_group = risk_summary.index[0]
                    charts[f"{key}_note"] = f"{category_label} 기준 이탈률이 가장 높은 그룹은 {top_group}입니다. 범주형 feature를 웹에서 직접 비교하기 좋습니다."
                    plt.close(fig_risk)

        if heart_dataset and heart_target:
            target_counts = csv_dataframe[heart_target].value_counts().sort_index()
            fig_target, ax_target = plt.subplots(figsize=(7.8, 4.5))
            target_counts.plot(kind="bar", ax=ax_target, color=["#38bdf8", "#ef4444"])
            ax_target.set_title("HeartDisease 발생 여부 클래스 분포")
            ax_target.set_xlabel("HeartDisease (0=비발생, 1=발생)")
            ax_target.set_ylabel("환자 수")
            ax_target.grid(axis="y", alpha=0.22)
            ax_target.tick_params(axis="x", rotation=0)
            charts["heart_target"] = make_chart_uri(fig_target)
            positive_count = int(target_counts.get(1, 0))
            total_count = int(target_counts.sum())
            positive_ratio = (positive_count / total_count * 100) if total_count else 0
            charts["heart_target_note"] = f"전체 {total_count}명 중 HeartDisease=1 환자는 {positive_count}명이며 비율은 {positive_ratio:.1f}%입니다. 모델 학습 전 클래스 불균형 여부를 확인하는 기준 그래프입니다."
            plt.close(fig_target)

            boxplot_columns = heart_numeric_columns(csv_dataframe)
            if boxplot_columns:
                boxplot_frame = csv_dataframe[[heart_target] + boxplot_columns].copy()
                boxplot_frame[heart_target] = pd.to_numeric(boxplot_frame[heart_target], errors="coerce")
                for column in boxplot_columns:
                    boxplot_frame[column] = pd.to_numeric(boxplot_frame[column], errors="coerce")
                    if column.lower().replace("_", "") in {"cholesterol", "restingbp"}:
                        boxplot_frame.loc[boxplot_frame[column] == 0, column] = pd.NA
                fig_heart_box, axes = plt.subplots(1, len(boxplot_columns), figsize=(3.0 * len(boxplot_columns), 4.4), squeeze=False)
                for axis, column in zip(axes[0], boxplot_columns):
                    boxplot_frame.boxplot(column=column, by=heart_target, ax=axis, grid=False)
                    axis.set_title(column)
                    axis.set_xlabel("HeartDisease")
                    axis.set_ylabel("값")
                    axis.grid(axis="y", alpha=0.22)
                fig_heart_box.suptitle("")
                fig_heart_box.tight_layout()
                charts["heart_boxplot_by_target"] = make_chart_uri(fig_heart_box)
                charts["heart_boxplot_by_target_note"] = (
                    f"{', '.join(boxplot_columns)} 분포를 HeartDisease 발생/미발생 그룹으로 나누어 비교했습니다. "
                    "평균/최댓값보다 그룹 간 중앙값, 사분위 범위, 이상치 차이를 더 직접적으로 확인할 수 있습니다."
                )
                plt.close(fig_heart_box)

            category_column = heart_column(csv_dataframe, "ChestPainType") or heart_column(csv_dataframe, "ExerciseAngina")
            if category_column:
                category_frame = csv_dataframe[[category_column, heart_target]].copy()
                category_frame[heart_target] = pd.to_numeric(category_frame[heart_target], errors="coerce")
                risk_summary = category_frame.groupby(category_column)[heart_target].mean().sort_values(ascending=False)
                fig_heart_category, ax_heart_category = plt.subplots(figsize=(7.8, 4.5))
                risk_summary.plot(kind="bar", ax=ax_heart_category, color="#8b5cf6")
                ax_heart_category.set_title(f"{category_column}별 HeartDisease 발생 비율")
                ax_heart_category.set_xlabel(category_column)
                ax_heart_category.set_ylabel("발생 비율")
                ax_heart_category.set_ylim(0, 1)
                ax_heart_category.grid(axis="y", alpha=0.22)
                ax_heart_category.tick_params(axis="x", rotation=20)
                charts["heart_category_risk"] = make_chart_uri(fig_heart_category)
                top_category = risk_summary.index[0]
                charts["heart_category_risk_note"] = f"{category_column} 기준 HeartDisease 평균값이 가장 높은 그룹은 {top_category}입니다. 웹 페이지에서는 범주형 검진 정보별 위험 차이를 보여줍니다."
                plt.close(fig_heart_category)

            st_slope_column = heart_column(csv_dataframe, "ST_Slope")
            if st_slope_column:
                st_slope_frame = csv_dataframe[[st_slope_column, heart_target]].copy()
                st_slope_frame[heart_target] = pd.to_numeric(st_slope_frame[heart_target], errors="coerce")
                st_slope_summary = st_slope_frame.groupby(st_slope_column)[heart_target].mean().sort_values(ascending=False)
                fig_st_slope, ax_st_slope = plt.subplots(figsize=(7.8, 4.5))
                st_slope_summary.plot(kind="bar", ax=ax_st_slope, color="#ef4444")
                ax_st_slope.set_title("ST slope별 HeartDisease 발생 비율")
                ax_st_slope.set_xlabel("ST slope")
                ax_st_slope.set_ylabel("발생 비율")
                ax_st_slope.set_ylim(0, 1)
                ax_st_slope.grid(axis="y", alpha=0.22)
                ax_st_slope.tick_params(axis="x", rotation=0)
                charts["heart_st_slope_risk"] = make_chart_uri(fig_st_slope)
                highest_st_slope = st_slope_summary.index[0]
                lowest_st_slope = st_slope_summary.index[-1]
                if "Flat" in st_slope_summary.index and "Up" in st_slope_summary.index:
                    charts["heart_st_slope_risk_note"] = "ST slope가 Flat인 그룹에서 HeartDisease 발생 비율이 높게 나타났으며, Up 그룹은 상대적으로 낮게 나타났습니다. 따라서 ST slope는 심장질환 발생 여부를 구분하는 핵심 범주형 변수로 볼 수 있습니다."
                else:
                    charts["heart_st_slope_risk_note"] = f"ST slope 기준 HeartDisease 발생 비율은 {highest_st_slope} 그룹에서 가장 높고 {lowest_st_slope} 그룹에서 가장 낮게 나타났습니다. 따라서 ST slope는 심장질환 발생 여부를 구분하는 핵심 범주형 변수로 볼 수 있습니다."
                plt.close(fig_st_slope)

        if generic_target and not (churn_dataset and churn_target) and not (heart_dataset and heart_target):
            target_series = csv_dataframe[generic_target]
            target_unique_count = target_series.nunique(dropna=True)
            if target_unique_count <= 20:
                target_counts = target_series.value_counts(dropna=False).head(20)
                fig_generic_target, ax_generic_target = plt.subplots(figsize=(7.8, 4.5))
                target_counts.plot(kind="bar", ax=ax_generic_target, color="#4a6fe3")
                ax_generic_target.set_title(f"{generic_target} target 후보 분포")
                ax_generic_target.set_xlabel(generic_target)
                ax_generic_target.set_ylabel("행 수")
                ax_generic_target.grid(axis="y", alpha=0.22)
                ax_generic_target.tick_params(axis="x", rotation=20)
                charts["generic_target_distribution"] = make_chart_uri(fig_generic_target)
                charts["generic_target_distribution_note"] = (
                    f"자동 추천 target 후보 '{generic_target}'의 값 분포입니다. "
                    f"서로 다른 값은 {target_unique_count}개이며, 모델링 페이지에서 다른 후보로 바꿔 실행할 수도 있습니다."
                )
                plt.close(fig_generic_target)
            elif pd.api.types.is_numeric_dtype(target_series):
                fig_generic_target, ax_generic_target = plt.subplots(figsize=(7.8, 4.5))
                pd.to_numeric(target_series, errors="coerce").dropna().plot(kind="hist", bins=20, ax=ax_generic_target, color="#4a6fe3")
                ax_generic_target.set_title(f"{generic_target} target 후보 분포")
                ax_generic_target.set_xlabel(generic_target)
                ax_generic_target.set_ylabel("빈도")
                ax_generic_target.grid(axis="y", alpha=0.22)
                charts["generic_target_distribution"] = make_chart_uri(fig_generic_target)
                charts["generic_target_distribution_note"] = (
                    f"자동 추천 target 후보 '{generic_target}'는 연속형 수치로 보여 회귀 문제로 처리할 가능성이 큽니다. "
                    "분포가 한쪽으로 치우치면 로그 변환이나 이상치 점검이 필요할 수 있습니다."
                )
                plt.close(fig_generic_target)

        generic_numeric_columns = [
            column
            for column in analysis_numeric_dataframe.columns.tolist()
            if column != generic_target
        ][:4]
        if generic_numeric_columns and not (churn_dataset and churn_target) and not (heart_dataset and heart_target):
            fig_dist, axes = plt.subplots(1, len(generic_numeric_columns), figsize=(3.1 * len(generic_numeric_columns), 4.2), squeeze=False)
            for axis, column in zip(axes[0], generic_numeric_columns):
                pd.to_numeric(csv_dataframe[column], errors="coerce").dropna().plot(kind="hist", bins=18, ax=axis, color="#50c878")
                axis.set_title(column)
                axis.set_xlabel("값")
                axis.set_ylabel("빈도")
                axis.grid(axis="y", alpha=0.22)
            fig_dist.tight_layout()
            charts["generic_numeric_distribution"] = make_chart_uri(fig_dist)
            charts["generic_numeric_distribution_note"] = f"{', '.join(generic_numeric_columns)} 컬럼의 분포를 확인했습니다. 치우침, 긴 꼬리, 다봉 분포가 있으면 모델링 전 변환 후보입니다."
            plt.close(fig_dist)

            fig_outlier, ax_outlier = plt.subplots(figsize=(7.8, 4.5))
            csv_dataframe[generic_numeric_columns].apply(pd.to_numeric, errors="coerce").plot(kind="box", ax=ax_outlier, rot=25)
            ax_outlier.set_title("주요 수치형 컬럼 이상치 분포")
            ax_outlier.grid(axis="y", alpha=0.22)
            charts["generic_outlier_boxplot"] = make_chart_uri(fig_outlier)
            outlier_counts = []
            for column in generic_numeric_columns:
                values = pd.to_numeric(csv_dataframe[column], errors="coerce").dropna()
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1
                if pd.notna(iqr) and iqr > 0:
                    outlier_counts.append((column, int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())))
            outlier_text = ", ".join(f"{column} {count}개" for column, count in outlier_counts) or "감지된 이상치 없음"
            charts["generic_outlier_boxplot_note"] = f"IQR 기준 이상치 후보는 {outlier_text}입니다. 전처리 페이지에서는 이 값을 삭제하지 않고 경계값 조정 대상으로 설명합니다."
            plt.close(fig_outlier)

        generic_categorical_columns = [
            column
            for column in column_profile["categorical_columns"]
            if column != generic_target and csv_dataframe[column].nunique(dropna=True) <= 30
        ][:3]
        if generic_categorical_columns and not (churn_dataset and churn_target) and not (heart_dataset and heart_target):
            category_column = generic_categorical_columns[0]
            category_counts = csv_dataframe[category_column].fillna("missing").value_counts().head(12)
            fig_category, ax_category = plt.subplots(figsize=(7.8, 4.5))
            category_counts.plot(kind="bar", ax=ax_category, color="#8b5cf6")
            ax_category.set_title(f"{category_column} 범주 빈도")
            ax_category.set_xlabel(category_column)
            ax_category.set_ylabel("행 수")
            ax_category.grid(axis="y", alpha=0.22)
            ax_category.tick_params(axis="x", rotation=25)
            charts["generic_category_frequency"] = make_chart_uri(fig_category)
            top_category = category_counts.index[0]
            charts["generic_category_frequency_note"] = f"{category_column}에서 가장 많은 범주는 '{top_category}'입니다. 범주 비율이 한쪽으로 몰려 있으면 모델이 다수 범주에 치우칠 수 있습니다."
            plt.close(fig_category)

        if generic_target and generic_numeric_columns and not (churn_dataset and churn_target) and not (heart_dataset and heart_target):
            target_candidate = next((candidate for candidate in target_candidates if candidate["column"] == generic_target), None)
            if target_candidate and target_candidate["task_type"] == "분류" and csv_dataframe[generic_target].nunique(dropna=True) <= 12:
                relation_columns = generic_numeric_columns[:3]
                relation_frame = csv_dataframe[[generic_target] + relation_columns].copy()
                for column in relation_columns:
                    relation_frame[column] = pd.to_numeric(relation_frame[column], errors="coerce")
                target_means = relation_frame.groupby(generic_target)[relation_columns].mean().head(12)
                fig_relation, ax_relation = plt.subplots(figsize=(7.8, 4.5))
                target_means.plot(kind="bar", ax=ax_relation)
                ax_relation.set_title(f"{generic_target}별 수치형 평균 차이")
                ax_relation.set_xlabel(generic_target)
                ax_relation.set_ylabel("평균")
                ax_relation.grid(axis="y", alpha=0.22)
                ax_relation.tick_params(axis="x", rotation=20)
                charts["generic_target_numeric_relation"] = make_chart_uri(fig_relation)
                charts["generic_target_numeric_relation_note"] = f"{generic_target} 값별로 {', '.join(relation_columns)} 평균 차이를 비교했습니다. 그룹 간 차이가 큰 변수는 모델링에서 중요한 feature 후보입니다."
                plt.close(fig_relation)
            elif target_candidate and target_candidate["task_type"] == "회귀" and generic_target in analysis_numeric_dataframe.columns and len(analysis_numeric_dataframe.columns) >= 2:
                corr_with_target = analysis_numeric_dataframe.corr(numeric_only=True)[generic_target].drop(labels=[generic_target], errors="ignore").abs().sort_values(ascending=False).head(8)
                if not corr_with_target.empty:
                    fig_relation, ax_relation = plt.subplots(figsize=(7.8, 4.5))
                    corr_with_target.plot(kind="bar", ax=ax_relation, color="#ef4444")
                    ax_relation.set_title(f"{generic_target}와 수치형 feature 상관관계")
                    ax_relation.set_xlabel("feature")
                    ax_relation.set_ylabel("절대 상관계수")
                    ax_relation.grid(axis="y", alpha=0.22)
                    ax_relation.tick_params(axis="x", rotation=25)
                    charts["generic_target_numeric_relation"] = make_chart_uri(fig_relation)
                    top_feature = corr_with_target.index[0]
                    charts["generic_target_numeric_relation_note"] = f"{generic_target}와 가장 상관이 큰 수치형 feature는 {top_feature}입니다. 상관관계는 선형 관계 점검용이며 인과로 해석하지 않습니다."
                    plt.close(fig_relation)

        if not heart_dataset and not churn_dataset and not analysis_numeric_dataframe.empty:
            numeric_summary = (
                analysis_numeric_dataframe.agg(["mean", "max"])
                .transpose()
                .sort_values("mean", ascending=False)
                .head(8)
            )
            fig_numeric, ax_numeric = plt.subplots(figsize=(7.8, 4.5))
            numeric_summary.plot(kind="bar", ax=ax_numeric, color=["#4a6fe3", "#50c878"])
            ax_numeric.set_title(f"{csv_title} 주요 수치 변수 평균/최댓값")
            ax_numeric.set_xlabel("수치 변수")
            ax_numeric.set_ylabel("값")
            ax_numeric.grid(axis="y", alpha=0.22)
            ax_numeric.tick_params(axis="x", rotation=25)
            charts["csv_numeric"] = make_chart_uri(fig_numeric)
            widest_column = numeric_summary.assign(
                gap=numeric_summary["max"] - numeric_summary["mean"]
            ).sort_values("gap", ascending=False).index[0]
            charts["csv_numeric_note"] = f"이 데이터에서는 {', '.join(numeric_summary.index.tolist())} 변수를 비교했습니다. 평균과 최댓값 차이가 가장 큰 변수는 {widest_column}로, 일부 관측값이 평균보다 크게 튀는 구간이 있습니다."
            plt.close(fig_numeric)

        date_column = None
        parsed_dates = None
        for column in csv_dataframe.columns:
            if column in numeric_dataframe.columns:
                continue
            candidate_dates = parse_datetime_series(csv_dataframe[column], pd)
            if candidate_dates.notna().mean() >= 0.6:
                date_column = column
                parsed_dates = candidate_dates
                break

        if date_column and not analysis_numeric_dataframe.empty:
            trend_columns = analysis_numeric_dataframe.columns.tolist()[:3]
            trend_dataframe = pd.DataFrame({"date": parsed_dates})
            for column in trend_columns:
                trend_dataframe[column] = pd.to_numeric(csv_dataframe[column], errors="coerce")
            trend_dataframe = trend_dataframe.dropna(subset=["date"])

            if not trend_dataframe.empty:
                trend_dataframe["date"] = trend_dataframe["date"].dt.date
                trend_frame = trend_dataframe.groupby("date")[trend_columns].mean()
                fig_csv_time, ax_csv_time = plt.subplots(figsize=(7.8, 4.5))
                trend_frame.plot(kind="line", marker="o", ax=ax_csv_time, linewidth=2.3)
                ax_csv_time.set_title(f"{date_column} 기준 주요 수치 변수 평균 추이")
                ax_csv_time.set_xlabel("날짜")
                ax_csv_time.set_ylabel("평균값")
                ax_csv_time.grid(alpha=0.24)
                charts["csv_time"] = make_chart_uri(fig_csv_time)
                most_variable = trend_frame.std(numeric_only=True).sort_values(ascending=False).index[0]
                start_date = trend_frame.index.min()
                end_date = trend_frame.index.max()
                charts["csv_time_note"] = f"{date_column} 기준 {start_date}부터 {end_date}까지의 흐름을 보면 {', '.join(trend_columns)} 중 {most_variable}의 날짜별 변동이 가장 큽니다."
                plt.close(fig_csv_time)

        day_column = next(
            (column for column in csv_dataframe.columns if column.lower() in {"day_of_week", "weekday", "day_name"}),
            None,
        )
        if day_column and not analysis_numeric_dataframe.empty:
            weekday_columns = analysis_numeric_dataframe.columns.tolist()[:3]
            weekday_frame = csv_dataframe[[day_column] + weekday_columns].copy()
            for column in weekday_columns:
                weekday_frame[column] = pd.to_numeric(weekday_frame[column], errors="coerce")
            weekday_summary = weekday_frame.groupby(day_column)[weekday_columns].mean()
            fig_weekday, ax_weekday = plt.subplots(figsize=(7.8, 4.5))
            weekday_summary.plot(kind="bar", ax=ax_weekday)
            ax_weekday.set_title(f"{day_column} 기준 주요 수치 변수 평균")
            ax_weekday.set_xlabel("요일")
            ax_weekday.set_ylabel("평균값")
            ax_weekday.grid(axis="y", alpha=0.22)
            ax_weekday.tick_params(axis="x", rotation=20)
            charts["csv_weekday"] = make_chart_uri(fig_weekday)
            first_weekday_column = weekday_columns[0]
            top_weekday = weekday_summary[first_weekday_column].idxmax()
            charts["csv_weekday_note"] = f"{day_column}별 평균을 비교하면 {first_weekday_column} 값은 {top_weekday}에서 가장 높게 나타납니다. 나머지 변수도 요일별 높낮이를 함께 비교할 수 있습니다."
            plt.close(fig_weekday)

        if len(analysis_numeric_dataframe.columns) >= 2:
            if churn_dataset:
                preferred_corr_columns = ["age", "calls", "monthly_charge", "churn"]
                corr_columns = [
                    column
                    for column in [churn_column(csv_dataframe, preferred_column) for preferred_column in preferred_corr_columns]
                    if column and column in analysis_numeric_dataframe.columns
                ]
            else:
                corr_columns = analysis_numeric_dataframe.columns.tolist()[:8]
            if len(corr_columns) < 2:
                corr_columns = analysis_numeric_dataframe.columns.tolist()[:8]
            corr_frame = analysis_numeric_dataframe[corr_columns].corr()
            fig_corr, ax_corr = plt.subplots(figsize=(7.8, 5.2))
            image = ax_corr.imshow(corr_frame, cmap="coolwarm", vmin=-1, vmax=1)
            ax_corr.set_title("주요 수치 변수 상관관계")
            ax_corr.set_xticks(range(len(corr_columns)))
            ax_corr.set_xticklabels(corr_columns, rotation=35, ha="right")
            ax_corr.set_yticks(range(len(corr_columns)))
            ax_corr.set_yticklabels(corr_columns)
            fig_corr.colorbar(image, ax=ax_corr, fraction=0.046, pad=0.04)
            for row_index in range(len(corr_columns)):
                for column_index in range(len(corr_columns)):
                    ax_corr.text(
                        column_index,
                        row_index,
                        f"{corr_frame.iloc[row_index, column_index]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="#172033",
                    )
            charts["csv_corr"] = make_chart_uri(fig_corr)
            corr_pairs = corr_frame.where(~pd.DataFrame(
                [[row == column for column in corr_columns] for row in corr_columns],
                index=corr_columns,
                columns=corr_columns,
            )).abs().stack()
            strongest_pair = corr_pairs.idxmax()
            strongest_value = corr_frame.loc[strongest_pair[0], strongest_pair[1]]
            if churn_dataset:
                charts["csv_corr_note"] = (
                    f"churn 주제와 직접 연결되는 {', '.join(corr_columns)}만 남겨 상관관계를 비교했습니다. "
                    f"가장 강한 조합은 {strongest_pair[0]}와 {strongest_pair[1]}이며, 상관계수는 {strongest_value:.2f}입니다. "
                    "상관관계는 인과가 아니라 변수 간 선형 관계를 보는 탐색 지표로 해석합니다."
                )
            else:
                charts["csv_corr_note"] = f"상관관계가 가장 강한 조합은 {strongest_pair[0]}와 {strongest_pair[1]}이며, 상관계수는 {strongest_value:.2f}입니다."
            plt.close(fig_corr)

    charts["eda_tabs"] = build_eda_tabs(charts)

    return {
        "charts": charts,
        "csv_profiles": csv_profiles,
    }


def build_preprocess_charts(documents):
    matplotlib_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".matplotlib_cache")
    os.makedirs(matplotlib_cache, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", matplotlib_cache)

    try:
        import pandas as pd
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("전처리 시각화를 사용하려면 pandas와 matplotlib 설치가 필요합니다.") from error

    csv_frames = read_csv_documents(documents, pd)
    csv_profiles = [
        build_csv_profile(csv_file["document"], csv_file["dataframe"], pd)
        for csv_file in csv_frames
    ]
    charts = {}
    preprocess_summary = None

    plt.rcParams["font.family"] = ["Malgun Gothic", "Arial", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    if csv_frames:
        first_csv = csv_frames[0]
        csv_dataframe = first_csv["dataframe"]
        column_profile = classify_csv_columns(csv_dataframe, pd)
        numeric_columns = column_profile["numeric_columns"]
        date_columns = column_profile["date_columns"]
        categorical_columns = column_profile["categorical_columns"]
        charts["column_type_rows"] = column_profile["column_rows"]
        charts["target_candidates"] = column_profile["target_candidates"]
        churn_dataset = is_churn_dataset(csv_dataframe, first_csv["title"])
        heart_dataset = is_heart_dataset(csv_dataframe, first_csv["title"])
        heart_zero_missing_columns = [
            column
            for column in [
                heart_column(csv_dataframe, "Cholesterol"),
                heart_column(csv_dataframe, "RestingBP"),
            ]
            if column and column in csv_dataframe.columns
        ]
        zero_missing_counts = {}
        for column in heart_zero_missing_columns:
            zero_count = int((pd.to_numeric(csv_dataframe[column], errors="coerce") == 0).sum())
            if zero_count:
                zero_missing_counts[column] = zero_count

        before_missing = csv_dataframe.isna().sum()
        logical_before_missing = before_missing.copy()
        for column, count in zero_missing_counts.items():
            logical_before_missing[column] = logical_before_missing.get(column, 0) + count

        before_duplicate_count = int(csv_dataframe.duplicated().sum())
        processed_dataframe = csv_dataframe.drop_duplicates().copy()

        for column in date_columns:
            processed_dataframe[column] = parse_datetime_series(processed_dataframe[column], pd)
        for column in numeric_columns:
            numeric_series = pd.to_numeric(processed_dataframe[column], errors="coerce")
            if heart_dataset and column in heart_zero_missing_columns:
                numeric_series = numeric_series.mask(numeric_series == 0)
            processed_dataframe[column] = numeric_series.fillna(numeric_series.median())
        for column in categorical_columns:
            codes, _ = pd.factorize(processed_dataframe[column].fillna("missing").astype(str))
            processed_dataframe[column] = codes

        after_missing = processed_dataframe.isna().sum()
        if heart_dataset and zero_missing_counts:
            missing_summary = pd.Series(zero_missing_counts).sort_values(ascending=False).head(10)
            missing_total = sum(zero_missing_counts.values())
        else:
            missing_summary = logical_before_missing.sort_values(ascending=False).head(10)
            missing_total = int(logical_before_missing.sum())
        if missing_total:
            fig_missing, ax_missing = plt.subplots(figsize=(7.8, 4.5))
            missing_summary.plot(kind="bar", ax=ax_missing, color="#f59e0b")
            if heart_dataset and zero_missing_counts:
                ax_missing.set_title("컬럼별 결측성 0값 분포")
                charts["missing_by_column_title"] = "컬럼별 결측성 0값 분포"
            else:
                ax_missing.set_title("컬럼별 결측치/결측성 이상치 개수")
                charts["missing_by_column_title"] = "컬럼별 결측치 개수"
            ax_missing.set_xlabel("컬럼")
            ax_missing.set_ylabel("개수")
            ax_missing.grid(axis="y", alpha=0.22)
            ax_missing.tick_params(axis="x", rotation=25)
            charts["missing_by_column"] = make_chart_uri(fig_missing)
            plt.close(fig_missing)

            top_missing_column = logical_before_missing.sort_values(ascending=False).index[0]
            if heart_dataset and zero_missing_counts:
                zero_text = ", ".join(f"{column} {count}건" for column, count in zero_missing_counts.items())
                major_column = max(zero_missing_counts, key=zero_missing_counts.get)
                minor_parts = [
                    f"{column}는 {count}건"
                    for column, count in zero_missing_counts.items()
                    if column != major_column
                ]
                minor_text = f", {' / '.join(minor_parts)}만 발견되었습니다" if minor_parts else "에 집중되어 있습니다"
                if not minor_parts:
                    major_sentence = f"전체 결측성 0값은 {major_column}에 집중되어 있습니다."
                else:
                    major_sentence = f"전체 결측성 0값은 {major_column}에 집중되어 있으며{minor_text}."
                charts["missing_by_column_note"] = (
                    f"일반적인 NaN 결측치는 발견되지 않았지만, {zero_text}의 0값은 실제 의학적 수치로 보기 어려워 결측성 이상치로 판단했습니다. "
                    f"{major_sentence} 따라서 주요 보정 대상은 {major_column} 컬럼입니다."
                )
            else:
                charts["missing_by_column_note"] = f"전처리 전 결측치는 총 {missing_total}개이며, 가장 많이 비어 있는 컬럼은 {top_missing_column}입니다."

            compare_source = pd.Series(zero_missing_counts) if heart_dataset and zero_missing_counts else logical_before_missing
            compare_after = pd.Series(0, index=compare_source.index)
            compare_frame = pd.DataFrame({"전처리 전": compare_source, "전처리 후": compare_after}).sort_values("전처리 전", ascending=False).head(10)
            fig_compare, ax_compare = plt.subplots(figsize=(7.8, 4.5))
            compare_frame.plot(kind="bar", ax=ax_compare, color=["#f59e0b", "#50c878"])
            if heart_dataset and zero_missing_counts:
                ax_compare.set_title("전처리 전/후 결측성 0값 비교")
                charts["missing_before_after_title"] = "전처리 전/후 결측성 0값 비교"
            else:
                ax_compare.set_title("전처리 전/후 결측치 비교")
                charts["missing_before_after_title"] = "전처리 전/후 결측치 비교"
            ax_compare.set_xlabel("컬럼")
            ax_compare.set_ylabel("개수")
            ax_compare.grid(axis="y", alpha=0.22)
            ax_compare.tick_params(axis="x", rotation=25)
            if heart_dataset and zero_missing_counts:
                x_positions = list(range(len(compare_frame.index)))
                ax_compare.scatter(
                    [position + 0.125 for position in x_positions],
                    [0 for _ in x_positions],
                    color="#50c878",
                    edgecolor="#15803d",
                    s=70,
                    zorder=5,
                    label="전처리 후 0개",
                )
                for position in x_positions:
                    ax_compare.annotate(
                        "0개",
                        (position + 0.125, 0),
                        textcoords="offset points",
                        xytext=(0, 8),
                        ha="center",
                        color="#15803d",
                        fontsize=9,
                        fontweight="bold",
                    )
                ax_compare.legend()
            charts["missing_before_after"] = make_chart_uri(fig_compare)
            after_total = int(compare_after.sum())
            if heart_dataset and zero_missing_counts:
                charts["missing_before_after_note"] = f"전처리 후 값은 0개라 막대 높이가 없어 보일 수 있어 초록색 마커와 '0개' 라벨로 표시했습니다. 중앙값 대체 후 결측성 0값 문제는 {after_total}개로 정리되었습니다."
            else:
                charts["missing_before_after_note"] = f"결측치와 결측성 이상치는 전처리 전 {missing_total}개에서 전처리 후 {after_total}개로 정리됩니다."
            plt.close(fig_compare)
        else:
            if churn_dataset:
                charts["missing_status"] = "현재 고객 이탈 CSV에는 NaN 결측치가 없습니다. 이 파일은 이미 gender 라벨 인코딩과 payment_method/region 원-핫 인코딩이 적용된 모델 입력용 데이터입니다."
            else:
                charts["missing_status"] = "이 데이터는 전처리 전부터 감지된 결측치와 결측성 이상치가 없습니다. 중복, 타입, 인코딩, 이상치 처리는 아래 전처리 표에서 함께 확인합니다."

        iqr_outlier_count = 0
        changed_count = 0
        if numeric_columns:
            if churn_dataset:
                preferred_box_columns = ["monthly_charge", "customer_support_calls", "contract_length", "data_usage"]
                box_columns = [
                    column
                    for column in [churn_column(csv_dataframe, preferred_column) for preferred_column in preferred_box_columns]
                    if column and column in numeric_columns
                ]
            elif heart_dataset:
                preferred_box_columns = ["Age", "RestingBP", "Cholesterol", "FastingBS"]
                box_columns = [
                    column
                    for column in [heart_column(csv_dataframe, preferred_column) for preferred_column in preferred_box_columns]
                    if column and column in numeric_columns
                ]
            else:
                box_columns = numeric_columns[:4]
            if not box_columns:
                box_columns = numeric_columns[:4]
            before_box = csv_dataframe[box_columns].apply(pd.to_numeric, errors="coerce")
            after_box = before_box.copy()
            for column in box_columns:
                if heart_dataset and column in heart_zero_missing_columns:
                    after_box[column] = after_box[column].mask(after_box[column] == 0)
                    after_box[column] = after_box[column].fillna(after_box[column].median())
                q1 = after_box[column].quantile(0.25)
                q3 = after_box[column].quantile(0.75)
                iqr = q3 - q1
                if pd.notna(iqr) and iqr > 0:
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    iqr_outlier_count += int(((after_box[column] < lower_bound) | (after_box[column] > upper_bound)).sum())
                    after_box[column] = after_box[column].clip(lower_bound, upper_bound)

            fig_box, axes = plt.subplots(1, 2, figsize=(9.4, 4.5), sharey=False)
            before_box.plot(kind="box", ax=axes[0], rot=25)
            after_box.plot(kind="box", ax=axes[1], rot=25)
            axes[0].set_title("이상치 처리 전")
            axes[1].set_title("이상치 처리 후")
            for axis in axes:
                axis.grid(axis="y", alpha=0.22)
            charts["outlier_boxplot"] = make_chart_uri(fig_box)
            changed_count = int((before_box.fillna(0) != after_box.fillna(0)).sum().sum())
            if churn_dataset:
                charts["outlier_boxplot_note"] = (
                    f"{', '.join(box_columns)}를 기준으로 IQR 방식의 이상치를 탐지했습니다. "
                    f"극단값 {iqr_outlier_count}개는 행 삭제 없이 상·하한값으로 조정해 고객 수를 유지합니다."
                )
            elif heart_dataset:
                charts["outlier_boxplot_note"] = (
                    f"{', '.join(box_columns)}를 기준으로 IQR 방식으로 이상치를 탐지했습니다. "
                    f"극단값 {iqr_outlier_count}개는 제거하지 않고 상·하한값으로 조정해 데이터 수를 유지했습니다. "
                    "cholesterol 또는 restingbp의 0값은 실제 의학적 값으로 보기 어려워 결측성 이상치로 판단하고 중앙값으로 보완했습니다."
                )
            else:
                charts["outlier_boxplot_note"] = f"{', '.join(box_columns)} 기준으로 IQR 범위를 벗어난 값 {changed_count}개가 경계값 안으로 조정됩니다."
            plt.close(fig_box)

        zero_missing_total = sum(zero_missing_counts.values())
        total_adjustment_count = zero_missing_total + iqr_outlier_count
        if churn_dataset:
            charts["preprocess_mapping_rows"] = build_churn_preprocess_mapping_rows()
            charts["preprocess_step_rows"] = [
                {"step": "df.info()", "evidence": f"전체 {len(csv_dataframe)}행, {len(csv_dataframe.columns)}열 / 수치형 {len(numeric_columns)}개, 범주형 {len(categorical_columns)}개"},
                {"step": "isnull()", "evidence": f"NaN 결측치 {int(before_missing.sum())}개"},
                {"step": "duplicated()", "evidence": f"중복 행 {before_duplicate_count}개"},
                {"step": "IQR 이상치", "evidence": f"IQR 기준 이상치 {iqr_outlier_count}개, 행 삭제 없이 클리핑 기준 확인"},
                {"step": "인코딩", "evidence": "gender 라벨 인코딩, payment_method/region drop-first 원-핫 인코딩 적용"},
                {"step": "스케일링", "evidence": "모델 학습 시 수치형 feature에 StandardScaler 적용"},
                {"step": "data split", "evidence": "train 70% / validation 30%, 분류 문제는 가능한 경우 stratify로 클래스 비율 유지"},
            ]
            zero_missing_row = {
                "item": "결측성 0값",
                "before": "해당 없음",
                "after": "보정 없음",
                "method": "고객 이탈 데이터에서는 0을 결측으로 판단하지 않음",
            }
            encoding_row = {
                "item": "인코딩",
                "before": "원본 범주형: gender, payment_method, region",
                "after": "현재 CSV는 인코딩 완료",
                "method": "gender는 0/1, payment_method와 region은 drop-first 원-핫 인코딩",
            }
        else:
            zero_missing_row = {
                "item": "결측성 0값",
                "before": f"{zero_missing_total}개" if heart_dataset else "해당 없음",
                "after": "중앙값 대체" if zero_missing_total else "해당 없음",
                "method": "cholesterol/restingbp의 0값을 비정상 값으로 판단" if heart_dataset else "0값을 별도 결측으로 처리하지 않음",
            }
            encoding_row = {
                "item": "인코딩",
                "before": f"범주형 {len(categorical_columns)}개",
                "after": "LabelEncoding 완료",
                "method": "문자열 범주를 숫자 코드로 변환",
            }
            charts["preprocess_step_rows"] = [
                {"step": "컬럼 자동 분류", "evidence": f"수치형 {len(numeric_columns)}개, 범주형 {len(categorical_columns)}개, 날짜형 {len(date_columns)}개, 식별자 후보 {len(column_profile['identifier_columns'])}개"},
                {"step": "target 후보 추천", "evidence": column_profile["target_candidates"][0]["reason"] if column_profile["target_candidates"] else "명확한 target 후보가 없어 모델링 페이지에서 설명 메시지를 제공합니다."},
                {"step": "isnull()", "evidence": f"NaN 결측치 {int(before_missing.sum())}개"},
                {"step": "duplicated()", "evidence": f"중복 행 {before_duplicate_count}개"},
                {"step": "타입 변환", "evidence": f"숫자로 해석 가능한 문자열은 수치형으로, 날짜로 해석 가능한 문자열은 datetime으로 변환"},
                {"step": "IQR 이상치", "evidence": f"IQR 기준 이상치 {iqr_outlier_count}개, 행 삭제 없이 클리핑 기준 확인"},
                {"step": "인코딩/스케일링", "evidence": "모델링 시 범주형은 LabelEncoding, 수치형 feature는 StandardScaler 적용"},
                {"step": "data split", "evidence": "train 70% / validation 30%, 분류 문제는 가능한 경우 stratify로 클래스 비율 유지"},
            ]
        charts["preprocess_table_rows"] = [
            {
                "item": "NaN 결측치",
                "before": f"{int(before_missing.sum())}개",
                "after": f"{int(after_missing.sum())}개",
                "method": "기본 결측치 검사",
            },
            zero_missing_row,
            {
                "item": "중복 행",
                "before": f"{before_duplicate_count}개",
                "after": "0개",
                "method": "중복 레코드 제거 기준 적용",
            },
            {
                "item": "데이터 타입",
                "before": f"수치형 {len(numeric_columns)}개 / 범주형 {len(categorical_columns)}개",
                "after": "모델 입력 타입 정리",
                "method": "수치형 변환, 범주형 코드화",
            },
            {
                "item": "IQR 이상치",
                "before": f"IQR 기준 {iqr_outlier_count}개",
                "after": "상·하한값 조정" if iqr_outlier_count else "해당 없음",
                "method": "극단값을 제거하지 않고 클리핑",
            },
            {
                "item": "전체 보정값",
                "before": f"{total_adjustment_count}개",
                "after": "보정 완료" if total_adjustment_count else "해당 없음",
                "method": "데이터 수 유지를 위해 행 삭제 없이 값만 조정",
            },
            encoding_row,
            {
                "item": "스케일링",
                "before": "원본 단위",
                "after": "모델 학습 시 표준화",
                "method": "StandardScaler로 train/validation feature 정규화",
            },
            {
                "item": "데이터 분할",
                "before": "전체 데이터",
                "after": "train 70% / validation 30%",
                "method": "분류 문제는 가능한 경우 stratify 적용",
            },
            {
                "item": "최종 행 수",
                "before": f"{len(csv_dataframe)}행",
                "after": f"{len(processed_dataframe)}행",
                "method": "결측/이상치 보완 후 분석용 데이터 유지",
            },
        ]
        if churn_dataset:
            charts["preprocess_detail_note"] = (
                "수업 자료의 전처리 흐름(df.info, isnull, IQR 이상치, 인코딩, 스케일링, data split)에 맞춰 현재 CSV의 상태와 모델 입력용 변환 결과를 함께 정리했습니다. "
                "현재 업로드된 CSV는 gender가 0/1로, payment_method와 region이 drop-first 원-핫 컬럼으로 이미 변환된 모델 입력 스키마입니다."
            )
        elif heart_dataset:
            charts["preprocess_detail_note"] = (
                "일반적인 NaN 결측치는 발견되지 않았지만, cholesterol 0값과 restingbp 0값은 실제 의학적 수치로 보기 어려워 결측성 이상치로 판단했습니다. "
                "결측성 0값은 중앙값으로 대체하고, IQR 이상치는 행을 삭제하지 않고 상·하한값으로 조정해 데이터 수를 유지했습니다."
            )
        else:
            charts["preprocess_detail_note"] = (
                "CSV 컬럼 구조를 자동으로 분석해 수치형, 범주형, 날짜형, 식별자 후보를 나누고 결측치, 중복, 이상치를 점검했습니다. "
                "모델링 가능 여부는 target 후보와 학습 가능한 feature 수를 기준으로 판단합니다."
            )

        preprocess_summary = {
            "missing": f"결측치는 수치형 {len(numeric_columns)}개 컬럼은 중앙값으로, 범주형 {len(categorical_columns)}개 컬럼은 missing 값으로 보완합니다.",
            "date": f"날짜로 해석 가능한 컬럼 {len(date_columns)}개를 datetime 형식으로 변환합니다.",
            "outlier": "수치형 변수는 IQR 기준으로 과도하게 튀는 값을 상·하한 경계로 조정합니다.",
            "numeric": f"수치형 변수 {len(numeric_columns)}개는 모델링과 시각화를 위해 숫자 타입으로 정리합니다.",
            "categorical": f"범주형 변수 {len(categorical_columns)}개는 문자열 결측치를 정리한 뒤 숫자 코드로 인코딩합니다.",
        }

    return {
        "charts": charts,
        "csv_profiles": csv_profiles,
        "preprocess_summary": preprocess_summary,
    }


def build_ml_analysis(document, target_column):
    matplotlib_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".matplotlib_cache")
    os.makedirs(matplotlib_cache, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", matplotlib_cache)

    try:
        import pandas as pd
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, StandardScaler
    except ImportError as error:
        raise RuntimeError("머신러닝 분석을 사용하려면 scikit-learn 설치가 필요합니다.") from error

    plt.rcParams["font.family"] = ["Malgun Gothic", "Arial", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    dataframe = normalize_csv_dataframe_for_analysis(pd.read_csv(StringIO(document.content)), pd)
    if not target_column:
        target_column = auto_target_column(dataframe, pd)
        if not target_column:
            raise RuntimeError(
                "모델링을 자동 실행할 target 후보를 찾지 못했습니다. "
                "값이 하나뿐인 컬럼, 날짜 컬럼, 고유 ID 컬럼을 제외하면 예측 대상으로 삼을 만한 컬럼이 부족합니다."
            )
    if target_column not in dataframe.columns:
        raise RuntimeError("선택한 target 컬럼을 CSV에서 찾지 못했습니다.")

    dataframe = dataframe.dropna(subset=[target_column]).copy()
    if len(dataframe) < 10:
        raise RuntimeError(f"머신러닝 분석에는 target 결측치를 제외한 행이 최소 10개 필요합니다. 현재 사용 가능한 행은 {len(dataframe)}개입니다.")

    y = dataframe[target_column]
    if y.nunique() < 2:
        raise RuntimeError(f"'{target_column}' 컬럼은 값이 1종류뿐이라 예측 문제를 만들 수 없습니다. 서로 다른 값이 2개 이상인 target을 선택하세요.")

    is_numeric_target = pd.api.types.is_numeric_dtype(y)
    unique_ratio = y.nunique() / len(y)
    is_regression = is_numeric_target and (y.nunique() > 10 or unique_ratio > 0.2)
    if not is_regression and y.nunique() > min(50, max(10, len(y) // 2)):
        raise RuntimeError(
            f"'{target_column}' 컬럼은 서로 다른 값이 {y.nunique()}개라 분류 target으로 쓰기 어렵습니다. "
            "고유 ID나 자유 텍스트에 가까운 컬럼일 수 있으니 값 종류가 더 적은 target을 선택하세요."
        )

    column_profile = classify_csv_columns(dataframe, pd)
    feature_columns = [
        column
        for column in dataframe.columns
        if column != target_column
        and column not in column_profile["date_columns"]
        and column not in column_profile["identifier_columns"]
        and column not in column_profile["constant_columns"]
    ]
    if not feature_columns:
        raise RuntimeError(
            "학습에 사용할 feature 컬럼이 없습니다. target, 날짜, 고유 ID, 상수 컬럼을 제외하니 모델에 넣을 설명 변수가 남지 않았습니다."
        )

    X = dataframe[feature_columns].copy()
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    for column in numeric_columns:
        median_value = X[column].median()
        if pd.isna(median_value):
            X = X.drop(columns=[column])
            continue
        X[column] = X[column].fillna(median_value).astype(float)
    numeric_columns = [column for column in numeric_columns if column in X.columns]

    for column in categorical_columns:
        if column not in X.columns:
            continue
        X[column] = X[column].fillna("missing").astype(str)
        encoder = LabelEncoder()
        X[column] = encoder.fit_transform(X[column])
    categorical_columns = [column for column in categorical_columns if column in X.columns]
    if X.shape[1] == 0:
        raise RuntimeError("전처리 후 학습 가능한 feature가 남지 않았습니다. 결측치가 너무 많거나 식별자/날짜/상수 컬럼만 있는 CSV입니다.")
    if len(dataframe) < 20 and not is_regression and y.value_counts().min() < 2:
        raise RuntimeError("분류 모델링에는 각 target 클래스가 train/validation에 나뉠 수 있도록 클래스별 최소 2개 이상의 행이 필요합니다.")

    min_validation_rows = max(1, int(len(y) * 0.3))
    stratify = None
    if not is_regression and y.value_counts().min() >= 2 and y.nunique() <= min_validation_rows:
        stratify = y
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=123,
        stratify=stratify,
    )

    if numeric_columns:
        scaler = StandardScaler()
        X_train.loc[:, numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
        X_val.loc[:, numeric_columns] = scaler.transform(X_val[numeric_columns])

    if is_regression:
        models = [
            ("LinearRegression", LinearRegression()),
            ("RandomForestRegressor", RandomForestRegressor(random_state=123)),
        ]
    else:
        models = [
            ("LogisticRegression", LogisticRegression(max_iter=1000)),
            ("RandomForestClassifier", RandomForestClassifier(random_state=123)),
        ]

    def binary_roc_auc(model, X_data, y_true):
        if is_regression or y.nunique() != 2 or not hasattr(model, "predict_proba"):
            return None
        try:
            y_encoder = LabelEncoder()
            y_encoder.fit(y)
            y_true_encoded = y_encoder.transform(y_true)
            positive_label = y_encoder.classes_[-1]
            class_index = list(model.classes_).index(positive_label)
            y_score = model.predict_proba(X_data)[:, class_index]
            return round(roc_auc_score(y_true_encoded, y_score), 4)
        except Exception:
            return None

    predictions_by_model = {}

    try:
        from xgboost import XGBClassifier, XGBRegressor

        if is_regression:
            xgb_model = XGBRegressor(
                n_estimators=80,
                max_depth=3,
                min_child_weight=1,
                random_state=123,
            )
            xgb_model.fit(X_train, y_train)
            xgb_train_pred = xgb_model.predict(X_train)
            xgb_val_pred = xgb_model.predict(X_val)
            xgb_row = {
                "name": "XGBRegressor",
                "train_score": round(r2_score(y_train, xgb_train_pred), 4),
                "valid_score": round(r2_score(y_val, xgb_val_pred), 4),
                "secondary_score": round(mean_squared_error(y_val, xgb_val_pred) ** 0.5, 4),
            }
        else:
            y_encoder = LabelEncoder()
            y_train_xgb = y_encoder.fit_transform(y_train)
            y_val_xgb = y_encoder.transform(y_val)
            xgb_model = XGBClassifier(
                n_estimators=80,
                max_depth=3,
                min_child_weight=1,
                random_state=123,
                eval_metric="mlogloss",
            )
            xgb_model.fit(X_train, y_train_xgb)
            xgb_train_pred = y_encoder.inverse_transform(xgb_model.predict(X_train))
            xgb_val_pred = y_encoder.inverse_transform(xgb_model.predict(X_val))
            predictions_by_model["XGBClassifier"] = xgb_val_pred
            xgb_roc_auc = None
            if y.nunique() == 2 and hasattr(xgb_model, "predict_proba"):
                try:
                    xgb_roc_auc = round(roc_auc_score(y_val_xgb, xgb_model.predict_proba(X_val)[:, 1]), 4)
                except Exception:
                    xgb_roc_auc = None
            xgb_row = {
                "name": "XGBClassifier",
                "train_score": round(f1_score(y_train, xgb_train_pred, average="macro"), 4),
                "valid_score": round(f1_score(y_val, xgb_val_pred, average="macro"), 4),
                "secondary_score": round(accuracy_score(y_val, xgb_val_pred), 4),
                "precision": round(precision_score(y_val, xgb_val_pred, average="macro", zero_division=0), 4),
                "recall": round(recall_score(y_val, xgb_val_pred, average="macro", zero_division=0), 4),
                "roc_auc": xgb_roc_auc,
            }
    except Exception:
        xgb_row = None

    scores = []
    for name, model in models:
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        if is_regression:
            scores.append(
                {
                    "name": name,
                    "train_score": round(r2_score(y_train, train_pred), 4),
                    "valid_score": round(r2_score(y_val, val_pred), 4),
                    "secondary_score": round(mean_squared_error(y_val, val_pred) ** 0.5, 4),
                }
            )
        else:
            roc_auc = binary_roc_auc(model, X_val, y_val)
            predictions_by_model[name] = val_pred
            scores.append(
                {
                    "name": name,
                    "train_score": round(f1_score(y_train, train_pred, average="macro"), 4),
                    "valid_score": round(f1_score(y_val, val_pred, average="macro"), 4),
                    "secondary_score": round(accuracy_score(y_val, val_pred), 4),
                    "precision": round(precision_score(y_val, val_pred, average="macro", zero_division=0), 4),
                    "recall": round(recall_score(y_val, val_pred, average="macro", zero_division=0), 4),
                    "roc_auc": roc_auc,
                }
            )

    if xgb_row:
        scores.append(xgb_row)

    score_frame = pd.DataFrame(scores).sort_values("valid_score", ascending=False)
    best_model = score_frame.iloc[0].to_dict()
    confusion_rows = []
    class_distribution_text = ""
    imbalance_note = ""
    if not is_regression:
        class_counts = y.value_counts().sort_index()
        class_distribution_text = ", ".join(f"{label}: {int(count)}명" for label, count in class_counts.items())
        minority_ratio = float(class_counts.min() / class_counts.sum()) if int(class_counts.sum()) else 0
        if minority_ratio < 0.35:
            imbalance_note = "이 데이터는 이탈 클래스 비율이 낮으므로 accuracy만 보면 모델이 좋아 보일 수 있습니다. 따라서 precision, recall, F1-score, ROC-AUC, confusion matrix를 함께 확인해야 합니다."
        else:
            imbalance_note = "분류 모델은 accuracy뿐 아니라 precision, recall, F1-score, ROC-AUC, confusion matrix를 함께 확인했습니다."
        best_predictions = predictions_by_model.get(best_model["name"])
        if best_predictions is not None:
            labels = sorted(y.dropna().unique().tolist())
            matrix = confusion_matrix(y_val, best_predictions, labels=labels)
            for row_index, actual_label in enumerate(labels):
                confusion_rows.append(
                    {
                        "actual": actual_label,
                        "predicted": [
                            {"label": predicted_label, "count": int(matrix[row_index][column_index])}
                            for column_index, predicted_label in enumerate(labels)
                        ],
                    }
                )
    row_count = len(dataframe)
    column_count = len(dataframe.columns)
    feature_preview = feature_columns[:8]
    remaining_feature_count = max(len(feature_columns) - len(feature_preview), 0)

    if is_regression:
        valid_score = best_model["valid_score"]
        if valid_score >= 0.7:
            result_summary = "검증 데이터에서도 target 값을 비교적 잘 설명하는 모델입니다."
        elif valid_score >= 0.3:
            result_summary = "일부 패턴은 잡았지만 실제 예측에는 추가 검증이 필요합니다."
        elif valid_score >= 0:
            result_summary = "검증 설명력이 낮아 모델 결과는 참고용으로 해석하는 것이 좋습니다."
        else:
            result_summary = "검증 R2가 음수라서 단순 평균 예측보다도 성능이 낮을 수 있습니다."
    else:
        valid_score = best_model["valid_score"]
        if valid_score >= 0.8:
            result_summary = "검증 데이터에서도 분류 성능이 높은 편입니다."
        elif valid_score >= 0.5:
            result_summary = "기본적인 분류 패턴은 잡았지만 오분류 가능성을 함께 봐야 합니다."
        else:
            result_summary = "검증 분류 성능이 낮아 feature 보강이나 target 재검토가 필요합니다."

    category_name = document.category.name if document.category else "미분류"
    created_at = document.created_at.strftime("%Y-%m-%d %H:%M") if document.created_at else "알 수 없음"
    feature_text = ", ".join(feature_preview)
    if remaining_feature_count:
        feature_text = f"{feature_text} 외 {remaining_feature_count}개"
    churn_dataset = is_churn_dataset(dataframe, document.title)
    preprocess_text = f"수치형 {len(numeric_columns)}개는 결측치를 중앙값으로 채운 뒤 스케일링했고, 범주형 {len(categorical_columns)}개는 missing 처리 후 LabelEncoding했습니다."
    if churn_dataset:
        preprocess_text = (
            "현재 업로드된 CSV는 gender 0/1 라벨 인코딩과 payment_method/region drop-first 원-핫 인코딩이 이미 적용된 모델 입력 스키마입니다. "
            f"모델 학습 단계에서는 수치형 입력 {len(numeric_columns)}개를 결측치 보완 후 StandardScaler로 스케일링하고 train 70% / validation 30%로 분할했습니다."
        )

    fig_score, ax_score = plt.subplots(figsize=(7.8, 4.5))
    score_frame.plot(x="name", y=["train_score", "valid_score"], kind="bar", ax=ax_score, color=["#4a6fe3", "#50c878"])
    primary_metric = "R2 score" if is_regression else "F1 macro"
    secondary_metric = "RMSE" if is_regression else "Accuracy"
    tertiary_metric = "" if is_regression else "ROC-AUC"
    ax_score.set_title(f"{target_column} 모델별 {primary_metric}")
    ax_score.set_xlabel("모델")
    ax_score.set_ylabel(primary_metric)
    if not is_regression:
        ax_score.set_ylim(0, 1.05)
    ax_score.grid(axis="y", alpha=0.22)
    ax_score.tick_params(axis="x", rotation=15)
    score_chart = make_chart_uri(fig_score)
    plt.close(fig_score)

    return {
        "target_column": target_column,
        "task_type": "회귀" if is_regression else "분류",
        "primary_metric": primary_metric,
        "secondary_metric": secondary_metric,
        "tertiary_metric": tertiary_metric,
        "row_count": row_count,
        "column_count": column_count,
        "train_count": len(X_train),
        "valid_count": len(X_val),
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "class_count": int(y.nunique()),
        "scores": score_frame.to_dict("records"),
        "best_model": best_model,
        "confusion_rows": confusion_rows,
        "class_distribution": class_distribution_text,
        "classification_note": imbalance_note,
        "score_chart": score_chart,
        "explanation": {
            "source": f"{document.title} / 카테고리: {category_name} / 저장일: {created_at}",
            "dataset": f"총 {row_count}행, {column_count}열 CSV에서 target 결측치가 있는 행을 제외하고 분석했습니다.",
            "target": f"'{target_column}' 컬럼을 예측 대상으로 선택해 {'연속형 수치를 예측하는 회귀' if is_regression else '범주를 맞히는 분류'} 문제로 처리했습니다.",
            "features": f"예측에는 target과 id를 제외한 {len(feature_columns)}개 특성을 사용했습니다. 주요 특성: {feature_text}",
            "preprocess": preprocess_text,
            "result": f"검증 {primary_metric} 기준 최고 모델은 {best_model['name']}이며 점수는 {best_model['valid_score']}입니다. {result_summary} {imbalance_note}",
        },
    }


def save_model_result(db: Session, document: Document, ml_result: dict):
    if not document or not ml_result:
        return None

    metrics_payload = {
        key: value
        for key, value in ml_result.items()
        if key != "score_chart"
    }
    best_model = ml_result.get("best_model") or {}
    model_result = ModelResult(
        document_id=document.id,
        target_column=ml_result.get("target_column", ""),
        task_type=ml_result.get("task_type", ""),
        best_model=best_model.get("name", ""),
        primary_metric=ml_result.get("primary_metric", ""),
        valid_score=str(best_model.get("valid_score", "")),
        metrics_json=json.dumps(metrics_payload, ensure_ascii=False),
    )
    db.add(model_result)
    db.commit()
    db.refresh(model_result)
    return model_result


@app.get("/")
def home():
    return RedirectResponse(url="/login", status_code=302)


@app.get("/signup")
def signup_page(request: Request):
    return templates.TemplateResponse(request, "signup.html", {"error": None})


@app.post("/signup")
def signup(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    existing_user = (
        db.query(User)
        .filter((User.username == username) | (User.email == email))
        .first()
    )

    if existing_user:
        return templates.TemplateResponse(
            request,
            "signup.html",
            {"error": "이미 사용 중인 아이디 또는 이메일입니다."},
        )

    new_user = User(username=username, email=email, password=password)
    db.add(new_user)
    db.commit()

    return RedirectResponse(url="/login", status_code=302)


@app.get("/login")
def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {"error": None})


@app.post("/login")
def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = (
        db.query(User)
        .filter(User.username == username, User.password == password)
        .first()
    )

    if user is None:
        return templates.TemplateResponse(
            request,
            "login.html",
            {"error": "아이디 또는 비밀번호가 틀렸습니다."},
        )

    documents, categories = dashboard_records(db)

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        dashboard_context(
            user.username,
            documents,
            categories,
            pipeline_stats=build_pipeline_stats(documents, categories),
        ),
    )


@app.get("/dashboard")
def dashboard_page(request: Request, username: str, db: Session = Depends(get_db)):
    documents, categories = dashboard_records(db)

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        dashboard_context(
            username,
            documents,
            categories,
            pipeline_stats=build_pipeline_stats(documents, categories),
        ),
    )


@app.get("/eda")
def eda_page(request: Request, username: str, db: Session = Depends(get_db)):
    documents, categories = dashboard_records(db)
    csv_documents = [
        document for document in documents
        if document.title.lower().endswith(".csv")
    ]

    return templates.TemplateResponse(
        request,
        "eda.html",
        dashboard_context(
            username,
            documents,
            categories,
            active_view="eda",
            pipeline_stats=build_pipeline_stats(documents, categories),
            csv_documents=csv_documents,
        ),
    )


@app.get("/preprocess")
def preprocess_page(request: Request, username: str, db: Session = Depends(get_db)):
    documents, categories = dashboard_records(db)
    csv_documents = [
        document for document in documents
        if document.title.lower().endswith(".csv")
    ]

    return templates.TemplateResponse(
        request,
        "preprocess.html",
        dashboard_context(
            username,
            documents,
            categories,
            active_view="preprocess",
            pipeline_stats=build_pipeline_stats(documents, categories),
            csv_documents=csv_documents,
        ),
    )


@app.get("/preprocess/{document_id}")
def preprocess_detail_page(
    request: Request,
    document_id: int,
    username: str,
    db: Session = Depends(get_db),
):
    documents, categories = dashboard_records(db)
    csv_documents = [
        document for document in documents
        if document.title.lower().endswith(".csv")
    ]
    selected_document = (
        db.query(Document)
        .filter(Document.id == document_id)
        .first()
    )
    error = None
    preprocess_result = {"charts": {}, "csv_profiles": [], "preprocess_summary": None}

    if not selected_document or not selected_document.title.lower().endswith(".csv"):
        error = "전처리할 CSV 파일을 찾지 못했습니다."
    else:
        try:
            preprocess_result = build_preprocess_charts([selected_document])
        except Exception as exc:
            error = str(exc)

    return templates.TemplateResponse(
        request,
        "preprocess.html",
        dashboard_context(
            username,
            documents,
            categories,
            active_view="preprocess",
            error=error,
            pipeline_stats=build_pipeline_stats(documents, categories),
            eda_charts=preprocess_result["charts"],
            csv_profiles=preprocess_result["csv_profiles"],
            csv_documents=csv_documents,
            selected_document=selected_document,
            preprocess_summary=preprocess_result["preprocess_summary"],
            analysis_header=build_analysis_header(selected_document, "preprocess"),
        ),
    )


@app.get("/eda/{document_id}")
def analysis_select_page(
    request: Request,
    document_id: int,
    username: str,
    db: Session = Depends(get_db),
):
    documents, categories = dashboard_records(db)
    csv_documents = [
        document for document in documents
        if document.title.lower().endswith(".csv")
    ]
    selected_document = (
        db.query(Document)
        .filter(Document.id == document_id)
        .first()
    )
    error = None
    csv_profile = None

    if not selected_document or not selected_document.title.lower().endswith(".csv"):
        error = "분석할 CSV 파일을 찾지 못했습니다."
    else:
        try:
            import pandas as pd

            selected_frame = normalize_csv_dataframe_for_analysis(pd.read_csv(StringIO(selected_document.content)), pd)
            csv_profile = build_csv_profile(selected_document, selected_frame, pd)
        except Exception:
            csv_profile = None

    return templates.TemplateResponse(
        request,
        "analysis_select.html",
        dashboard_context(
            username,
            documents,
            categories,
            active_view="eda",
            error=error,
            pipeline_stats=build_pipeline_stats(documents, categories),
            csv_documents=csv_documents,
            selected_document=selected_document,
            source_info=build_document_source_info(selected_document),
            feature_guide=build_feature_guide(selected_document),
            schema_info=build_churn_schema_info(selected_document),
            csv_profile=csv_profile,
            analysis_header=build_analysis_header(selected_document, "select"),
        ),
    )


@app.get("/eda/{document_id}/charts")
def eda_detail_page(
    request: Request,
    document_id: int,
    username: str,
    db: Session = Depends(get_db),
):
    documents, categories = dashboard_records(db)
    csv_documents = [
        document for document in documents
        if document.title.lower().endswith(".csv")
    ]
    selected_document = (
        db.query(Document)
        .filter(Document.id == document_id)
        .first()
    )
    error = None
    eda_result = {"charts": {}, "csv_profiles": []}

    if not selected_document or not selected_document.title.lower().endswith(".csv"):
        error = "분석할 CSV 파일을 찾지 못했습니다."
    else:
        try:
            eda_result = build_eda_charts([selected_document], include_storage_charts=False)
        except Exception as exc:
            error = str(exc)

    return templates.TemplateResponse(
        request,
        "eda.html",
        dashboard_context(
            username,
            documents,
            categories,
            active_view="eda",
            error=error,
            pipeline_stats=build_pipeline_stats(documents, categories),
            eda_charts=eda_result["charts"],
            csv_profiles=eda_result["csv_profiles"],
            csv_documents=csv_documents,
            selected_document=selected_document,
            analysis_header=build_analysis_header(selected_document, "eda"),
        ),
    )


@app.get("/eda/{document_id}/modeling")
def modeling_detail_page(
    request: Request,
    document_id: int,
    username: str,
    target_column: str = "",
    db: Session = Depends(get_db),
):
    documents, categories = dashboard_records(db)
    csv_documents = [
        document for document in documents
        if document.title.lower().endswith(".csv")
    ]
    selected_document = (
        db.query(Document)
        .filter(Document.id == document_id)
        .first()
    )
    error = None
    ml_result = None
    selected_target_column = target_column.strip()
    target_candidates = []

    if not selected_document or not selected_document.title.lower().endswith(".csv"):
        error = "모델링할 CSV 파일을 찾지 못했습니다."
    else:
        try:
            import pandas as pd

            selected_frame = normalize_csv_dataframe_for_analysis(pd.read_csv(StringIO(selected_document.content)), pd)
            column_profile = classify_csv_columns(selected_frame, pd)
            target_candidates = column_profile["target_candidates"]
            default_target = (
                selected_target_column
                or churn_column(selected_frame, "churn")
                or heart_column(selected_frame, "HeartDisease")
                or (target_candidates[0]["column"] if target_candidates else "")
            )
            if not default_target:
                error = (
                    "모델링에 사용할 target 후보를 찾지 못했습니다. "
                    "날짜/고유 ID/상수 컬럼을 제외하면 예측 대상으로 삼을 컬럼이 부족합니다. "
                    "전처리 페이지에서 컬럼 유형과 고유값 수를 먼저 확인하세요."
                )
            else:
                selected_target_column = default_target
                ml_result = build_ml_analysis(selected_document, default_target)
                save_model_result(db, selected_document, ml_result)
        except Exception as exc:
            db.rollback()
            error = str(exc)

    return templates.TemplateResponse(
        request,
        "modeling.html",
        dashboard_context(
            username,
            documents,
            categories,
            active_view="eda",
            error=error,
            pipeline_stats=build_pipeline_stats(documents, categories),
            csv_documents=csv_documents,
            selected_document=selected_document,
            target_column=selected_target_column,
            target_candidates=target_candidates,
            ml_result=ml_result,
            analysis_header=build_analysis_header(selected_document, "modeling"),
        ),
    )


@app.get("/documents/new")
def new_document_page(request: Request, username: str, db: Session = Depends(get_db)):
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    categories = db.query(Category).order_by(Category.name).all()

    return templates.TemplateResponse(
        request,
        "document_create.html",
        dashboard_context(username, documents, categories, active_view="create"),
    )


@app.get("/documents/kaggle")
def new_kaggle_document_page(
    request: Request,
    username: str,
    kaggle_search: str = "",
    selected_dataset_id: str = "",
    db: Session = Depends(get_db),
):
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    categories = db.query(Category).order_by(Category.name).all()
    search_keyword = kaggle_search.strip()
    search_results = []
    error = None

    if search_keyword:
        try:
            search_results = search_kaggle_datasets(search_keyword)
            if not search_results:
                error = f"'{search_keyword}' 검색어로 찾은 CSV Kaggle 데이터셋이 없습니다."
        except Exception as exc:
            error = str(exc)

    return templates.TemplateResponse(
        request,
        "kaggle_create.html",
        dashboard_context(
            username,
            documents,
            categories,
            active_view="create",
            rag_error=error,
            kaggle_dataset_id=selected_dataset_id,
            kaggle_search_keyword=search_keyword,
            kaggle_search_results=search_results,
        ),
    )


@app.get("/documents/crawl")
def new_crawl_document_page(request: Request, username: str, db: Session = Depends(get_db)):
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    categories = db.query(Category).order_by(Category.name).all()

    return templates.TemplateResponse(
        request,
        "crawl_create.html",
        dashboard_context(username, documents, categories, active_view="create"),
    )


@app.get("/documents/search-page")
def search_document_page(request: Request, username: str, db: Session = Depends(get_db)):
    documents, categories = dashboard_records(db)

    return templates.TemplateResponse(
        request,
        "document_search.html",
        dashboard_context(username, documents, categories, active_view="search"),
    )


@app.get("/rag")
def rag_page(request: Request, username: str, db: Session = Depends(get_db)):
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    categories = db.query(Category).order_by(Category.name).all()

    return templates.TemplateResponse(
        request,
        "rag.html",
        dashboard_context(username, documents, categories, active_view="rag"),
    )


@app.get("/documents/list")
def document_list_page(
    request: Request,
    username: str,
    category_id: int = None,
    db: Session = Depends(get_db),
):
    ensure_default_churn_document(db)
    categories = db.query(Category).order_by(Category.name).all()

    if category_id:
        documents = (
            db.query(Document)
            .filter(Document.category_id == category_id)
            .order_by(Document.created_at.desc())
            .all()
        )
    else:
        documents = db.query(Document).order_by(Document.created_at.desc()).all()

    return templates.TemplateResponse(
        request,
        "document_list.html",
        dashboard_context(
            username,
            documents,
            categories,
            active_view="list",
            selected_category_id=category_id,
        ),
    )


@app.post("/documents")
async def create_document(
    request: Request,
    username: str = Form(...),
    title: str = Form(""),
    display_name: str = Form(""),
    content: str = Form(""),
    category_id: int = Form(None),
    attachments: list[UploadFile] | None = File(None),
    db: Session = Depends(get_db),
):
    error = None
    saved_count = 0
    documents_to_save = []
    categories = db.query(Category).order_by(Category.name).all()

    cleaned_title = title.strip()
    cleaned_display_name = display_name.strip()
    cleaned_content = content.strip()

    if cleaned_content:
        direct_document = Document(
            title=cleaned_title or "직접 작성 문서",
            display_name=cleaned_display_name or cleaned_title or "직접 작성 문서",
            content=cleaned_content,
            category_id=category_id,
            source_type="manual",
        )
        db.add(direct_document)
        documents_to_save.append(direct_document)

    uploaded_files = [
        upload_file
        for upload_file in (attachments or [])
        if upload_file.filename
    ]

    for upload_file in uploaded_files:
        try:
            uploaded_payload = await extract_upload_text(upload_file)
        except RuntimeError as exc:
            error = str(exc)
            continue

        file_document = Document(
            title=upload_file.filename or "업로드 문서",
            display_name=cleaned_display_name or infer_display_name_from_csv_content(upload_file.filename, uploaded_payload["content"]),
            content=uploaded_payload["content"],
            category_id=category_id,
            source_type="upload",
            source_url=upload_file.filename,
            file_path=uploaded_payload.get("file_path"),
            processed_path=uploaded_payload.get("processed_path"),
            storage_uri=uploaded_payload.get("storage_uri"),
        )
        db.add(file_document)
        documents_to_save.append(file_document)

    if not documents_to_save:
        documents = db.query(Document).order_by(Document.created_at.desc()).all()
        return templates.TemplateResponse(
            request,
            "document_create.html",
            dashboard_context(
                username,
                documents,
                categories,
                active_view="create",
                rag_error=error or "직접 작성 내용 또는 업로드 파일이 필요합니다.",
            ),
        )

    db.commit()

    for document in documents_to_save:
        db.refresh(document)
        try:
            upsert_document(document)
            saved_count += 1
        except RuntimeError as exc:
            error = str(exc)

    documents = db.query(Document).order_by(Document.created_at.desc()).all()

    return templates.TemplateResponse(
        request,
        "document_create.html",
        dashboard_context(
            username,
            documents,
            categories,
            active_view="create",
            rag_error=error,
            message=f"문서 {saved_count}개가 저장되었습니다.",
        ),
    )


@app.post("/documents/kaggle")
def create_kaggle_document(
    request: Request,
    username: str = Form(...),
    dataset_id: str = Form(...),
    category_id: int = Form(None),
    db: Session = Depends(get_db),
):
    categories = db.query(Category).order_by(Category.name).all()
    cleaned_dataset_id = dataset_id.strip()

    try:
        kaggle_result = download_and_preprocess_dataset(cleaned_dataset_id)
        document = Document(
            title=kaggle_result["document_title"],
            display_name=infer_display_name_from_csv_content(kaggle_result["document_title"], kaggle_result["processed_csv"]),
            content=kaggle_result["processed_csv"],
            category_id=category_id,
            source_type="kaggle",
            source_url=f"https://www.kaggle.com/datasets/{kaggle_result['dataset_id']}",
            processed_path=kaggle_result["processed_path"],
            metadata_path=kaggle_result["metadata_path"],
            storage_uri=kaggle_result.get("storage_uri") or "",
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        upsert_document(document)
        message = (
            f"{cleaned_dataset_id} Kaggle 데이터셋을 수집하고 전처리한 CSV를 저장했습니다. "
            f"metadata: {kaggle_result['metadata_path']}"
        )
        error = None
    except Exception as exc:
        db.rollback()
        message = None
        error = str(exc)

    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    return templates.TemplateResponse(
        request,
        "kaggle_create.html",
        dashboard_context(
            username,
            documents,
            categories,
            active_view="create",
            rag_error=error,
            message=message,
            kaggle_dataset_id=cleaned_dataset_id,
        ),
    )


@app.post("/documents/crawl")
def create_crawl_document(
    request: Request,
    username: str = Form(...),
    url: str = Form(...),
    category_id: int = Form(None),
    db: Session = Depends(get_db),
):
    categories = db.query(Category).order_by(Category.name).all()
    cleaned_url = url.strip()

    try:
        crawl_result = crawl_web_page(cleaned_url)
        document = Document(
            title=crawl_result["document_title"],
            display_name=infer_display_name_from_csv_content(crawl_result["document_title"], crawl_result["document_content"]),
            content=crawl_result["document_content"],
            category_id=category_id,
            source_type="crawl",
            source_url=crawl_result["url"],
            processed_path=crawl_result["processed_path"],
            metadata_path=crawl_result["metadata_path"],
            storage_uri=crawl_result.get("storage_uri") or "",
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        upsert_document(document)

        if crawl_result["document_type"] == "csv_table":
            data_type_message = "HTML 표를 CSV로 변환"
        else:
            data_type_message = "본문 텍스트를 추출"
        message = (
            f"{crawl_result['url']} 페이지에서 {data_type_message}해 저장했습니다. "
            f"metadata: {crawl_result['metadata_path']}"
        )
        error = None
    except Exception as exc:
        db.rollback()
        message = None
        error = str(exc)

    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    return templates.TemplateResponse(
        request,
        "crawl_create.html",
        dashboard_context(
            username,
            documents,
            categories,
            active_view="create",
            rag_error=error,
            message=message,
            crawl_url=cleaned_url,
        ),
    )


@app.get("/documents/search")
def search_documents(
    request: Request,
    username: str,
    keyword: str = "",
    scope: str = "all",
    db: Session = Depends(get_db),
):
    ensure_default_churn_document(db)
    categories = db.query(Category).order_by(Category.name).all()
    search_scope = scope if scope in {"all", "title", "content", "recent"} else "all"
    query = db.query(Document)

    if keyword:
        search_word = f"%{keyword}%"
        if search_scope == "title":
            query = query.filter(Document.title.like(search_word))
        elif search_scope == "content":
            query = query.filter(Document.content.like(search_word))
        else:
            query = query.filter(
                (Document.title.like(search_word))
                | (Document.content.like(search_word))
            )

    query = query.order_by(Document.created_at.desc())
    if search_scope == "recent":
        query = query.limit(5)

    documents = query.all()

    return templates.TemplateResponse(
        request,
        "document_search.html",
        dashboard_context(
            username,
            documents,
            categories,
            active_view="search",
            keyword=keyword,
            search_scope=search_scope,
        ),
    )


@app.post("/rag/ask")
def ask_document_question(
    request: Request,
    username: str = Form(...),
    question: str = Form(...),
    category_id: int = Form(None),
    db: Session = Depends(get_db),
):
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    categories = db.query(Category).order_by(Category.name).all()
    rag_answer = ""
    rag_sources = []
    rag_error = None

    try:
        document_ids = None
        search_documents = documents
        if category_id:
            search_documents = (
                db.query(Document)
                .filter(Document.category_id == category_id)
                .order_by(Document.created_at.desc())
                .all()
            )
            document_ids = [document.id for document in search_documents]

        title_matched_ids = find_title_matched_document_ids(question, search_documents)
        if title_matched_ids:
            document_ids = title_matched_ids
            for document in search_documents:
                if document.id in title_matched_ids:
                    upsert_document(document)

        rag_result = ask_rag(question, document_ids=document_ids)
        rag_answer = rag_result["answer"]
        rag_sources = rag_result["sources"]
    except RuntimeError as exc:
        rag_error = str(exc)

    return templates.TemplateResponse(
        request,
        "rag.html",
        dashboard_context(
            username,
            documents,
            categories,
            active_view="rag",
            rag_question=question,
            rag_answer=rag_answer,
            rag_sources=rag_sources,
            rag_error=rag_error,
        ),
    )


@app.get("/documents/{document_id}")
def document_viewer(
    request: Request,
    document_id: int,
    username: str,
    highlight: str = "",
    db: Session = Depends(get_db),
):
    import json as _json
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        return RedirectResponse(url=f"/documents/list?username={username}", status_code=302)
    attach_display_titles([document])

    try:
        highlight_list = _json.loads(highlight) if highlight else []
    except Exception:
        highlight_list = []

    return templates.TemplateResponse(
        request,
        "viewer.html",
        {
            "username": username,
            "document": document,
            "highlight": highlight_list,
        },
    )


@app.post("/documents/{document_id}/display-name")
def update_document_display_name(
    document_id: int,
    username: str = Form(...),
    display_name: str = Form(""),
    db: Session = Depends(get_db),
):
    document = db.query(Document).filter(Document.id == document_id).first()
    if document:
        cleaned_display_name = display_name.strip()
        document.display_name = cleaned_display_name or infer_display_name_from_csv_content(document.title, document.content or "")
        db.commit()
    return RedirectResponse(url=f"/documents/list?username={username}", status_code=303)


@app.post("/categories")
def create_category(
    request: Request,
    username: str = Form(...),
    name: str = Form(...),
    db: Session = Depends(get_db),
):
    existing = db.query(Category).filter(Category.name == name.strip()).first()
    if existing:
        documents = db.query(Document).order_by(Document.created_at.desc()).all()
        categories = db.query(Category).order_by(Category.name).all()
        return templates.TemplateResponse(
            request,
            "categories.html",
            dashboard_context(
                username,
                documents,
                categories,
                active_view="categories",
                rag_error="이미 존재하는 카테고리 이름입니다.",
            ),
        )

    db.add(Category(name=name.strip()))
    db.commit()

    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    categories = db.query(Category).order_by(Category.name).all()

    return templates.TemplateResponse(
        request,
        "categories.html",
        dashboard_context(
            username,
            documents,
            categories,
            active_view="categories",
            message=f'카테고리 "{name.strip()}"이 생성되었습니다.',
        ),
    )


@app.post("/categories/{category_id}/delete")
def delete_category(
    request: Request,
    category_id: int,
    username: str = Form(...),
    db: Session = Depends(get_db),
):
    category = db.query(Category).filter(Category.id == category_id).first()
    if category:
        db.query(Document).filter(Document.category_id == category_id).update(
            {"category_id": None}
        )
        db.delete(category)
        db.commit()

    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    categories = db.query(Category).order_by(Category.name).all()

    return templates.TemplateResponse(
        request,
        "categories.html",
        dashboard_context(
            username,
            documents,
            categories,
            active_view="categories",
            message="카테고리가 삭제되었습니다.",
        ),
    )


@app.get("/categories")
def categories_page(request: Request, username: str, db: Session = Depends(get_db)):
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    categories = db.query(Category).order_by(Category.name).all()

    return templates.TemplateResponse(
        request,
        "categories.html",
        dashboard_context(username, documents, categories, active_view="categories"),
    )
