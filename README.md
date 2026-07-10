# AI Cloud Data Pipeline

## 1. 서비스 소개

AI Cloud Data Pipeline은 공공데이터, CSV, PDF, 텍스트 문서, 웹 페이지, Kaggle 데이터셋을 수집하고 저장한 뒤 문서 검색, RAG 질의응답, EDA, 전처리 점검, 머신러닝 모델링까지 하나의 웹 대시보드에서 수행할 수 있는 데이터 분석 서비스입니다.

AI 서비스 설계 및 구현 수업에서 여러 데이터를 분석하면서 코드 실행 결과만 확인하는 것보다 데이터의 분포, 변수 간 관계, 주요 영향 요인과 모델 예측 결과를 웹 화면에서 시각적으로 확인하면 분석 내용을 더 쉽게 이해할 수 있다고 생각해 기획하였습니다.

사용자는 다양한 데이터를 수집하거나 업로드할 수 있으며, 문서는 ChromaDB 기반 검색과 RAG에 활용되고 CSV 데이터는 EDA, 전처리 점검, scikit-learn 및 XGBoost 기반 모델링에 활용됩니다.

---

## 2. 주요 기능

| 기능                  | 설명                                                           |
| --------------------- | -------------------------------------------------------------- |
| 회원가입 및 로그인    | 사용자 이름 기반으로 대시보드 접근                             |
| 카테고리 관리         | 문서를 분석 목적별 카테고리로 분류                             |
| 직접 문서 생성        | 제목, 본문, 카테고리를 입력해 문서 저장                        |
| 파일 업로드           | PDF, TXT, MD, CSV 파일 업로드 및 텍스트 추출                   |
| 웹 크롤링             | URL 본문과 HTML 표를 수집하고 표 데이터는 CSV로 변환           |
| Kaggle 수집           | Kaggle 데이터셋 검색, 다운로드, 기본 전처리                    |
| 저장 계층 분리        | 로컬/서버 파일시스템 저장과 OCI Object Storage 업로드 분리     |
| Document Lineage 기록 | source, 원본/가공 파일 경로, metadata, Object Storage URI 저장 |
| VectorDB 색인         | 저장 문서를 ChromaDB에 chunk 단위로 색인                       |
| 문서 검색             | 전체, 제목, 본문, 최근 문서 기준 검색                          |
| RAG 질의응답          | 저장 문서를 근거로 질문에 답변하고 출처 문서 표시              |
| CSV EDA               | 데이터 타입, 결측치, 분포, 상관관계, 주요 차트 확인            |
| 전처리 점검           | 분석 대상 CSV의 컬럼 분류와 전처리 요약 제공                   |
| 머신러닝 모델링       | 분류 모델 학습, 평가 지표 비교, 최고 모델 결과 저장            |
| 배치 수집             | 설정된 Kaggle 데이터셋을 배치로 수집 및 전처리                 |

---

## 3. 사용 시나리오

### 시나리오 1. 공공데이터/문서 수집 후 검색

1. 사용자가 로그인 후 문서 생성 화면에 접속한다.
2. PDF, TXT, MD, CSV 파일을 업로드하거나 직접 문서를 입력한다.
3. 서비스는 파일 내용을 추출하고 원본/가공 파일을 저장한다.
4. 문서 정보와 저장 위치가 MySQL의 `documents` 테이블에 기록된다.
5. 저장된 문서는 ChromaDB에 색인되어 검색과 RAG 질문에 활용된다.

### 시나리오 2. 웹 크롤링과 Kaggle 데이터 수집

1. 사용자가 웹 크롤링 화면에서 URL을 입력한다.
2. 서비스는 웹 페이지의 본문 텍스트와 HTML 표를 수집한다.
3. 표 데이터가 있으면 CSV 파일로 변환하여 저장한다.
4. 사용자가 Kaggle 화면에서 데이터셋 ID를 입력하면 CSV를 다운로드하고 기본 전처리를 수행한다.
5. 수집 결과는 문서 목록, 검색, EDA, 모델링 화면에서 이어서 사용할 수 있다.

### 시나리오 3. CSV 분석과 모델링

1. 사용자가 CSV 문서를 선택해 EDA 화면으로 이동한다.
2. 서비스는 컬럼 타입, 결측치, 분포, 상관관계 등 데이터 프로파일을 생성한다.
3. 사용자는 전처리 화면에서 분석 가능 컬럼과 타깃 후보를 확인한다.
4. 모델링 화면에서 타깃 컬럼을 선택하면 여러 모델을 학습하고 평가한다.
5. 최고 모델, 검증 점수, metric JSON이 `model_results` 테이블에 저장된다.

### 시나리오 4. 저장 문서 기반 RAG 질의응답

1. 사용자가 RAG 화면에서 분석 질문을 입력한다.
2. 서비스는 ChromaDB에서 관련 문서 chunk를 검색한다.
3. 검색된 근거 문서를 기반으로 LLM 답변을 생성한다.
4. 사용자는 답변과 함께 참조된 문서 제목, 발췌문, 문서 링크를 확인한다.

---

## 4. 전체 아키텍처

AI Cloud Data Pipeline은 FastAPI 웹 서버, MySQL 메타데이터 저장소, 로컬/서버 파일시스템 저장소, 선택적 OCI Object Storage, ChromaDB VectorDB, Gemini/OpenAI 호환 LLM 호출, 데이터 분석 파이프라인으로 구성됩니다.

```mermaid
flowchart TD
    User[사용자] --> UI[Jinja2 Web Dashboard]
    Scheduler[배치 스케줄러] --> Batch[batch_collect.py]

    UI --> API[FastAPI Server]
    Batch --> API

    API --> Auth[회원가입/로그인]
    API --> DocAPI[문서 생성/업로드]
    API --> CrawlAPI[웹 크롤링]
    API --> KaggleAPI[Kaggle 수집]
    API --> SearchAPI[문서 검색]
    API --> RagAPI[RAG 질문]
    API --> AnalysisAPI[EDA/전처리/모델링]

    DocAPI --> Extract[PDF/TXT/MD/CSV 텍스트 추출]
    CrawlAPI --> Crawler[crawler_pipeline.py]
    KaggleAPI --> Kaggle[kaggle_pipeline.py]

    Extract --> Storage[Local File Storage]
    Crawler --> Storage
    Kaggle --> Storage

    Storage --> ObjectStorage[OCI Object Storage]
    Storage --> Metadata[Metadata JSON]
    API --> MySQL[(MySQL)]

    MySQL --> Documents[documents]
    MySQL --> Categories[categories]
    MySQL --> Results[model_results]

    Documents --> Chroma[ChromaDB VectorDB]
    RagAPI --> Chroma
    Chroma --> LLM[LLM API]
    LLM --> Answer[RAG 답변 및 근거]

    AnalysisAPI --> Pandas[pandas 전처리]
    AnalysisAPI --> Charts[matplotlib 시각화]
    AnalysisAPI --> ML[scikit-learn/XGBoost 모델링]
    ML --> Results
```

### 사용한 OCI 리소스 목록

| OCI 리소스            | 사용 목적                                              | 프로젝트 설정                                          |
| --------------------- | ------------------------------------------------------ | ------------------------------------------------------ |
| Compute VM Instance   | FastAPI 애플리케이션 실행, 배치 스크립트 실행          | `uvicorn main:app --reload`                            |
| 로컬/서버 파일시스템  | 원본 파일, 가공 파일, metadata, ChromaDB 영구 저장     | `LOCAL_STORAGE_ROOT=data`, `data_files/`, `chroma_db/` |
| Object Storage Bucket | 원본/가공/metadata 파일을 객체 저장소에 백업 또는 공유 | `STORAGE_MODE=oci`, `OCI_BUCKET_NAME`, `OCI_NAMESPACE` |
| Virtual Cloud Network | VM, DB, Object Storage 접근을 위한 네트워크 구성       | OCI 배포 환경에서 구성                                 |
| MySQL Database        | 사용자, 카테고리, 문서 lineage, 모델링 결과 저장       | `DB_HOST`, `DB_NAME`, `documents`, `model_results`     |
| IAM Policy/API Key    | Object Storage 업로드 권한 및 OCI SDK 인증             | `OCI_CONFIG_FILE`, `OCI_CONFIG_PROFILE`                |

기본 개발 환경에서는 `STORAGE_MODE=local`로 실행되며, 서버 배포 시에도 애플리케이션이 접근 가능한 로컬 파일시스템 경로를 저장소로 사용합니다. 필요한 경우 `STORAGE_MODE=oci`로 Object Storage 업로드를 활성화할 수 있습니다.

---

## 5. Workflow 다이어그램

```mermaid
flowchart TD
    Start([START]) --> Login[로그인/회원가입]
    Login --> Dashboard[대시보드]

    Dashboard --> Manual[직접 문서 생성]
    Dashboard --> Upload[파일 업로드]
    Dashboard --> Crawl[웹 크롤링]
    Dashboard --> Kaggle[Kaggle 수집]

    Manual --> Save[문서 저장]
    Upload --> Extract[파일 텍스트 추출]
    Crawl --> WebParse[본문/표 추출]
    Kaggle --> Preprocess[CSV 다운로드 및 전처리]

    Extract --> Save
    WebParse --> Save
    Preprocess --> Save

    Save --> DB[MySQL 메타데이터 기록]
    Save --> FileStore[파일 저장]
    FileStore --> OptionalOCI{STORAGE_MODE=oci?}
    OptionalOCI -->|yes| OCI[Object Storage 업로드]
    OptionalOCI -->|no| LocalOnly[로컬 파일 유지]

    DB --> Vector[ChromaDB 색인]
    Vector --> Search[문서 검색]
    Vector --> RAG[RAG 질문]
    DB --> EDA[EDA/전처리]
    DB --> Modeling[모델링]

    RAG --> Response[근거 포함 답변]
    Modeling --> Result[model_results 저장]
    Response --> End([END])
    Result --> End
```

---

## 6. 데이터 흐름 상세 설명

이미지 평가 기준의 핵심 흐름인 `수집 소스 -> 저장 위치 -> 가공 로직 -> 제공 방식`은 다음과 같습니다.

```mermaid
flowchart LR
    Source[수집 소스] --> Collect[수집/추출]
    Collect --> Store[저장 위치]
    Store --> Process[가공 로직]
    Process --> Serve[제공 방식]

    Source --> S1[파일 업로드]
    Source --> S2[직접 입력]
    Source --> S3[웹 URL]
    Source --> S4[Kaggle Dataset]

    Store --> T1[data/]
    Store --> T2[data_files/]
    Store --> T3[chroma_db/]
    Store --> T4[MySQL]
    Store --> T5[OCI Object Storage]

    Process --> P1[PDF 텍스트 추출]
    Process --> P2[HTML 본문/표 파싱]
    Process --> P3[CSV 전처리]
    Process --> P4[Chunking 및 VectorDB 색인]
    Process --> P5[EDA/모델 학습]

    Serve --> V1[문서 목록/검색]
    Serve --> V2[RAG 답변]
    Serve --> V3[EDA 차트]
    Serve --> V4[전처리 리포트]
    Serve --> V5[모델링 결과]
```

### 6.1 수집 소스

| 수집 소스       | 처리 파일/모듈        | 설명                                         |
| --------------- | --------------------- | -------------------------------------------- |
| 직접 입력       | `main.py`             | 사용자가 제목과 본문을 입력해 문서 생성      |
| 파일 업로드     | `main.py`, `pypdf`    | PDF, TXT, MD, CSV 파일 업로드 및 텍스트 추출 |
| 웹 페이지       | `crawler_pipeline.py` | URL 본문 텍스트와 HTML 표 수집               |
| Kaggle 데이터셋 | `kaggle_pipeline.py`  | Kaggle API로 CSV 다운로드 및 기본 전처리     |
| 배치 수집       | `batch_collect.py`    | `.env`에 설정된 Kaggle 데이터셋 자동 수집    |

### 6.2 저장 위치

| 저장 위치          | 저장 내용                                              |
| ------------------ | ------------------------------------------------------ |
| `data/`            | 원본/가공 데이터, metadata JSON, local storage root    |
| `data_files/`      | 업로드 CSV 등 분석 대상 파일                           |
| `chroma_db/`       | ChromaDB 영구 VectorDB 저장소                          |
| MySQL              | 사용자, 카테고리, 문서 metadata, 모델링 결과           |
| OCI Object Storage | `STORAGE_MODE=oci`일 때 원본/가공/metadata 객체 업로드 |

### 6.3 가공 로직

1. 업로드 파일은 확장자에 따라 PDF 텍스트 추출, 텍스트 디코딩, CSV 프로파일링으로 분기됩니다.
2. 웹 크롤링 결과는 본문 문서와 HTML 표 CSV로 분리 저장됩니다.
3. Kaggle 데이터셋은 다운로드 후 CSV 탐색, 결측치/컬럼 정보 기반 기본 전처리 metadata를 생성합니다.
4. 문서 본문은 chunk로 분할되어 ChromaDB에 색인됩니다.
5. CSV 문서는 pandas로 EDA 차트와 전처리 요약을 생성합니다.
6. 모델링 화면에서는 타깃 컬럼 기준으로 분류 모델을 학습하고 평가 결과를 저장합니다.

### 6.4 제공 방식

| 제공 화면/API                 | 설명                                   |
| ----------------------------- | -------------------------------------- |
| `/dashboard`                  | 수집/검색/분석으로 이동하는 메인 화면  |
| `/documents/list`             | 저장된 문서 목록과 source lineage 확인 |
| `/documents/search-page`      | 제목, 본문, 전체, 최근 문서 검색       |
| `/rag`                        | 저장 문서 기반 RAG 질의응답            |
| `/eda`                        | CSV 분석 대상 선택                     |
| `/eda/{document_id}/charts`   | EDA 차트 및 데이터 프로파일 표시       |
| `/preprocess`                 | 전처리 점검 대상 선택                  |
| `/preprocess/{document_id}`   | 컬럼 분류, 결측치, 타깃 후보 확인      |
| `/eda/{document_id}/modeling` | 머신러닝 모델링 결과 표시              |

---

## 7. RAG 처리 흐름

```mermaid
flowchart TD
    A[문서 저장] --> B[텍스트 chunk 분할]
    B --> C[Hash Embedding 생성]
    C --> D[ChromaDB upsert]
    E[사용자 질문] --> F[관련 chunk 검색]
    F --> G[문맥 구성]
    G --> H[LLM 답변 생성]
    H --> I[답변, 근거 문서, 발췌문 제공]
```

### RAG 동작 과정

1. 문서가 생성되거나 수집되면 `rag.py`의 `upsert_document`가 실행됩니다.
2. 본문은 chunk 단위로 나뉘고 metadata와 함께 ChromaDB에 저장됩니다.
3. 사용자가 질문하면 관련 chunk를 검색합니다.
4. 검색된 문서 발췌문으로 context를 구성합니다.
5. LLM API를 호출해 근거 기반 답변을 생성합니다.
6. 답변과 함께 참조 문서 제목, 발췌문, 문서 ID를 화면에 표시합니다.

---

## 8. 기술 스택

| 구분             | 기술                                                            |
| ---------------- | --------------------------------------------------------------- |
| Backend          | FastAPI, Uvicorn                                                |
| Template         | Jinja2                                                          |
| Database         | MySQL, SQLAlchemy, PyMySQL                                      |
| VectorDB         | ChromaDB                                                        |
| LLM              | OpenAI 호환 API 설정, `.env`의 `OPENAI_API_KEY`, `OPENAI_MODEL` |
| Data Processing  | pandas                                                          |
| Visualization    | matplotlib                                                      |
| Machine Learning | scikit-learn, XGBoost                                           |
| File Processing  | pypdf                                                           |
| Data Collection  | Kaggle API, urllib/HTMLParser 기반 웹 크롤러                    |
| Cloud Storage    | OCI Object Storage SDK, 로컬/서버 파일 저장                     |

---

## 9. 프로젝트 구조

```text
AI_Cloud/
├── main.py                 # FastAPI 라우팅, 화면 렌더링, EDA/모델링 로직
├── rag.py                  # 문서 chunking, ChromaDB 색인, RAG 답변 생성
├── database.py             # MySQL 연결 설정
├── models.py               # User, Category, Document, ModelResult DB 모델
├── storage.py              # 로컬/OCI Object Storage 저장 계층
├── batch_collect.py        # 배치 수집 및 전처리 실행 스크립트
├── crawler_pipeline.py     # 웹 페이지 크롤링 및 표/본문 추출
├── kaggle_pipeline.py      # Kaggle 데이터 다운로드 및 전처리
├── templates/              # Jinja2 HTML 템플릿
├── static/                 # CSS 정적 파일
├── data/                   # 원본/처리 데이터 저장 위치
├── data_files/             # 업로드 CSV 저장 위치
├── chroma_db/              # ChromaDB 영구 저장소
├── requirements.txt        # Python 의존성
├── .env.example            # 환경 변수 예시
└── README.md
```

### 폴더별 역할

| 폴더/파일              | 설명                                             |
| ---------------------- | ------------------------------------------------ |
| `main.py`              | 서비스 화면, API 라우팅, 데이터 분석/모델링 제어 |
| `rag.py`               | VectorDB 색인, 검색, RAG 답변 생성               |
| `storage.py`           | 로컬 저장과 OCI Object Storage 업로드 추상화     |
| `crawler_pipeline.py`  | 웹 페이지 본문과 HTML 표 수집                    |
| `kaggle_pipeline.py`   | Kaggle 데이터셋 다운로드 및 전처리               |
| `batch_collect.py`     | 정기 데이터 수집용 배치 실행                     |
| `templates/`           | 대시보드, 문서, RAG, EDA, 모델링 화면            |
| `static/`              | 화면 스타일 CSS                                  |
| `data/`, `data_files/` | 원본/가공 파일 저장소                            |
| `chroma_db/`           | ChromaDB VectorDB 파일                           |

---

## 10. 설치 및 실행 방법

### 1. 프로젝트 클론

```bash
git clone <repository-url>
cd AI_Cloud
```

### 2. 가상환경 생성

```bash
python -m venv venv
```

### 3. 가상환경 실행

Windows PowerShell 기준:

```bash
venv\Scripts\activate
```

macOS 또는 Linux 기준:

```bash
source venv/bin/activate
```

### 4. 패키지 설치

```bash
pip install -r requirements.txt
```

### 5. MySQL 데이터베이스 생성

```sql
CREATE DATABASE rag_project DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### 6. 환경변수 설정

`.env.example` 파일을 복사하여 `.env` 파일을 생성합니다.

Windows PowerShell 기준:

```bash
copy .env.example .env
```

macOS 또는 Linux 기준:

```bash
cp .env.example .env
```

`.env` 파일 예시는 다음과 같습니다.

```env
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_HOST=localhost
DB_PORT=3306
DB_NAME=rag_project

OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini

STORAGE_MODE=local
LOCAL_STORAGE_ROOT=data
OCI_BUCKET_NAME=your_object_storage_bucket
OCI_NAMESPACE=your_namespace
OCI_CONFIG_FILE=
OCI_CONFIG_PROFILE=DEFAULT
OCI_OBJECT_PREFIX=ai-cloud-pipeline

BATCH_CATEGORY_NAME=배치 수집 데이터
BATCH_KAGGLE_DATASETS=blastchar/telco-customer-churn,fedesoriano/heart-failure-prediction
```

OCI Object Storage까지 업로드하려면 `STORAGE_MODE=oci`로 변경하고 OCI CLI config 또는 Instance Principal 환경에 맞게 인증을 준비합니다. Kaggle 수집 기능을 사용하려면 `kaggle.json` 또는 `KAGGLE_USERNAME`, `KAGGLE_KEY` 환경 변수 설정도 필요합니다.

### 7. 서버 실행

```bash
uvicorn main:app --reload
```

### 8. 브라우저 접속

```text
http://127.0.0.1:8000
```

---

## 11. 배치 실행 방법

Kaggle 데이터셋을 자동으로 수집하고 전처리하려면 아래 명령을 실행합니다.

```bash
python batch_collect.py
```

정기 수집이 필요하면 위 명령을 cron 또는 Windows 작업 스케줄러에 등록할 수 있습니다. 여러 데이터셋은 `.env`의 `BATCH_KAGGLE_DATASETS`에 쉼표로 구분하여 입력합니다.

---

## 12. 주요 화면 및 API 구성

| Method   | URL                                | 설명                     |
| -------- | ---------------------------------- | ------------------------ |
| GET      | `/`                                | 로그인 화면으로 이동     |
| GET/POST | `/signup`                          | 회원가입 화면 및 처리    |
| GET/POST | `/login`                           | 로그인 화면 및 처리      |
| GET      | `/dashboard`                       | 메인 대시보드            |
| GET/POST | `/documents/new`, `/documents`     | 직접 입력 및 파일 업로드 |
| GET/POST | `/documents/kaggle`                | Kaggle 데이터셋 수집     |
| GET/POST | `/documents/crawl`                 | 웹 페이지 크롤링         |
| GET      | `/documents/search-page`           | 문서 검색 화면           |
| GET      | `/documents/search`                | 문서 검색 실행           |
| GET      | `/documents/list`                  | 문서 목록                |
| GET      | `/documents/{document_id}`         | 문서 상세 보기           |
| GET/POST | `/rag`, `/rag/ask`                 | RAG 질문 화면 및 답변    |
| GET      | `/eda`                             | EDA 분석 대상 선택       |
| GET      | `/eda/{document_id}/charts`        | EDA 차트 화면            |
| GET      | `/preprocess`                      | 전처리 점검 대상 선택    |
| GET      | `/preprocess/{document_id}`        | 전처리 상세 화면         |
| GET      | `/eda/{document_id}/modeling`      | 모델링 결과 화면         |
| GET/POST | `/categories`                      | 카테고리 목록 및 생성    |
| POST     | `/categories/{category_id}/delete` | 카테고리 삭제            |

---

## 13. 실행 가능한 제출 형태

본 프로젝트는 실행 가능한 형태로 제출하기 위해 다음 파일과 폴더를 포함합니다.

| 파일/폴더             | 설명                                                              |
| --------------------- | ----------------------------------------------------------------- |
| `main.py`             | FastAPI 서버 실행 및 라우팅 파일                                  |
| `requirements.txt`    | 실행에 필요한 Python 패키지 목록                                  |
| `.env.example`        | DB, LLM, OCI, 배치 설정 예시                                      |
| `README.md`           | 서비스 소개, 사용 시나리오, 아키텍처, 실행 방법, 데이터 흐름 설명 |
| `templates/`          | 웹 대시보드 화면 템플릿                                           |
| `static/`             | CSS 정적 파일                                                     |
| `rag.py`              | VectorDB/RAG 처리 코드                                            |
| `storage.py`          | 로컬/OCI 저장 계층 코드                                           |
| `crawler_pipeline.py` | 웹 크롤링 코드                                                    |
| `kaggle_pipeline.py`  | Kaggle 수집 및 전처리 코드                                        |
| `batch_collect.py`    | 배치 수집 실행 코드                                               |

---

## 14. 사용 예시

```text
CSV 파일을 업로드하고 EDA 차트를 확인한다.
Kaggle 데이터셋 ID를 입력해 데이터를 수집한다.
웹 페이지 URL을 입력해 본문과 표 데이터를 저장한다.
저장된 문서에서 "고객 이탈에 영향을 주는 요인을 알려줘"라고 질문한다.
심장질환 데이터셋을 선택하고 타깃 컬럼 기준으로 모델링 결과를 확인한다.
문서 검색 화면에서 최근 수집된 자료와 특정 키워드 포함 문서를 찾는다.
```

---

## 15. 한계점

현재 구현에는 다음과 같은 한계가 있습니다.

- JavaScript 기반 동적 웹페이지는 수집이 어렵습니다.
- 수집·전처리 작업이 많아지면 처리 상태 관리가 어렵습니다.
- Object Storage는 OCI 버킷과 권한 설정이 완료되어야 사용할 수 있습니다.

---

## 16. 향후 개선 방향

향후 다음 기능을 추가하여 데이터 파이프라인 서비스의 완성도를 높일 수 있습니다.

- Playwright를 적용해 동적 웹페이지 수집을 지원합니다.
- 비동기 작업 큐를 도입해 수집·전처리 상태를 관리합니다.
- 로그와 모니터링 대시보드를 추가해 오류를 쉽게 확인합니다.

---

## 17. 공공데이터 API 자동 수집 배치

공공데이터 자동 수집은 매일 오전 6시에 외부 공공데이터 API를 호출해 최신 데이터를 저장하고, 결측치 처리 보고서와 함께 기존 `documents` 테이블에 적재하기 위한 배치 파이프라인입니다. 기존 파일 업로드, Kaggle 수집, 웹 크롤링, RAG, EDA, 전처리, 모델링, OCI Object Storage 기능은 그대로 유지되며, 공공데이터는 `source_type="public_data"` 문서로 추가 또는 갱신됩니다.

### 처리 흐름

```text
공공데이터 API 호출
→ 원본 응답 저장(data/raw/public_data)
→ 원본 CSV 저장(data/raw/public_data)
→ 결측치 분석 및 처리
→ 가공 CSV 저장(data/processed/public_data)
→ metadata JSON 저장
→ MySQL documents 테이블 upsert
→ 기존 전처리/EDA/모델링 화면에서 CSV 선택
```

### 환경변수 설정

`.env`에 아래 값을 설정합니다. API 키와 DB 비밀번호가 포함된 실제 `.env`는 GitHub에 올리지 않습니다.

```env
PUBLIC_DATA_API_URL=
PUBLIC_DATA_API_KEY=
PUBLIC_DATA_API_KEY_PARAM=serviceKey
PUBLIC_DATA_API_FORMAT=json
PUBLIC_DATA_API_PARAMS={"pageNo":1,"numOfRows":100}
PUBLIC_DATA_DATASET_NAME=public-statistics
PUBLIC_DATA_CATEGORY_NAME=공공데이터 자동 수집
PUBLIC_DATA_TARGET_COLUMN=
PUBLIC_DATA_REQUIRED_COLUMNS=
PUBLIC_DATA_MISSING_DROP_THRESHOLD=0.7
PUBLIC_DATA_HTTP_TIMEOUT=30
PUBLIC_DATA_ENABLE_RAG=false
```

`PUBLIC_DATA_API_PARAMS`는 JSON 문자열입니다. API 키 파라미터명이 `serviceKey`가 아니면 `PUBLIC_DATA_API_KEY_PARAM`에 실제 이름을 넣습니다. JSON/XML 응답을 모두 지원하지만, 응답 안에서 반복 레코드 목록을 찾을 수 있어야 합니다.

### 수동 실행

```bash
python batch_collect.py
```

공공데이터 설정이 없으면 배치 로그에 `PUBLIC_DATA_API_URL is not set`으로 표시되고 기존 Kaggle 배치는 계속 실행됩니다. API 실패, 빈 응답, JSON/XML 파싱 실패, 필수 컬럼 누락, 결측치 처리 후 빈 데이터는 저장하지 않고 해당 작업만 rollback합니다.

### Oracle Linux cron 실행

```bash
chmod +x scripts/run_batch.sh
crontab -e
```

매일 오전 6시 실행 예시는 다음과 같습니다.

```cron
0 6 * * * /home/opc/AI_Cloud/scripts/run_batch.sh
```

예시 파일은 `deploy/crontab.example`에 있습니다. 등록 후 확인과 로그 확인은 아래 명령을 사용합니다.

```bash
crontab -l
tail -f logs/batch.log
```

`scripts/run_batch.sh`는 `/home/opc/AI_Cloud`로 이동한 뒤 `.venv/bin/python`으로 `batch_collect.py`를 실행하고, `flock`으로 중복 실행을 방지합니다. 표준 출력과 오류는 `logs/batch.log`에 누적됩니다.

### 결측치 처리 규칙

- 컬럼별 결측치 개수와 비율을 계산합니다.
- `PUBLIC_DATA_TARGET_COLUMN`에 결측치가 있으면 해당 행을 제거합니다.
- 결측률이 `PUBLIC_DATA_MISSING_DROP_THRESHOLD` 이상인 컬럼은 삭제합니다.
- 수치형 컬럼은 중앙값으로 대체합니다.
- 범주형 컬럼은 최빈값으로 대체하고, 최빈값이 없으면 `Unknown`을 사용합니다.
- 날짜형 컬럼은 datetime 변환 후 앞 값 채우기와 뒤 값 채우기를 적용합니다.
- ID, 코드, 주소, 지역명처럼 의미가 중요한 컬럼은 수치형처럼 보여도 중앙값 대체를 피하고 `Unknown`으로 보완합니다.
- 원본 DataFrame은 변경하지 않고 복사본을 처리합니다.

보고서 예시는 다음과 같습니다.

```json
{
  "rows_before": 1000,
  "rows_after": 982,
  "columns_before": 20,
  "columns_after": 18,
  "missing_total_before": 42,
  "missing_total_after": 0,
  "dropped_rows": 18,
  "dropped_columns": ["unused_column"],
  "actions": [
    { "column": "amount", "action": "fill_numeric_median", "value": 120.5 },
    {
      "column": "region_name",
      "action": "fill_identifier_unknown",
      "value": "Unknown"
    }
  ]
}
```

### Object Storage 확인

`STORAGE_MODE=oci` 또는 `object_storage`, `cloud`로 설정되어 있고 OCI 환경변수가 올바르면 원본 응답, 원본 CSV, 가공 CSV, metadata JSON이 기존 `storage.py`를 통해 업로드됩니다. 처리 결과의 `metadata_path` JSON에서 `raw_storage`, `processed_storage`, `metadata_storage` 항목의 `object_storage_uri` 값을 확인합니다.
