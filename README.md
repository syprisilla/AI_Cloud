# Public Data RAG Workflow

공공데이터, CSV, PDF, 텍스트 문서를 수집하고 저장한 뒤 VectorDB 기반 RAG 검색과 데이터 분석을 함께 제공하는 FastAPI 웹 서비스입니다.

이 서비스는 문서 업로드, 웹 크롤링, Kaggle 데이터 수집, 카테고리 관리, 문서 검색, RAG 질의응답, CSV EDA, 전처리 점검, 머신러닝 모델링까지 하나의 대시보드 흐름으로 사용할 수 있도록 구성되어 있습니다.

## 주요 기능

- 회원가입 및 로그인
- 카테고리별 문서 관리
- 직접 입력 문서 저장
- PDF, TXT, MD, CSV 파일 업로드
- PDF 텍스트 추출 및 문서화
- 웹 페이지 크롤링 및 HTML 표 CSV 변환
- Kaggle 데이터셋 다운로드 및 기본 전처리
- 저장 문서 검색
  - 전체 검색
  - 제목 검색
  - 본문 검색
  - 최근 문서 보기
- ChromaDB 기반 VectorDB 색인
- 문서 기반 RAG 질의응답
- RAG 답변 근거 문서 및 발췌문 표시
- CSV 데이터 EDA 시각화
- CSV 전처리 요약
- 고객 이탈 및 심장질환 데이터셋 분석 보조
- scikit-learn, XGBoost 기반 모델링 결과 제공

## 기술 스택

- Backend: FastAPI, Uvicorn
- Template: Jinja2
- Database: MySQL, SQLAlchemy, PyMySQL
- VectorDB: ChromaDB
- LLM: Gemini API
- Data Processing: pandas
- Visualization: matplotlib
- Machine Learning: scikit-learn, XGBoost
- File Processing: pypdf
- Data Collection: Kaggle API, urllib 기반 웹 크롤러

## 프로젝트 구조

```text
AI_Cloud/
├── main.py                 # FastAPI 라우팅, 화면 렌더링, 분석 로직
├── rag.py                  # 문서 chunking, ChromaDB 색인, RAG 답변 생성
├── database.py             # MySQL 연결 설정
├── models.py               # User, Category, Document DB 모델
├── crawler_pipeline.py     # 웹 페이지 크롤링 및 표/본문 추출
├── kaggle_pipeline.py      # Kaggle 데이터 다운로드 및 전처리
├── templates/              # Jinja2 HTML 템플릿
├── static/                 # CSS 정적 파일
├── data/                   # 원본/처리 데이터 저장 위치
├── data_files/             # 업로드 CSV 저장 위치
├── chroma_db/              # ChromaDB 영구 저장소
├── requirements.txt        # Python 의존성
└── .env.example            # 환경 변수 예시
```

## 실행 전 준비

Python 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

MySQL에 사용할 데이터베이스를 생성합니다.

```sql
CREATE DATABASE rag_project DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

`.env.example`을 참고해 `.env` 파일을 생성합니다.

```env
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_HOST=localhost
DB_PORT=3306
DB_NAME=rag_project
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
```

Kaggle 데이터 수집 기능을 사용하려면 Kaggle API 토큰도 필요합니다.

- `C:\Users\<사용자>\.kaggle\kaggle.json` 위치에 토큰 저장
- 또는 `KAGGLE_USERNAME`, `KAGGLE_KEY` 환경 변수 설정

## 실행 방법

```bash
uvicorn main:app --reload
```

브라우저에서 아래 주소로 접속합니다.

```text
http://127.0.0.1:8000
```

## 사용 흐름

1. 회원가입 후 로그인합니다.
2. 대시보드에서 문서 생성, Kaggle 수집, 웹 크롤링 중 하나를 선택합니다.
3. 문서나 데이터를 저장하면 자동으로 RAG 검색용 VectorDB에 색인됩니다.
4. 문서 검색 화면에서 전체, 제목, 본문, 최근 문서 기준으로 자료를 확인합니다.
5. RAG 질문 화면에서 저장된 문서를 근거로 질문합니다.
6. CSV 데이터는 전처리, EDA, 모델링 화면에서 분석할 수 있습니다.

## RAG 동작 방식

1. 업로드 또는 수집된 문서를 DB에 저장합니다.
2. 문서 내용을 일정 크기의 chunk로 분할합니다.
3. 각 chunk를 ChromaDB에 저장합니다.
4. 사용자의 질문과 관련된 chunk를 VectorDB에서 검색합니다.
5. 검색된 문서 내용을 Gemini API 프롬프트에 넣어 답변을 생성합니다.
6. 답변과 함께 검색 근거 문서, 발췌문을 화면에 표시합니다.

현재 임베딩은 로컬 해시 기반 임베딩 함수로 구현되어 있어 별도 임베딩 API 없이 동작합니다. 더 높은 의미 검색 정확도가 필요하면 Gemini, OpenAI, HuggingFace 임베딩 등으로 교체할 수 있습니다.

## 주요 화면

- `/signup`: 회원가입
- `/login`: 로그인
- `/dashboard`: 메인 대시보드
- `/documents/new`: 직접 입력 및 파일 업로드
- `/documents/kaggle`: Kaggle 데이터셋 수집
- `/documents/crawl`: 웹 페이지 크롤링
- `/documents/search-page`: 문서 검색
- `/documents/list`: 문서 목록
- `/rag`: RAG 질의응답
- `/eda`: CSV 분석 선택
- `/preprocess`: 전처리 점검

## 참고 사항

- PDF는 텍스트 추출 가능한 문서만 안정적으로 처리됩니다.
- Gemini API 키가 없으면 관련 문서 검색은 가능하지만 LLM 답변 생성은 실패할 수 있습니다.
- Kaggle 다운로드는 Kaggle API 인증이 필요합니다.
- MySQL 연결 정보가 올바르지 않으면 앱 시작 또는 DB 접근 시 오류가 발생합니다.
- `chroma_db/`는 VectorDB 저장소이므로 색인을 초기화하려면 해당 디렉터리를 정리한 뒤 문서를 다시 저장해야 합니다.
