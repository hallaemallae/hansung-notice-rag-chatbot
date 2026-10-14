# 한성대학교 한성공지 RAG Chatbot

한성대학교 공지 PDF를 근거로 질문에 답하는 **대화형 RAG Chatbot**입니다.
(2026 AI·SW 융합 7주차 실습)

- **LLM**: Ollama `gemma3:4b` (로컬 실행)
- **Embedding**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Vector DB**: Chroma
- **UI**: Streamlit

## 동작 방식

1. `data/공지*.pdf`를 읽어 700자 단위(겹침 100자)로 나눕니다.
2. 임베딩해서 Chroma에 저장하고, 질문마다 관련 조각 3개를 검색합니다.
3. 직전 질문과 현재 질문을 합쳐 검색하므로 후속 질문도 맥락을 반영합니다.
4. 검색된 문서만 근거로 답하며, 문서에 없으면 "제공된 문서에서 확인할 수 없습니다."라고 답합니다.
5. 답변마다 "검색된 문서 확인"에서 검색 Query와 근거 원문을 볼 수 있습니다.

## 실행 방법

```powershell
# 1. Ollama 설치 (https://ollama.com) 후 모델 받기
ollama pull gemma3:4b

# 2. 가상환경 + 패키지 설치 (Python 3.10 ~ 3.12)
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt

# 3. 실행 → http://localhost:8501
streamlit run app.py
```

처음 실행할 때는 임베딩 모델(약 470MB)을 내려받아서 1~2분 정도 걸립니다.

## 폴더 구조

```text
├─ app.py             # Streamlit RAG Chatbot
├─ requirements.txt
└─ data/
   ├─ 공지1.pdf
   └─ 공지2.pdf
```

다른 공지를 추가하려면 `data/` 폴더에 `공지*.pdf` 이름으로 넣으면 됩니다.
