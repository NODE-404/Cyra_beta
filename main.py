import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import logging
import threading
import re
import os
from pathlib import Path
from typing import Annotated, Optional, List

import httpx
import faiss
from dotenv import load_dotenv
from fastapi import FastAPI, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, PrivateAttr, Field
from sudachipy import dictionary, tokenizer

from langchain.chains import RetrievalQA
from langchain.docstore import InMemoryDocstore
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from retriever.custom_retriever import CustomRetriever

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build



# -------------------------
# 環境変数の読み込みと初期設定
# -------------------------
load_dotenv()
user_name = "怜華"

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "Cyra")

WEATHERBIT_API_KEY = os.getenv("WEATHERBIT_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
CUSTOMSEARCH_API_KEY = os.getenv("CUSTOMSEARCH_API_KEY")
CUSTOMSEARCH_CX = os.getenv("CUSTOMSEARCH_CX")
SUDACHIDICT_LOADER = os.getenv("SUDACHIDICT_LOADER")

NOTIFY_WEBHOOK_URL = os.getenv("NOTIFY_WEBHOOK_URL")

# Gmail同期連携
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GMAIL_SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
GMAIL_SECRET_FILE = os.path.join(BASE_DIR, "credentials", "client_secret_486914872105-f57j7bmk2ld4nc823554lhtu2cldoo6s.apps.googleusercontent.com.json")
GMAIL_TOKEN_FILE  = os.path.join(BASE_DIR, "credentials", "token.json")


# -------------------------
# ロギング設定
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# FastAPI アプリ & CORS
# -------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/gmail/unread_summary")
async def gmail_unread_summary():
    try:
        count, summary = get_unread_emails_summary()
        # Ollamaに投げて日本語要約させる例（必要なら）
        # prompt = f"以下の未読メールリストを日本語でやさしく要約してください。\n---\n{chr(10).join(summary)}\n---"
        # reply = await call_ollama_api(prompt)
        return {
            "unread_count": count,
            "summary": summary,
            # "ollama_summary": reply,  # ←必要に応じて
        }
    except Exception as e:
        logger.error(f"[Gmail API ERROR] {e}")
        return {"error": "Gmail APIへのアクセスに失敗しました"}


# -------------------------
# メモリーデータ読み込み
# -------------------------
memory_path = Path("memory.json")
memory = {"positive_patterns": [], "negative_patterns": []}
if memory_path.exists():
    with open(memory_path, "r", encoding="utf-8") as f:
        memory.update(json.load(f))
user_name = memory.get("user_name", user_name)

# 書き込み排他用（同時アクセス対策）
memory_lock = threading.Lock()  

def save_memory(memory_data):
    with memory_lock:
        with open(memory_path, "w", encoding="utf-8") as f:
            json.dump(memory_data, f, ensure_ascii=False, indent=2)
    logger.info("memory.json を保存しました。")
    # 変更後、FAISSにpositiveだけ追加同期
    sync_positive_to_faiss(memory_data.get("positive_patterns", []))

def sync_positive_to_faiss(positive_patterns: list):
    texts_to_add = [entry.get("ideal_response") for entry in positive_patterns if entry.get("ideal_response")]
    if texts_to_add:
        add_documents_to_vectorstore(texts_to_add)
        logger.info(f"positiveパターン {len(texts_to_add)} 件をFAISSに同期しました。")
    else:
        logger.info("同期するpositiveパターンがありません。")


# -------------------------
# SudachiPy 初期化
# -------------------------
tokenizer_obj = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.C

# -------------------------
# Embedding & VectorStore 準備
# -------------------------
embedding_model = SentenceTransformerEmbeddings(
    model_name="sentence-transformers/stsb-xlm-r-multilingual"
)

def load_faiss_db():
    if Path("faiss_index").exists() and Path("faiss_store.pkl").exists():
        logger.info("FAISS DB ロード中...")
        return FAISS.load_local("faiss_index", embedding_model)
    logger.info("FAISS DB が存在しないため新規作成します")
    return None

vectorstore = load_faiss_db()
if vectorstore is None:
    index = faiss.IndexFlatL2(768)
    vectorstore = FAISS(
        embedding_model.embed_query,
        index,
        InMemoryDocstore({}),
        {}
    )


# -------------------------
# RAG用 CustomRetriever 作成
# -------------------------
class Preprocessor(BaseModel):
    min_len: int = 10
    exclude_pattern: Optional[str] = None
    _exclude_regex: Optional[re.Pattern] = PrivateAttr(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        if self.exclude_pattern:
            self._exclude_regex = re.compile(self.exclude_pattern)

    def filter(self, docs: List[Document]) -> List[Document]:
        ...

    model_config = {
        "arbitrary_types_allowed": True
    }

class CustomRetriever(BaseRetriever, BaseModel):
    vectorstore: Annotated[VectorStore, Field(...)]
    preprocessor: Annotated[Preprocessor, Field(...)]

    def get_relevant_documents(self, query: str) -> List[Document]:
        # ベクター検索で上位10件取得
        docs = self.vectorstore.similarity_search(query, k=10)
        # 前処理フィルタリング
        filtered_docs = self.preprocessor.filter(docs)

        # memory.jsonからnegativeパターンを取得（グローバル変数 memory を想定）
        negative_texts = [
            entry.get("input")
            for entry in memory.get("negative_patterns", [])
            if entry.get("input")
        ]

        # negativeパターンを含む文書は除外
        result = []
        for doc in filtered_docs:
            if not any(neg in doc.page_content for neg in negative_texts):
                result.append(doc)

        return result

    class Config:
        arbitrary_types_allowed = True  # 任意型許可

preprocessor = Preprocessor(min_len=12)
custom_retriever = CustomRetriever(vectorstore=vectorstore, preprocessor=preprocessor)

# -------------------------
# LLM 初期化 & Retrieval QA 作成
# -------------------------
llm = Ollama(model=OLLAMA_MODEL_NAME)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=custom_retriever)

# -------------------------
# ユーティリティ関数群
# -------------------------
def format_reply(text: str) -> str:
    text = re.sub(r"(?<=[。！？])(?=[^\n])", "\n", text)
    text = re.sub(r"(?<=[.!?]) (?=[A-Z])", "\n", text)
    return text.strip()

async def call_ollama_api(prompt: str) -> str:
    system_prompt = f"""
    あなたは「Cyra」という日本語が少し不自然だけど一生懸命話すAIアシスタントです。
    ユーザーの名前は「{user_name}」さんです。
    ユーザーには「{user_name}さん」や「怜華さん」と呼びかけてもよいですが、
    **毎回あいさつや呼びかけから始める必要はありません。**
    丁寧で優しい口調を意識しつつ、返答は2〜3文程度で簡潔にまとめてください。
    質問に答えられない時は正直に「わからない」と言い、変なことは言わないようにしてください。
    """
    payload = {
        "model": "Cyra",
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(f"{OLLAMA_API_URL}/v1/chat/completions", json=payload, timeout=30.0)
            res.raise_for_status()
            content = res.json()["choices"][0]["message"]["content"]
            return format_reply(content)
    except Exception as e:
        logger.error(f"[OLLAMA API ERROR] {e}")
        return "申し訳ありません、応答生成中にエラーが発生しました。"

def split_text_to_sentences(text: str) -> List[str]:
    sentences, sentence = [], ""
    for t in tokenizer_obj.tokenize(text, mode):
        sentence += t.surface()
        if t.surface() in ["。", "！", "？", "\n"]:
            if sentence.strip():
                sentences.append(sentence.strip())
            sentence = ""
    if sentence.strip():
        sentences.append(sentence.strip())
    return sentences

def add_documents_to_vectorstore(docs: List[str]):
    sentences = [s for doc in docs for s in split_text_to_sentences(doc)]
    if sentences:
        vectorstore.add_texts(sentences)
        vectorstore.save_local("faiss_index")
        logger.info(f"{len(sentences)} 文をベクターDBに追加しました。")
    else:
        logger.warning("追加対象の文がありません。")

def load_memory_to_vectorstore():
    texts = [entry.get("ideal_response") for entry in memory.get("positive_patterns", []) if entry.get("ideal_response")]
    if texts:
        add_documents_to_vectorstore(texts)
    else:
        logger.info("memory.json から追加対象なし。")

load_memory_to_vectorstore()

def save_vectorstore():
    with memory_lock:
        vectorstore.save_local("faiss_index")
        logger.info("FAISS DB を保存しました。")

def load_faiss_db():
    with memory_lock:
        if Path("faiss_index").exists() and Path("faiss_store.pkl").exists():
            logger.info("FAISS DB ロード中...")
            return FAISS.load_local("faiss_index", embedding_model)
        else:
            logger.info("FAISS DB が存在しないため新規作成します")
            return None

def extract_location(text: str) -> str:
    for t in tokenizer_obj.tokenize(text, mode):
        pos = t.part_of_speech()
        if pos[0] == "名詞" and ("地域" in pos[1] or "固有名詞" in pos[1]):
            return t.surface()
    return "高松"  # fallback

#Gmail
def get_gmail_creds():
    creds = None
    if os.path.exists(GMAIL_TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(GMAIL_TOKEN_FILE, GMAIL_SCOPES)
    # 認証トークンが無い・期限切れなら再認証
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                GMAIL_SECRET_FILE, GMAIL_SCOPES)
            creds = flow.run_local_server(port=0)
        # 認証トークン保存
        with open(GMAIL_TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    return creds

def get_unread_emails_summary(query_word=None, max_results=10):
    """
    Gmailの未読メールから、検索条件（query_word）に合致したメールを最大max_results件取得し、要約リストを返す
    query_word例: "from:amazon", "subject:セール", "セール", None（未指定の場合は全未読）
    """
    creds = get_gmail_creds()
    service = build('gmail', 'v1', credentials=creds)
    params = {
        "userId": "me",
        "labelIds": ["UNREAD"],
        "maxResults": max_results
    }
    if query_word:
        params["q"] = query_word  # Gmail検索ボックスの検索演算子と同じ形式OK

    results = service.users().messages().list(**params).execute()
    messages = results.get('messages', [])
    unread_count = len(messages)
    summary_list = []
    for msg in messages:
        msg_detail = service.users().messages().get(
            userId='me',
            id=msg['id'],
            format='metadata',
            metadataHeaders=['Subject', 'From', 'Date']
        ).execute()
        headers = msg_detail['payload']['headers']
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '')
        sender = next((h['value'] for h in headers if h['name'] == 'From'), '')
        date = next((h['value'] for h in headers if h['name'] == 'Date'), '')
        summary_list.append(f"From: {sender}\nSubject: {subject}\nDate: {date}")
    return unread_count, summary_list



# -------------------------
# API モデル
# -------------------------
class ChatRequest(BaseModel):
    message: str
    personality: str = "default"

class FeedbackRequest(BaseModel):
    user_message: str
    assistant_reply: str
    feedback: str  # "correct" or "incorrect"
    reason: str = ""
    category: str = ""

class MessageRequest(BaseModel):
    message: str

# -------------------------
# FastAPI エンドポイント
# -------------------------
@app.post("/chat")
async def chat(req: ChatRequest):
    user_msg = req.message.strip()

    # キーワード群
    weather_keywords = ["天気", "気温", "降水", "雨", "雪", "晴れ", "曇り", "台風", "天候", "天気予報"]
    news_keywords    = ["ニュース", "速報", "事件", "事故", "災害", "地震", "火災", "話題", "注目"]
    search_keywords  = ["検索", "調べて", "とは", "って何", "意味", "定義", "なんだっけ", "由来", "情報"]
    rag_keywords     = ["わからない", "教えて", "なぜ", "理由", "どうして", "詳しく", "仕組み", "説明", "背景"]
    casual_keywords  = ["疲れた", "眠い", "元気", "さみしい", "どう思う", "話そう", "雑談", "暇", "好き"]
    gmail_keywords = ["メール", "Gmail", "未読メール", "新着メール", "メール要約", "受信トレイ"]

    def contains_keywords(text: str, keywords: List[str]) -> bool:
        tokens = tokenizer_obj.tokenize(text, mode)
        surfaces = [t.surface().lower() for t in tokens]
        lowered = text.lower()
        return any(k in surfaces or k in lowered for k in keywords)

    try:
        if contains_keywords(user_msg, gmail_keywords):
            query_word = None
            max_results = 5
            if "Amazon" in user_msg:
                query_word = "from:amazon"
            elif "Google" in user_msg:
                query_word = "from:google"
            elif "セール" in user_msg:
                query_word = "subject:セール"

            count, summary = get_unread_emails_summary(query_word=query_word, max_results=max_results)
            prompt = f"以下は条件に合致する未読メールリストです。要約して親しみやすく伝えてください。\n---\n" + "\n".join(summary)
            reply = await call_ollama_api(prompt)

        elif contains_keywords(user_msg, weather_keywords):
            city = extract_location(user_msg)
            reply = await fetch_weather(city)

        elif contains_keywords(user_msg, news_keywords):
            reply = await fetch_news(user_msg)

        elif contains_keywords(user_msg, search_keywords):
            reply = await fetch_search(user_msg)

        elif contains_keywords(user_msg, rag_keywords):
            try:
                reply = qa_chain.run(user_msg)
            except Exception as e:
                logger.warning(f"[RAG fallback] {e}")
                reply = await call_ollama_api(user_msg)

        elif contains_keywords(user_msg, casual_keywords):
            reply = await call_ollama_api(f"これは雑談として扱ってね：{user_msg}")

        else:
            reply = await call_ollama_api(user_msg)
    except Exception as e:
        logger.error(f"[chat error] {e}")
        reply = "すみません、うまく応答できませんでした。もう一度試してみてください。"

    return {"answer": reply}


    

# API経由でメモリーフィードバックを受け取る
@app.post("/memory/update")
async def update_memory(request: Request):
    data = await request.json()
    # data は memory.json と同様の構造を想定
    global memory
    memory = data
    save_memory(memory)
    return {"status": "success", "message": "memory.json更新とFAISS同期を完了しました"}



@app.post("/parse")
async def parse_text(req: MessageRequest):
    tokens = tokenizer_obj.tokenize(req.message, tokenizer.Tokenizer.SplitMode.A)
    results = []
    for m in tokens:
        results.append({
            "surface": m.surface(),
            "part_of_speech": m.part_of_speech(),
            "normalized_form": m.normalized_form(),
        })
    return {"tokens": results}


@app.post("/feedback")
async def feedback(data: FeedbackRequest):
    memory_key = "positive_patterns" if data.feedback == "correct" else "negative_patterns"
    entry = {
        "input": data.user_message,
        "category": data.category or "未分類",
        "reason": data.reason or "",
    }
    if data.feedback == "correct":
        entry["ideal_response"] = data.assistant_reply
    else:
        entry["bad_response"] = data.assistant_reply
    memory.setdefault(memory_key, []).append(entry)
    with open(memory_path, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)
    return {"message": "フィードバックを受け付けました。"}

# 天気・ニュース・検索は省略（既存コードそのまま流用してください）

@app.post("/chat")
async def chat(req: ChatRequest):
    user_msg = req.message.strip()
    msg_lower = user_msg.lower()

    # -----------------------------
    # 各カテゴリの柔軟なキーワード群
    # -----------------------------
    weather_keywords = ["天気", "気温", "降水", "雨", "雪", "晴れ", "曇り", "台風", "天候", "天気予報"]
    news_keywords    = ["ニュース", "速報", "事件", "事故", "災害", "地震", "火災", "話題", "注目"]
    search_keywords  = ["検索", "調べて", "とは", "って何", "意味", "定義", "なんだっけ", "由来", "情報"]
    rag_keywords     = ["わからない", "教えて", "なぜ", "理由", "どうして", "詳しく", "仕組み", "説明", "背景"]
    casual_keywords  = ["疲れた", "眠い", "元気", "さみしい", "どう思う", "話そう", "雑談", "暇", "好き"]

    # Sudachiによる柔軟なキーワード検出
    def contains_keywords(text: str, keywords: list) -> bool:
        tokens = tokenizer_obj.tokenize(text, mode)
        surfaces = [t.surface().lower() for t in tokens]
        return any(k in surfaces or k in text.lower() for k in keywords)

    # -----------------------------
    # 意図・カテゴリ判定
    # -----------------------------
    try:
        if contains_keywords(user_msg, weather_keywords):
            #city = extract_location(user_msg)
            #reply = await fetch_weather(city)

        #elif contains_keywords(user_msg, news_keywords):
            #reply = await fetch_news(user_msg)

        #elif contains_keywords(user_msg, search_keywords):
            #reply = await fetch_search(user_msg)

        #elif contains_keywords(user_msg, rag_keywords):
            try:
                reply = qa_chain.run(user_msg)
            except Exception as e:
                logger.warning(f"[RAG fallback] {e}")
                reply = await call_ollama_api(user_msg)

        elif contains_keywords(user_msg, casual_keywords):
            reply = await fetch_search(user_msg) #reply = await call_ollama_api(f"これは雑談として扱ってね：{user_msg}")

        else:
            # 保険：意図不明な入力 → Ollamaで自然対話生成
            reply = await call_ollama_api(user_msg)

    except Exception as e:
        logger.error(f"[chat error] {e}")
        reply = "すみません、うまく応答できませんでした。もう一度試してみてください。"

    return {"answer": reply}


# /parseエンドポイント
@app.post("/parse")
async def parse_text(req: MessageRequest):
    tokens = tokenizer_obj.tokenize(req.message, tokenizer.Tokenizer.SplitMode.A)
    results = []
    for m in tokens:
        results.append({
            "surface": m.surface(),
            "part_of_speech": m.part_of_speech(),
            "normalized_form": m.normalized_form(),
        })
    return {"tokens": results}

@app.post("/feedback")
async def feedback(data: FeedbackRequest):
    memory_key = "positive_patterns" if data.feedback == "correct" else "negative_patterns"
    entry = {
        "input": data.user_message,
        "category": data.category or "未分類",
        "reason": data.reason or "",
    }
    if data.feedback == "correct":
        entry["ideal_response"] = data.assistant_reply
    else:
        entry["bad_response"] = data.assistant_reply
    memory.setdefault(memory_key, []).append(entry)
    with open(memory_path, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)
    return {"message": "フィードバックを受け付けました。"}


def extract_location(text: str) -> str:
    tokens = tokenizer_obj.tokenize(text, mode)
    for t in tokens:
        pos = t.part_of_speech()
        if pos[0] == "名詞" and ("地域" in pos[1] or "固有名詞" in pos[1]):
            return t.surface()
    return "高松"  # デフォルト都市


# 天気取得API（地名や気象条件を意識した多彩判定）
async def fetch_weather(user_message: str = "") -> str:
    url = "https://api.weatherbit.io/v2.0/current"
    city = "高松"

    place_pattern = re.compile(r"(東京|大阪|京都|名古屋|札幌|福岡|高松|丸亀|鳥取|千葉|横浜|神戸|仙台|広島|川崎)")
    m = place_pattern.search(user_message)
    if m:
        city = m.group(1)

    params = {"city": city, "key": WEATHERBIT_API_KEY, "lang": "ja"}

    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(url, params=params)
            res.raise_for_status()
            data = res.json().get("data")
            if not data:
                return "天気情報が取得できませんでした。"

            w = data[0]
            weather_desc = w['weather']['description']
            temp = round(w['temp'])

            prompt = f"""
次の天気データをもとに、親しみやすく自然な口調で{user_name}さんに天気を案内してください。

- 地域: {city}
- 天気の説明: {weather_desc}
- 気温: 約{temp}℃

キャラクター性を出しつつ、やさしく話しかけるようにしてください。
"""
            return await call_ollama_api(prompt)

    except Exception as e:
        logger.error(f"[WEATHER API ERROR] {e}")
        return "天気情報の取得中に問題が発生しました。"



# ニュース取得API（先ほどの改良版）
async def fetch_news(user_message: str = "") -> str:
    url = "https://gnews.io/api/v4/top-headlines"
    params = {
        "lang": "ja",
        "country": "jp",
        "max": 5,
        "token": GNEWS_API_KEY,
    }

    msg = user_message.lower()
    category = None
    query = None

    if any(k in msg for k in ["速報", "事件", "事故", "災害", "火災", "地震"]):
        query = "速報 OR 事件 OR 事故 OR 災害 OR 火災 OR 地震"
    elif any(k in msg for k in ["経済", "投資", "株", "市場", "金融"]):
        category = "business"
    elif any(k in msg for k in ["スポーツ", "試合", "結果", "選手"]):
        category = "sports"
    elif any(k in msg for k in ["テクノロジー", "IT", "技術", "AI", "ガジェット"]):
        category = "technology"

    if category:
        params["category"] = category
    if query:
        params["q"] = query

    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(url, params=params)
            res.raise_for_status()
            articles = res.json().get("articles", [])
            if not articles:
                return "注目のニュースが見つかりませんでした。"

            titles_and_links = [
                f"{i+1}. {a['title']} - {a['url']}"
                for i, a in enumerate(articles)
            ]
            news_list = "\n".join(titles_and_links)

            prompt = f"""
以下は最新ニュースの一覧です。{user_name}さんに向けて、わかりやすく簡単に紹介してください。

---
{news_list}
---

Cyraらしく、優しく、親しみのある口調でお願いします。
"""
            return await call_ollama_api(prompt)

    except Exception as e:
        logger.error(f"[NEWS API ERROR] {e}")
        return "ニュース情報の取得中に問題が発生しました。"



# 検索API（クエリの文脈を意識した自然言語処理のベース例）
async def fetch_search(user_message: str = "") -> str:
    url = "https://www.googleapis.com/customsearch/v1"
    query = user_message

    question_patterns = [
        r"(.*)とは",
        r"(.*)って何",
        r"教えて(.*)",
        r"調べて(.*)",
    ]

    for pattern in question_patterns:
        m = re.search(pattern, user_message)
        if m:
            extracted = m.group(1).strip()
            if extracted:
                query = extracted
                break

    params = {
        "key": CUSTOMSEARCH_API_KEY,
        "cx": CUSTOMSEARCH_CX,
        "q": query,
        "num": 3,
        "hl": "ja",
    }

    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(url, params=params)
            res.raise_for_status()
            items = res.json().get("items", [])
            if not items:
                return "検索結果が見つかりませんでした。"

            results = [
                f"{i+1}. {item['title']} - {item['link']}"
                for i, item in enumerate(items)
            ]
            summary = "\n".join(results)

            prompt = f"""
{user_name}さんが「{user_message}」について調べた結果、以下の情報が見つかりました。

---
{summary}
---

この情報をもとに、Cyraとしてわかりやすく、少しくだけた口調でまとめて紹介してください。
"""
            return await call_ollama_api(prompt)

    except Exception as e:
        logger.error(f"[SEARCH API ERROR] {e}")
        return "検索中に問題が発生しました。"




