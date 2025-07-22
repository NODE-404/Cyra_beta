# custom_retriever.py
from typing import List
from pydantic import BaseModel, Field
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore
from langchain.schema import BaseRetriever


# 前処理＋フィルタリング用クラス（必要に応じてルール追加可能）
class Preprocessor:
    def filter(self, docs: List[Document]) -> List[Document]:
        # 仮フィルタリング: ページ内容が20文字以上のもののみ通す
        return [doc for doc in docs if len(doc.page_content.strip()) > 20]


# Retriever本体クラス
class CustomRetriever(BaseRetriever, BaseModel):
    vectorstore: VectorStore = Field(...)
    preprocessor: Preprocessor = Field(default_factory=Preprocessor)

    def get_relevant_documents(self, query: str) -> List[Document]:
        # 類似検索 → 前処理でフィルタ → 結果返却
        docs = self.vectorstore.similarity_search(query, k=10)
        return self.preprocessor.filter(docs)

    @classmethod
    def create(cls, vectorstore: VectorStore, preprocessor: Preprocessor = None):
        return cls(
            vectorstore=vectorstore,
            preprocessor=preprocessor or Preprocessor()
        )

    class Config:
        arbitrary_types_allowed = True  # 任意の型(VectorStoreなど)を許可
