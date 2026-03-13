from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
import weaviate
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from weaviate.auth import AuthApiKey
from weaviate.classes.query import Filter
from weaviate.collections.classes.config import VectorDistances

from agent.config.rag_config import RAGConfig
from agent.rag.model import get_embeddings


@dataclass
class SimpleWeaviateConfig:
    """
    简化版 Weaviate 配置
    优先读取环境变量：
    - WEAVIATE_URL
    - WEAVIATE_API_KEY
    - WEAVIATE_GRPC_PORT
    - WEAVIATE_INDEX_NAME
    - WEAVIATE_SKIP_INIT_CHECKS
    """
    url: Optional[str] = None
    api_key: Optional[str] = None
    grpc_port: int = 50051
    index_name: str = "MarineNews"
    skip_init_checks: bool = False

    @staticmethod
    def from_env(default_index_name: str = "MarineNews") -> "SimpleWeaviateConfig":
        return SimpleWeaviateConfig(
            url=os.getenv("WEAVIATE_URL"),
            api_key=os.getenv("WEAVIATE_API_KEY"),
            grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
            index_name=os.getenv("WEAVIATE_INDEX_NAME", default_index_name),
            skip_init_checks=os.getenv("WEAVIATE_SKIP_INIT_CHECKS", "false").lower() == "true",
        )


class WeaviateNewsVectorStore:
    """
    精简版新闻向量库封装
    功能：
    - 写入文档
    - 混合检索
    - 混合检索 + rerank
    - 按 uuid / chunk_id 更新对象
    - 按 uuid / chunk_id 删除对象
    - 批量删除
    - 删除并重建 collection
    """

    DEFAULT_METADATA_KEYS = [
        "chunk_id",
        "news_id",
        "title",
        "publish_date",
        "source",
        "url",
        "chunk_idx",
        "total_chunks",
    ]

    def __init__(
        self,
        config: RAGConfig,
        embeddings: Optional[Embeddings] = None,
        conn: Optional[SimpleWeaviateConfig] = None,
        index_name: Optional[str] = None,
        text_key: str = "text",
        metadata_keys: Optional[List[str]] = None,
    ):
        self.config = config
        self.embeddings = embeddings or get_embeddings()

        default_index_name = (
            index_name
            or getattr(config, "weaviate_index_name", None)
            or "MarineNews"
        )
        self.conn = conn or SimpleWeaviateConfig.from_env(default_index_name=default_index_name)

        self.index_name = self.conn.index_name
        self.text_key = text_key
        self.metadata_keys = metadata_keys or self.DEFAULT_METADATA_KEYS

        self.client = self._connect()
        self.store = self._init_store()

    # -------------------------
    # 基础能力
    # -------------------------
    def __enter__(self) -> "WeaviateNewsVectorStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    def _connect(self) -> weaviate.WeaviateClient:
        """
        没传 WEAVIATE_URL 时默认连本地：
        http://localhost:8080 + grpc:50051
        """
        if not self.conn.url:
            return weaviate.connect_to_local(
                auth_credentials=AuthApiKey(self.conn.api_key) if self.conn.api_key else None
            )

        parsed = urlparse(self.conn.url)
        scheme = parsed.scheme or "http"
        secure = scheme == "https"
        host = parsed.hostname or "localhost"
        http_port = parsed.port or (443 if secure else 80)

        return weaviate.connect_to_custom(
            http_host=host,
            http_port=http_port,
            http_secure=secure,
            grpc_host=host,
            grpc_port=self.conn.grpc_port,
            grpc_secure=secure,
            auth_credentials=AuthApiKey(self.conn.api_key) if self.conn.api_key else None,
            skip_init_checks=self.conn.skip_init_checks,
        )

    def _init_store(self) -> WeaviateVectorStore:
        return WeaviateVectorStore(
            client=self.client,
            index_name=self.index_name,
            text_key=self.text_key,
            embedding=self.embeddings,
            attributes=self.metadata_keys,
        )

    def _build_search_kwargs(
        self,
        *,
        k: int,
        alpha: float,
        filters: Optional[Filter] = None,
        tenant: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        search_kwargs = {
            "k": k,
            "alpha": alpha,
            **kwargs,
        }
        if filters is not None:
            search_kwargs["filters"] = filters
        if tenant is not None:
            search_kwargs["tenant"] = tenant
        return search_kwargs

    # -------------------------
    # ID 设计
    # -------------------------
    @staticmethod
    def _uuid_from_chunk_id(chunk_id: str) -> str:
        """
        用 chunk_id 生成稳定 uuid，便于后续 update/delete
        """
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"marine-news::{chunk_id}"))

    def _get_doc_uuid(self, doc: Document) -> str:
        chunk_id = doc.metadata.get("chunk_id")
        if not chunk_id:
            raise ValueError("Document.metadata 必须包含 chunk_id，便于稳定写入/更新/删除")
        return self._uuid_from_chunk_id(str(chunk_id))

    # -------------------------
    # 写入
    # -------------------------
    def add_documents(
        self,
        docs: List[Document],
        *,
        tenant: Optional[str] = None,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        写入文档，并返回稳定 uuid 列表
        """
        if not docs:
            return []

        ids = [self._get_doc_uuid(doc) for doc in docs]

        if tenant:
            kwargs["tenant"] = tenant
        if batch_size:
            kwargs["batch_size"] = batch_size

        self.store.add_documents(documents=docs, ids=ids, **kwargs)
        return ids

    # 兼容你原来的命名
    def store_documents(
        self,
        docs: List[Document],
        *,
        tenant: Optional[str] = None,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> List[str]:
        return self.add_documents(
            docs,
            tenant=tenant,
            batch_size=batch_size,
            **kwargs,
        )

    # -------------------------
    # 查询
    # -------------------------
    def query(
        self,
        query: str,
        *,
        k: int = 5,
        alpha: float = 0.6,
        filters: Optional[Filter] = None,
        tenant: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        search_kwargs = self._build_search_kwargs(
            k=k,
            alpha=alpha,
            filters=filters,
            tenant=tenant,
            **kwargs,
        )
        return self.store.similarity_search(query, **search_kwargs)

    def query_with_score(
        self,
        query: str,
        *,
        k: int = 5,
        alpha: float = 0.6,
        filters: Optional[Filter] = None,
        tenant: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        search_kwargs = self._build_search_kwargs(
            k=k,
            alpha=alpha,
            filters=filters,
            tenant=tenant,
            **kwargs,
        )
        return self.store.similarity_search_with_score(query, **search_kwargs)

    def as_retriever(
        self,
        *,
        k: int = 5,
        alpha: float = 0.6,
        filters: Optional[Filter] = None,
        tenant: Optional[str] = None,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        kw = dict(search_kwargs or {})
        kw.setdefault("k", k)
        kw.setdefault("alpha", alpha)
        if filters is not None:
            kw["filters"] = filters
        if tenant is not None:
            kw["tenant"] = tenant
        return self.store.as_retriever(search_kwargs=kw)

    # -------------------------
    # Rerank
    # -------------------------
    def _siliconflow_rerank_sync(
        self,
        query: str,
        cand_texts: List[str],
        top_n: int = 8,
    ) -> List[Tuple[int, float]]:
        """
        返回 [(原始候选索引, rerank_score), ...]
        """
        if not cand_texts:
            return []

        api_key = os.getenv("SILICONFLOW_API_KEY")
        if not api_key:
            # 没配置就按原顺序截断返回
            return [(i, 0.0) for i in range(min(top_n, len(cand_texts)))]

        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": "Qwen/Qwen3-Reranker-8B",
            "query": query,
            "documents": cand_texts,
            "top_n": top_n,
            "return_documents": False,
        }

        try:
            resp = requests.post(
                "https://api.siliconflow.cn/v1/rerank",
                headers=headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                (item["index"], float(item.get("score", 0.0)))
                for item in data.get("results", [])
            ]
        except Exception as e:
            print(f"[rerank] 调用失败，降级为原始检索顺序: {e}")
            return [(i, 0.0) for i in range(min(top_n, len(cand_texts)))]

    def query_with_rerank(
        self,
        query: str,
        *,
        k: int = 8,
        fetch_k: int = 50,
        alpha: float = 0.6,
        filters: Optional[Filter] = None,
        tenant: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        流程：
        Weaviate 混合召回 -> SiliconFlow rerank -> 返回 TopK
        """
        cands = self.query(
            query,
            k=fetch_k,
            alpha=alpha,
            filters=filters,
            tenant=tenant,
            **kwargs,
        )
        if not cands:
            return []

        cand_texts = [doc.page_content for doc in cands]
        reranked_idx_score = self._siliconflow_rerank_sync(query, cand_texts, top_n=k)

        final_docs: List[Document] = []
        for idx, score in reranked_idx_score[:k]:
            doc = cands[idx]
            doc.metadata["rerank_score"] = round(score, 6)
            doc.metadata["fetch_k"] = fetch_k
            final_docs.append(doc)

        return final_docs

    # -------------------------
    # 更新 / 删除对象
    # -------------------------
    def _collection(self):
        return self.client.collections.use(self.index_name)

    def update_object(
        self,
        uuid_str: str,
        properties: Dict[str, Any],
        *,
        tenant: Optional[str] = None,
    ) -> None:
        """
        局部更新对象
        """
        kwargs = {"tenant": tenant} if tenant else {}
        self._collection().data.update(
            uuid=uuid_str,
            properties=properties,
            **kwargs,
        )

    def replace_object(
        self,
        uuid_str: str,
        properties: Dict[str, Any],
        *,
        tenant: Optional[str] = None,
    ) -> None:
        """
        全量替换对象
        """
        kwargs = {"tenant": tenant} if tenant else {}
        self._collection().data.replace(
            uuid=uuid_str,
            properties=properties,
            **kwargs,
        )

    def update_by_chunk_id(
        self,
        chunk_id: str,
        properties: Dict[str, Any],
        *,
        tenant: Optional[str] = None,
    ) -> None:
        uuid_str = self._uuid_from_chunk_id(chunk_id)
        self.update_object(uuid_str, properties, tenant=tenant)

    def replace_by_chunk_id(
        self,
        chunk_id: str,
        properties: Dict[str, Any],
        *,
        tenant: Optional[str] = None,
    ) -> None:
        uuid_str = self._uuid_from_chunk_id(chunk_id)
        self.replace_object(uuid_str, properties, tenant=tenant)

    def delete_object(
        self,
        uuid_str: str,
        *,
        tenant: Optional[str] = None,
    ) -> None:
        kwargs = {"tenant": tenant} if tenant else {}
        self._collection().data.delete_by_id(uuid_str, **kwargs)

    def delete_by_chunk_id(
        self,
        chunk_id: str,
        *,
        tenant: Optional[str] = None,
    ) -> None:
        uuid_str = self._uuid_from_chunk_id(chunk_id)
        self.delete_object(uuid_str, tenant=tenant)

    def delete_many(
        self,
        where: Filter,
        *,
        tenant: Optional[str] = None,
    ):
        kwargs = {"tenant": tenant} if tenant else {}
        return self._collection().data.delete_many(where=where, **kwargs)

    def delete_by_news_id(
        self,
        news_id: int,
        *,
        tenant: Optional[str] = None,
    ):
        return self.delete_many(
            where=Filter.by_property("news_id").equal(news_id),
            tenant=tenant,
        )

    def clear_all_objects(
        self,
        *,
        tenant: Optional[str] = None,
        max_loops: int = 10000,
    ) -> int:
        """
        保留 collection，只清空数据
        """
        total_deleted = 0
        where = Filter.by_property("chunk_id").like("*")
        kwargs = {"tenant": tenant} if tenant else {}
        collection = self._collection()

        for _ in range(max_loops):
            preview = collection.data.delete_many(where=where, dry_run=True, **kwargs)
            matches = self._extract_count(preview, keys=["matches", "match_count", "objects_matched", "totalMatches"])
            if matches <= 0:
                break

            result = collection.data.delete_many(where=where, **kwargs)
            deleted = self._extract_count(result, keys=["successful", "successes", "objects_deleted", "deleted", "totalDeleted"])
            total_deleted += deleted if deleted > 0 else min(matches, 10000)

        return total_deleted

    @staticmethod
    def _extract_count(resp: Any, keys: List[str]) -> int:
        if resp is None:
            return 0
        if isinstance(resp, dict):
            for key in keys:
                if key in resp:
                    try:
                        return int(resp[key])
                    except Exception:
                        return 0
        for key in keys:
            if hasattr(resp, key):
                try:
                    return int(getattr(resp, key))
                except Exception:
                    return 0
        return 0

    # -------------------------
    # Collection 管理
    # -------------------------
    def recreate_collection(self) -> None:
        """
        删除并重建 collection
        """
        from weaviate.classes.config import Configure, Property, DataType

        if self.client.collections.exists(self.index_name):
            self.client.collections.delete(self.index_name)

        self.client.collections.create(
            name=self.index_name,
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE
            ),
            properties=[
                Property(name="chunk_id", data_type=DataType.TEXT),
                Property(name="news_id", data_type=DataType.INT),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="publish_date", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="url", data_type=DataType.TEXT),
                Property(name="chunk_idx", data_type=DataType.INT),
                Property(name="total_chunks", data_type=DataType.INT),
            ],
        )

    def add_property(self, name: str, data_type: Any) -> None:
        """
        给 collection 新增字段
        用法：
            from weaviate.classes.config import DataType
            vs.add_property("language", DataType.TEXT)
        """
        from weaviate.classes.config import Property

        self._collection().config.add_property(
            Property(name=name, data_type=data_type)
        )

    # -------------------------
    # Filter 工具
    # -------------------------
    @staticmethod
    def build_filter_equal(field: str, value: Any) -> Filter:
        return Filter.by_property(field).equal(value)

    @staticmethod
    def build_filter_and(*filters: Filter) -> Filter:
        if not filters:
            raise ValueError("filters 不能为空")
        f = filters[0]
        for nxt in filters[1:]:
            f = f & nxt
        return f


if __name__ == "__main__":
    from datetime import datetime

    vs = WeaviateNewsVectorStore(RAGConfig())
    try:
        print("start:", datetime.now())
        docs = vs.query_with_rerank(
            "deep sea mining",
            k=10,
            fetch_k=30,
            alpha=0.8,
        )
        print("end:", datetime.now())

        for d in docs:
            print("=" * 80)
            print("news_id:", d.metadata.get("news_id"))
            print("chunk_id:", d.metadata.get("chunk_id"))
            print("chunk_idx:", d.metadata.get("chunk_idx"))
            print("title:", d.metadata.get("title"))
            print("publish_date:", d.metadata.get("publish_date"))
            print("url:", d.metadata.get("url"))
            print("rerank_score:", d.metadata.get("rerank_score"))
            print(d.page_content[:1000])
    finally:
        vs.close()