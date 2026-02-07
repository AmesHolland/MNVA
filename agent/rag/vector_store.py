from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable
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
from agent.rag.processor import DocumentProcessor

# rerank_api_fn 约定签名：
# 输入 query + documents(list[str])，输出按相关性降序的 index 列表，或 (index, score) 列表
RerankFn = Callable[[str, List[str], int], List[Tuple[int, float]]]
@dataclass
class WeaviateConnConfig:
    """
    Weaviate 连接配置（优先环境变量，其次默认值）
    - 本地：weaviate.connect_to_local()
    - 远程/自定义：weaviate.connect_to_custom()
    """
    url: Optional[str] = None                 # 例如: http://localhost:8080 或 https://xxx.weaviate.network
    api_key: Optional[str] = None             # 若启用鉴权
    grpc_port: int = 50051                    # v4 client 用 gRPC，端口需开放
    http_port: Optional[int] = None           # 可从 url 自动解析；不填则走 url 的 port 或默认 80/443
    http_host: Optional[str] = None           # 可从 url 自动解析
    secure: Optional[bool] = None             # 可从 url scheme 自动解析
    skip_init_checks: bool = False

    @staticmethod
    def from_env() -> "WeaviateConnConfig":
        return WeaviateConnConfig(
            url=os.getenv("WEAVIATE_URL"),
            api_key=os.getenv("WEAVIATE_API_KEY"),
            grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
            http_port=int(os.getenv("WEAVIATE_HTTP_PORT")) if os.getenv("WEAVIATE_HTTP_PORT") else None,
            http_host=os.getenv("WEAVIATE_HTTP_HOST"),
            secure=(os.getenv("WEAVIATE_SECURE").lower() == "true") if os.getenv("WEAVIATE_SECURE") else None,
            skip_init_checks=(os.getenv("WEAVIATE_SKIP_INIT_CHECKS", "false").lower() == "true"),
        )


class WeaviateNewsVectorStore:
    """
    面向“海洋新闻 Document 切片”的向量库封装：
    - 追加写入（embedding + upsert/add）
    - 混合检索（向量 + BM25 关键词），alpha 可调
    - 支持 metadata filters（Weaviate v4 Filter）
    """

    def __init__(
        self,
        config: RAGConfig,
        embeddings: Optional[Embeddings] = None,
        conn: Optional[WeaviateConnConfig] = None,
        index_name: Optional[str] = None,
        text_key: str = "text",
        metadata_keys: Optional[List[str]] = None,
    ):
        self.config = config
        self.embeddings: Embeddings = embeddings or get_embeddings()

        self.conn = conn or WeaviateConnConfig.from_env()

        # 你可以在 RAGConfig 里加 weaviate_index_name；这里做容错：
        self.index_name = (
            index_name
            or getattr(config, "weaviate_index_name", None)
            or os.getenv("WEAVIATE_INDEX_NAME", "MarineNews")
        )
        self.text_key = text_key

        # 对齐 processor.py 的 metadata（news_id/title/publish_date/source/url/chunk_idx/total_chunks）
        self.metadata_keys = metadata_keys or [
            "chunk_id",
            "news_id",
            "title",
            "publish_date",
            "source",
            "url",
            "chunk_idx",
            "total_chunks",
        ]

        self.client = self._connect()
        self.store = self._init_store()

    # -------------------------
    # Connection / Init
    # -------------------------
    def _connect(self) -> weaviate.WeaviateClient:
        """
        优先：
        1) 若显式提供 WEAVIATE_URL -> connect_to_custom（更通用）
        2) 否则 -> connect_to_local（localhost:8080 + gRPC:50051）
        """
        if not self.conn.url:
            # 本地默认连接（要求 8080/50051 可用）
            return weaviate.connect_to_local(
                auth_credentials=AuthApiKey(self.conn.api_key) if self.conn.api_key else None
            )

        parsed = urlparse(self.conn.url)
        scheme = parsed.scheme or "http"
        secure = self.conn.secure if self.conn.secure is not None else (scheme == "https")
        host = self.conn.http_host or parsed.hostname or "localhost"

        # http port：优先显式，其次 url 里解析，否则 80/443
        http_port = (
            self.conn.http_port
            if self.conn.http_port is not None
            else (parsed.port if parsed.port is not None else (443 if secure else 80))
        )

        auth = AuthApiKey(self.conn.api_key) if self.conn.api_key else None

        return weaviate.connect_to_custom(
            http_host=host,
            http_port=http_port,
            http_secure=secure,
            grpc_host=host,
            grpc_port=self.conn.grpc_port,
            grpc_secure=secure,
            auth_credentials=auth,
            skip_init_checks=self.conn.skip_init_checks,
        )

    def _init_store(self) -> WeaviateVectorStore:
        """
        连接（复用）到既有 collection；embedding 用于：
        - add_documents 时生成向量
        - query 时为 query 生成向量（同时 Weaviate 内部会做 hybrid）
        """
        return WeaviateVectorStore(
            client=self.client,
            index_name=self.index_name,
            text_key=self.text_key,
            embedding=self.embeddings,
            attributes=self.metadata_keys,
        )

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    # -------------------------
    # Store (ingest / upsert)
    # -------------------------
    def store_documents(
        self,
        docs: List[Document],
        *,
        tenant: Optional[str] = None,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        将 processor.py 产出的 Document 切片写入 Weaviate（可反复调用追加新数据）

        返回：写入对象 ids（chunk_id）
        """
        # prepared, ids = self._prepare_docs(docs)

        # WeaviateVectorStore 支持 add_documents；kwargs 会透传给 weaviate 的写入逻辑
        # tenant 用于 multi-tenancy（如你后续需要）：
        if tenant:
            kwargs["tenant"] = tenant
        if batch_size:
            kwargs["batch_size"] = batch_size
        self.store.add_documents(docs, **kwargs)
        print("Documents Embedding Store Success!")
        # try:
        #     self.store.add_documents(docs,  **kwargs)
        # except TypeError:
        #     # 某些版本只支持 add_texts：这里做兼容兜底
        #     texts = [d.page_content for d in prepared]
        #     metadatas = [d.metadata for d in prepared]
        #     self.store.add_texts(texts=texts, metadatas=metadatas, **kwargs)

        # return ids

    # -------------------------
    # Query / Retrieval (Hybrid)
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
        """
        混合检索（Hybrid Search）：
        - alpha=1 纯向量
        - alpha=0 纯关键词(BM25)
        """
        if tenant:
            kwargs["tenant"] = tenant
        return self.store.similarity_search(
            query,
            k=k,
            alpha=alpha,
            filters=filters,
            **kwargs,
        )

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
        """
        返回 (Document, score)；score 含义由 langchain-weaviate 适配层提供（便于排序/调试）
        """
        if tenant:
            kwargs["tenant"] = tenant
        return self.store.similarity_search_with_score(
            query,
            k=k,
            alpha=alpha,
            filters=filters,
            **kwargs,
        )

    def as_retriever(
        self,
        *,
        k: int = 5,
        alpha: float = 0.6,
        filters: Optional[Filter] = None,
        tenant: Optional[str] = None,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        返回 LangChain retriever，方便你在 RAG chain 里直接用 retriever.invoke(question)

        注意：WeaviateVectorStore 的 similarity_search 本身就是 hybrid，
        所以 retriever 走的也是同一个机制（alpha 可调）。
        """
        kw = dict(search_kwargs or {})
        kw.setdefault("k", k)
        kw.setdefault("alpha", alpha)
        if filters is not None:
            kw["filters"] = filters
        if tenant is not None:
            kw["tenant"] = tenant
        return self.store.as_retriever(search_kwargs=kw)

    # -------------------------
    # ReRanking (同步精简版：检索+重排合并，无异步)
    # -------------------------
    def _siliconflow_rerank_sync(self, query: str, cand_texts: List[str], top_n: int = 8) -> List[Tuple[int, float]]:
        """
        私有同步重排方法：直接调用SiliconFlow Qwen3-Reranker-8B，返回(原始索引, 重排分数)
        适配原重排逻辑的输出格式，无异步、无冗余格式转换
        """
        if not cand_texts:
            return []
        api_key = os.getenv("SILICONFLOW_API_KEY")
        if not api_key:
            print("[rerank] 未配置SILICONFLOW_API_KEY，跳过重排")
            return [(i, 0.0) for i in range(min(top_n, len(cand_texts)))]

        print(f"[rerank] 对{len(cand_texts)}条候选结果重排（SiliconFlow Qwen3-Reranker-8B）...")
        # 构造同步请求
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": "Qwen/Qwen3-Reranker-8B",
            "query": query,
            "documents": cand_texts,
            "top_n": top_n,
            "return_documents": False,
        }
        try:
            # 同步POST请求，添加超时（避免卡壳）
            resp = requests.post(
                url="https://api.siliconflow.cn/v1/rerank",
                headers=headers,
                json=payload,
                timeout=30  # 30秒超时，网络慢可适当调大
            )
            resp.raise_for_status()  # 触发HTTP错误（如404/500）
            data = resp.json()
            # 直接提取(原始索引, 重排分数)，一步到位
            reranked = [(res["index"], float(res.get("score", 0.0))) for res in data["results"]]
            print(f"[rerank] 重排完成，保留Top-{len(reranked)}条")
            return reranked
        except Exception as e:
            print(f"[rerank] 调用失败: {str(e)[:100]}，返回原始检索顺序")
            return [(i, 0.0) for i in range(min(top_n, len(cand_texts)))]

    def query_with_rerank(
            self,
            query: str,
            *,
            k: int = 8,  # 最终返回的精排结果数
            fetch_k: int = 50,  # 前置检索候选数（建议k*5~k*10）
            alpha: float = 0.6,  # 混合检索权重：1=纯向量，0=纯BM25
            filters: Optional[Filter] = None,
            tenant: Optional[str] = None,
            **kwargs: Any,
    ) -> List[Document]:
        """
        检索+重排合并版（同步精简，无异步）：Weaviate混合检索 → SiliconFlow重排 → TopK筛选
        无需外部传参，一键调用，重排分数写入metadata，异常自动降级
        """
        # 1. 混合检索：从Weaviate召回候选集（fetch_k >> k，保证重排效果）
        if tenant:
            kwargs["tenant"] = tenant
        print(f"[retrieve] 混合检索召回候选集(fetch_k={fetch_k}, alpha={alpha})...")
        cands = self.store.similarity_search(query, k=fetch_k, alpha=alpha, filters=filters, **kwargs)
        if not cands:
            print("[retrieve] 未召回任何文档，返回空")
            return []

        # 2. 提取候选文本，调用同步重排方法
        cand_texts = [doc.page_content for doc in cands]
        reranked_idx_score = self._siliconflow_rerank_sync(query, cand_texts, top_n=k)

        # 3. 按重排结果筛选TopK，写入重排分数到metadata
        final_docs = []
        for idx, score in reranked_idx_score[:k]:
            doc = cands[idx]
            doc.metadata["rerank_score"] = round(score, 6)  # 保留6位小数，便于调试
            doc.metadata["fetch_k"] = fetch_k  # 新增元数据，记录检索候选数
            final_docs.append(doc)

        return final_docs

    # -------------------------
    # Optional utilities
    # -------------------------
    @staticmethod
    def build_filter_equal(field: str, value: Any) -> Filter:
        """快速构造 metadata 等值过滤：Filter.by_property(field).equal(value)"""
        return Filter.by_property(field).equal(value)

    @staticmethod
    def build_filter_and(*filters: Filter) -> Filter:
        """多个 Filter 做 AND 组合"""
        if not filters:
            raise ValueError("filters 不能为空")
        f = filters[0]
        for nxt in filters[1:]:
            f = f & nxt
        return f

    def clear_all_weaviate_objects(
            self,
            *,
            tenant: Optional[str] = None,
            where: Optional[Filter] = None,
            max_loops: int = 10_000,
    ) -> int:
        """
        清空当前 index_name(collection) 内的所有对象，但保留 collection 本身（schema 不删）。
        - Weaviate delete_many 单次有 QUERY_MAXIMUM_RESULTS 上限（默认 10k），所以循环执行直到删完。
        - 返回：累计删除数量（尽力而为，少数版本/配置下可能只能返回近似值）。
        """
        # 默认匹配全量（你的数据里 chunk_id 必然存在）
        where = where or Filter.by_property("chunk_id").like("*")

        # v4 推荐用 collections.use
        collection = self.client.collections.use(self.index_name)

        total_deleted = 0

        def _to_int(x: Any) -> int:
            try:
                return int(x)
            except Exception:
                return 0

        def _extract_matches(resp: Any) -> int:
            # 兼容 dict / pydantic / dataclass 等返回结构
            if resp is None:
                return 0
            if isinstance(resp, dict):
                for k in ("matches", "match_count", "objects_matched", "totalMatches"):
                    if k in resp:
                        return _to_int(resp.get(k))
            for attr in ("matches", "match_count", "objects_matched", "totalMatches"):
                if hasattr(resp, attr):
                    return _to_int(getattr(resp, attr))
            return 0

        def _extract_deleted(resp: Any) -> int:
            if resp is None:
                return 0
            if isinstance(resp, dict):
                for k in ("successful", "successes", "objects_deleted", "deleted", "totalDeleted"):
                    if k in resp:
                        return _to_int(resp.get(k))
            for attr in ("successful", "successes", "objects_deleted", "deleted", "totalDeleted"):
                if hasattr(resp, attr):
                    return _to_int(getattr(resp, attr))
            return 0

        kwargs: Dict[str, Any] = {}
        if tenant:
            kwargs["tenant"] = tenant

        for _ in range(max_loops):
            # dry_run 先看是否还有可删对象（避免无意义循环）
            try:
                preview = collection.data.delete_many(where=where, dry_run=True, **kwargs)
            except TypeError:
                # 兼容某些版本参数差异
                preview = collection.data.delete_many(where=where, dry_run=True, verbose=False, **kwargs)

            matches = _extract_matches(preview)
            if matches <= 0:
                break

            # 正式删除（不要 verbose=True，部分版本/集群会触发 delete_many 响应解析问题）
            try:
                result = collection.data.delete_many(where=where, **kwargs)
            except TypeError:
                result = collection.data.delete_many(where=where, verbose=False, **kwargs)

            deleted = _extract_deleted(result)
            if deleted <= 0:
                # 有些版本不回传 deleted 数，至少用 matches 做近似累加，避免总为 0
                deleted = min(matches, 10_000)

            total_deleted += deleted

        return total_deleted

def rebuild():
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType

    # 连接Weaviate
    client = weaviate.connect_to_local()

    # 校验连接状态
    assert client.is_ready()

    # 2. 删除旧集合（谨慎执行！会清空所有数据）
    client.collections.delete("MarineNews")
    print("旧集合已删除")

    # 3. 重建集合，配置所有字段与向量参数
    collection = client.collections.create(
        name="MarineNews",
        # 向量配置：根据你使用的嵌入模型选择距离算法（cosine为通用选择）
        vectorizer_config=Configure.Vectorizer.none(),  # 手动传入向量（LangChain场景常用）
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE
        ),
        # 定义所有属性字段（严格匹配你的 metadata_keys）
        properties=[
            Property(name="chunk_id", data_type=DataType.TEXT),
            Property(name="news_id", data_type=DataType.INT),
            Property(name="title", data_type=DataType.TEXT),
            Property(name="publish_date", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="url", data_type=DataType.TEXT),
            Property(name="chunk_idx", data_type=DataType.INT),  # 修复为整数类型
            Property(name="total_chunks", data_type=DataType.INT),
        ],
        # 可选：配置倒排索引，优化过滤/检索性能
        # inverted_index_config=Configure.inverted_index(
        #     index_timestamps=True,
        #     index_property_length=True
        # )
    )

    print("新集合 MarineNews 创建完成，所有字段已配置")

    # 关闭连接
    client.close()

if __name__ == '__main__':

    # cfg = RAGConfig()
    # processor = DocumentProcessor(cfg)
    #
    # docs = processor.load_documents_from_mysql(
    #     start_time="2025-01-11",
    #     end_time="2025-12-31",
    # )
    #
    # vs = WeaviateNewsVectorStore(cfg)
    # try:
    #     vs.store_documents(docs, batch_size=200)
    #     # 混合检索：alpha 越大越偏“语义向量”，越小越偏“关键词(BM25)”
    #     hits = vs.query_with_rerank("deep sea mining", k=8, alpha=0.8)
    #     for d in hits:
    #         print(d.metadata["news_id"], d.metadata["chunk_id"], d.page_content)
    # finally:
    #     vs.close()

    vs = WeaviateNewsVectorStore(RAGConfig())
    print(datetime.now())
    try:
        docs = vs.query_with_rerank("deep sea mining", k=10, alpha=0.8)
        print(datetime.now())
        for d in docs:
             print(
                 d.metadata["news_id"],
                 d.metadata["chunk_idx"],
                 d.metadata["title"],
                 d.metadata["publish_date"],
                 d.metadata["url"],
                 "\n",
                 d.page_content)
    finally:
        vs.close()

    # ==== 清空数据库所有内容 ====
    # vs = WeaviateNewsVectorStore(cfg)
    # try:
    #     deleted = vs.clear_all_weaviate_objects()  # 清空当前 index_name(Collection) 的全部数据
    #     print("deleted:", deleted)
    # finally:
    #     vs.close()

    # 过滤示例：只查某来源
    # from weaviate.classes.query import Filter
    #
    # flt = Filter.by_property("source").equal("xxx")
    # hits2 = vs.query("海洋污染", k=5, alpha=0.6, filters=flt)

    # === 数据库删除重建 ===
    # rebuild()
