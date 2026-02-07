import glob
import os
from datetime import datetime
from typing import List

import pandas as pd
# LangChain 核心导入
from langchain_core.documents import Document
# 文本处理
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent.config.rag_config import RAGConfig
from agent.rag.model import get_embeddings
from agent.rag.mysql_store import MySQLDB


class DocumentProcessor:
    """文档处理器：加载指定时间段的海洋新闻xlsx、分块、绑定全量元数据"""

    def __init__(self, config: RAGConfig):
        self.config = config
        # 文本切片器：适配中英文海洋新闻（保留原分隔符）
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", ",", " ", ""]
        )
        self.embeddings = get_embeddings()  # 嵌入模型
        self.vector_store = None
        self.prefix = config.news_id_prefix  # 新闻ID前缀
        # 核心列名映射：兼容中英文列名（你的文件是英文，可根据需要修改）
        self.COLS_MAP = {
            "title": ["标题", "Title", "title"],
            "publish_date": ["发布时间", "PublishTime", "publish_date"],
            "content": ["内容", "Content", "content"],
            "source": ["来源", "Source", "source"],
            "url": ["详情页URL", "URL", "url", "详情页url"]
        }

    def _get_col_name(self, df_cols, target_cols):
        """容错获取列名：从df列中匹配目标列（忽略大小写/中英文）"""
        df_cols_lower = [col.lower().strip() for col in df_cols]
        for col in target_cols:
            col_lower = col.lower().strip()
            if col_lower in df_cols_lower:
                return df_cols[df_cols_lower.index(col_lower)]
        return None

    def _parse_file_month(self, file_name):
        """从文件名解析年月：适配2020-01_news.xlsx格式，返回datetime(2020,1,1)"""
        try:
            date_str = os.path.basename(file_name).split("_")[0]
            return datetime.strptime(date_str, "%Y-%m")
        except (IndexError, ValueError):
            return None

    def _is_in_time_range(self, file_month, start_time, end_time):
        """判断文件年月是否在指定时间段内"""
        if not file_month:
            return False
        return start_time <= file_month <= end_time

    def load_documents(self, file_path: str, start_time: str, end_time: str):
        """
        加载指定时间段的海洋新闻xlsx文件，转换为带元数据的Document切片
        :param file_path: xlsx文件所在文件夹路径
        :param start_time: 起始时间段（格式：YYYY-MM，如2018-03）
        :param end_time: 结束时间段（格式：YYYY-MM，如2020-01）
        :return: 带全量元数据的Document切片列表
        """
        # 校验输入路径
        if not os.path.isdir(file_path):
            raise ValueError(f"文件路径不是有效文件夹：{file_path}")
        # 解析并校验时间段
        try:
            start_dt = datetime.strptime(start_time, "%Y-%m")
            end_dt = datetime.strptime(end_time, "%Y-%m")
        except ValueError:
            raise ValueError("时间段格式错误，请使用YYYY-MM（如2018-03）")
        if start_dt > end_dt:
            raise ValueError("起始时间不能晚于结束时间")

        documents = []
        # 获取文件夹下所有xlsx文件
        xlsx_files = glob.glob(os.path.join(file_path, "*.xlsx"))
        if not xlsx_files:
            raise FileNotFoundError(f"文件夹{file_path}下未找到xlsx文件")

        for xlsx_file in xlsx_files:
            # 解析文件年月并筛选
            file_month = self._parse_file_month(xlsx_file)
            if not self._is_in_time_range(file_month, start_dt, end_dt):
                print(f"跳过非目标时间段文件：{os.path.basename(xlsx_file)}")
                continue

            # 读取xlsx并过滤空行
            df = pd.read_excel(xlsx_file).dropna(how="all")
            if df.empty:
                print(f"文件{xlsx_file}无有效数据，跳过")
                continue

            # 容错获取核心列名（防止列名大小写/中英文不一致）
            title_col = self._get_col_name(df.columns, self.COLS_MAP["title"])
            time_col = self._get_col_name(df.columns, self.COLS_MAP["publish_date"])
            content_col = self._get_col_name(df.columns, self.COLS_MAP["content"])
            source_col = self._get_col_name(df.columns, self.COLS_MAP["source"])
            url_col = self._get_col_name(df.columns, self.COLS_MAP["url"])
            # 校验核心列（内容为必选，其他列可选）
            if not content_col:
                print(f"文件{xlsx_file}未找到「内容」列，跳过")
                continue

            # 逐行处理新闻数据
            for row_idx, row in df.iterrows():
                # 提取核心字段并过滤空内容
                news_content = str(row[content_col]).strip()
                if not news_content or news_content == "nan":
                    continue
                # 提取其他字段（空值填充为"未知"）
                news_title = str(row[title_col]).strip() if title_col else "unknown"
                news_title = news_title if news_title != "nan" else "unknown"
                news_pub_time = str(row[time_col]).strip() if time_col else "unknown"
                news_pub_time = news_pub_time if news_pub_time != "nan" else "unknown"
                news_source = str(row[source_col]).strip() if source_col else "unknown"
                news_source = news_source if news_source != "nan" else "unknown"
                news_url = str(row[url_col]).strip() if url_col else "unknown"
                news_url = news_url if news_url != "nan" else "unknown"

                # 生成唯一新闻ID：前缀_文件年月_行号（避免多文件重复）
                file_month_str = file_month.strftime("%Y%m") if file_month else "unknown"
                news_id = f"{self.prefix}{file_month_str}_{row_idx + 1:04d}"

                # 构造原始新闻Document（绑定全量元数据）
                full_news_doc = Document(
                    page_content=news_content,
                    metadata={
                        "news_id": news_id,          # 唯一新闻ID
                        "title": news_title,         # 新闻标题
                        "publish_date": news_pub_time,# 发布时间
                        "source": news_source,       # 新闻来源
                        "url": news_url            # 详情页URL
                    }
                )

                # 对单条新闻进行切片
                chunks = self.text_splitter.split_documents([full_news_doc])
                # 为每个切片绑定切片索引和总切片数
                for chunk_idx, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        "chunk_idx": chunk_idx + 1,  # 切片序号（从1开始）
                        "total_chunks": len(chunks)  # 该新闻的总切片数
                    })
                # 将切片加入总列表
                documents.extend(chunks)

        if not documents:
            raise ValueError(f"目标时间段[{start_time}至{end_time}]内未找到有效新闻数据")
        print(f"成功加载并切片：共{len(documents)}个新闻切片（来自目标时间段的xlsx文件）")
        return documents

    def load_documents_from_mysql(
            self,
            start_time: str,
            end_time: str,
            table_name: str = "marine_news"
    ) -> List[Document]:
        """
        从MySQL加载指定时间段的海洋新闻，转换为带元数据的Document切片
        :param start_time: 起始时间段（格式：YYYY-MM-DD，如2025-01-01）
        :param end_time: 结束时间段（格式：YYYY-MM-DD，如2025-01-31）
        :param table_name: 数据库表名
        :return: Document切片列表
        """
        db = MySQLDB(
            host="localhost",
            port=3306,
            user="root",
            password="123456",
            database="marine_news_db"
        )

        # 1. 校验时间格式与范围：升级为 年月日 格式校验
        try:
            # 修改解析格式为 YYYY-MM-DD，支持精确到日期
            start_dt = datetime.strptime(start_time, "%Y-%m-%d")
            end_dt = datetime.strptime(end_time, "%Y-%m-%d")
        except ValueError:
            raise ValueError("时间段格式错误，请使用 YYYY-MM-DD（如2025-01-01）")

        if start_dt > end_dt:
            raise ValueError("起始时间不能晚于结束时间")

        # 2. 构建SQL查询：修复语法错误 + 适配DATE类型直接查询（性能更优）
        # 核心优化：publish_date是DATE类型，无需格式化，直接范围查询
        sql = f"""
            SELECT * FROM {table_name} 
            WHERE `publish_date` BETWEEN %s AND %s
        """
        # 执行参数化查询，防止SQL注入
        news_list = db.query(sql, params=(start_time, end_time))

        # 校验查询结果
        if not news_list:
            raise ValueError(f"目标时间段[{start_time} 至 {end_time}]内未找到有效新闻数据")

        documents = []
        # 3. 遍历数据，构造文档并切片（未使用的索引改为_，消除代码警告）
        for _, news in enumerate(news_list):
            # 安全提取字段，增加空值兜底处理
            news_id = news["id"]
            # strip() + 空字符串兜底，避免None导致strip()报错
            news_content = str(news.get("content", "")).strip()
            news_title = str(news.get("title", "")).strip()
            # 数据库返回的datetime.date对象直接转字符串，格式为YYYY-MM-DD
            news_pub_time = str(news["publish_date"])
            news_source = str(news.get("source", "")).strip()
            news_url = str(news.get("url", "")).strip()

            # 跳过无正文内容的新闻，避免生成空切片
            if not news_content:
                continue

            # 构造完整新闻文档
            full_doc = Document(
                page_content=news_content,
                metadata={
                    "news_id": news_id,
                    "title": news_title,
                    "publish_date": news_pub_time,
                    "source": news_source,
                    "url": news_url
                }
            )

            # 文本切片并补充切片元数据
            chunks = self.text_splitter.split_documents([full_doc])
            for chunk_idx, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_idx": chunk_idx + 1,
                    "chunk_id": f"{news_id}_c{chunk_idx:03d}",
                    "total_chunks": len(chunks)
                })
            documents.extend(chunks)

        # 最终校验与返回
        print(f"成功加载并切片：共{len(documents)}个新闻切片")
        return documents


if __name__ == '__main__':
    print(datetime.now())
    dp = DocumentProcessor(RAGConfig())
    print(datetime.now())

    print(datetime.now())
    documents = dp.load_documents_from_mysql(
        start_time="2025-01-01",end_time="2025-01-25")
    # for doc in documents:
    #     print(doc)
    print(datetime.now())