import os
import pandas as pd
import pymysql
from pymysql import Error
from typing import List, Dict
from datetime import datetime


class MarineNewsImporter:
    """海洋新闻批量导入MySQL工具类"""

    def __init__(self, db_host: str, db_port: int, db_user: str, db_pwd: str, db_name: str):
        """初始化数据库连接配置"""
        self.db_config = {
            "host": db_host,
            "port": db_port,
            "user": db_user,
            "password": db_pwd,
            "database": db_name,
            "charset": "utf8mb4"
        }
        self.table_name = "marine_news"  # 数据库表名
        # 创建表（首次运行时执行）
        self._create_table()

    def _create_table(self):
        """创建海洋新闻表（不存在则创建）"""
        conn = None
        cursor = None
        try:
            conn = pymysql.connect(**self.db_config)
            cursor = conn.cursor()
            # 建表SQL（包含唯一ID、英文字段、链接唯一约束）
            create_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id INT AUTO_INCREMENT PRIMARY KEY COMMENT 'NewsID',
                title VARCHAR(500) NOT NULL COMMENT 'Title',
                publish_date DATE COMMENT 'publish_date',
                content TEXT COMMENT 'content',
                source VARCHAR(100) COMMENT 'source',
                url VARCHAR(500) UNIQUE COMMENT 'url',
                create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'create_time'
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='news_table';
            """
            cursor.execute(create_sql)
            conn.commit()
            print(f"✅ 成功创建/验证表 {self.table_name}")
        except Error as e:
            print(f"❌ 创建表失败: {e}")
            if conn:
                conn.rollback()
        finally:
            self._close_conn(conn, cursor)

    def _read_xlsx_files(self, folder_path: str) -> List[Dict]:
        """读取指定文件夹下所有XLSX文件，返回整合后的新闻数据列表"""
        all_news = []
        # 遍历文件夹下所有.xlsx文件
        for filename in os.listdir(folder_path):
            if filename.endswith(".xlsx"):
                file_path = os.path.join(folder_path, filename)
                try:
                    # 读取xlsx文件（假设数据在第一个sheet，无表头，按列顺序对应）
                    # 列顺序：title, publish_date, content, source, url
                    df = pd.read_excel(
                        file_path,
                        header=0,  # 你的xlsx无表头，设为None ["标题", "发布时间", "内容", "来源", "详情页URL"]
                        names=["title", "publish_date", "content", "source", "url"],  # 映射为英文字段 标题	发布时间	内容	来源	详情页URL
                        dtype=str,  # 统一按字符串读取，避免格式问题
                        engine="openpyxl"
                    )
                    # 数据清洗：去除空值、处理日期格式
                    # df = df.dropna(subset=["标题", "详情页URL"])  # 标题/链接为空的行丢弃
                    # 处理日期格式（转为YYYY-MM-DD）
                    df["publish_date"] = pd.to_datetime(
                        df["publish_date"],
                        errors="coerce"  # 解析失败设为NaT
                    ).dt.date
                    # 转为字典列表
                    news_list = df.to_dict("records")
                    all_news.extend(news_list)
                    print(f"✅ 读取文件 {filename} 成功，共 {len(news_list)} 条数据")
                except Exception as e:
                    print(f"❌ 读取文件 {filename} 失败: {e}")
        print(f"\n📊 总计读取 {len(all_news)} 条新闻数据")
        return all_news

    def _batch_insert(self, news_list: List[Dict], batch_size: int = 100):
        """批量插入数据到MySQL，按batch_size分批插入（避免单次插入过多）"""
        if not news_list:
            print("⚠️ 无数据可插入")
            return

        conn = None
        cursor = None
        total_inserted = 0
        total_skipped = 0  # 重复数据跳过数

        # 插入SQL（参数化查询，防注入）
        insert_sql = f"""
        INSERT IGNORE INTO {self.table_name} 
        (title, publish_date, content, source, url)
        VALUES (%s, %s, %s, %s, %s)
        """
        # IGNORE：遇到url重复时跳过，不报错

        try:
            conn = pymysql.connect(**self.db_config)
            cursor = conn.cursor()

            # 分批插入
            for i in range(0, len(news_list), batch_size):
                batch_data = news_list[i:i + batch_size]
                # 提取批量插入的参数（按SQL字段顺序）
                batch_params = [
                    (
                        news.get("title", ""),
                        news.get("publish_date"),
                        news.get("content", ""),
                        news.get("source", ""),
                        news.get("url", "")
                    ) for news in batch_data
                ]
                # 执行批量插入
                affected_rows = cursor.executemany(insert_sql, batch_params)
                conn.commit()
                total_inserted += affected_rows
                total_skipped += len(batch_params) - affected_rows
                print(
                    f"📦 插入第 {i // batch_size + 1} 批数据，成功 {affected_rows} 条，跳过 {len(batch_params) - affected_rows} 条（重复）")

        except Error as e:
            print(f"❌ 批量插入失败: {e}")
            if conn:
                conn.rollback()
        finally:
            self._close_conn(conn, cursor)

        print(f"\n🎉 批量插入完成：成功插入 {total_inserted} 条，跳过重复 {total_skipped} 条")

    def _close_conn(self, conn, cursor):
        """关闭游标和连接"""
        if cursor:
            cursor.close()
        if conn and conn.open:
            conn.close()

    def run_import(self, folder_path: str, batch_size: int = 100):
        """执行完整导入流程：读取文件 → 批量插入"""
        # 1. 读取所有XLSX文件
        news_data = self._read_xlsx_files(folder_path)
        # 2. 批量插入数据库
        self._batch_insert(news_data, batch_size)


# ------------------- 执行导入 -------------------
if __name__ == "__main__":
    # 1. 配置信息（替换为你的实际配置）
    DB_HOST = "localhost"
    DB_PORT = 3306
    DB_USER = "root"
    DB_PASSWORD = "123456"
    DB_NAME = "marine_news_db"  # 你的数据库名
    XLSX_FOLDER = r"D:\ProgramFiles\Project\Python\MNVA\agent\rag\data\demo_data"  # 存放所有XLSX文件的文件夹路径（绝对/相对路径）

    # 2. 初始化导入工具
    importer = MarineNewsImporter(DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME)

    # 3. 执行批量导入（批量大小设为100，可根据数据库性能调整）
    importer.run_import(XLSX_FOLDER, batch_size=100)