import pymysql
from pymysql import Error
from typing import List, Dict, Optional, Tuple


class MySQLDB:
    """
    适配低频次访问的MySQL操作类（半分钟一次）
    每次操作都新建连接，执行完立即关闭，逻辑简单、稳定
    """

    def __init__(self, host: str = 'localhost', port: int = 3306,
                 user: str = 'root', password: str = '', database: str = '',
                 charset: str = 'utf8mb4'):
        """初始化数据库配置（仅存参数，不提前创建连接）"""
        self.config = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
            "charset": charset
        }

    def _get_connection(self) -> Optional[pymysql.connections.Connection]:
        """临时创建连接（每次操作都新建）"""
        try:
            conn = pymysql.connect(**self.config)
            return conn
        except Error as e:
            print(f"❌ 创建连接失败: {e}")
            return None

    def query(self, sql: str, params: Optional[Tuple] = None) -> List[Dict]:
        """
        查询数据（每次查询都新建连接，执行完关闭）
        :param sql: 查询SQL（%s占位符）
        :param params: SQL参数（元组）
        :return: 字典格式的查询结果
        """
        conn = self._get_connection()
        if not conn:
            return []

        cursor = None
        try:
            # 创建字典游标（结果按字段名取值）
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            cursor.execute(sql, params or ())
            results = cursor.fetchall()
            print(f"✅ 查询成功，返回 {len(results)} 条数据")
            return results
        except Error as e:
            print(f"❌ 查询失败: {e} | SQL: {sql} | 参数: {params}")
            return []
        finally:
            # 无论成功失败，都关闭游标和连接
            if cursor:
                cursor.close()
            if conn and conn.open:
                conn.close()

    def execute(self, sql: str, params: Optional[Tuple] = None) -> int:
        """
        执行插入/更新/删除（每次操作都新建连接，自动提交事务）
        :return: 受影响行数（失败返回-1）
        """
        conn = self._get_connection()
        if not conn:
            return -1

        cursor = None
        try:
            cursor = conn.cursor()
            affected_rows = cursor.execute(sql, params or ())
            conn.commit()  # 执行写操作必须提交
            print(f"✅ 执行成功，受影响行数: {affected_rows}")
            return affected_rows
        except Error as e:
            conn.rollback()  # 失败回滚
            print(f"❌ 执行失败: {e} | SQL: {sql} | 参数: {params}")
            return -1
        finally:
            if cursor:
                cursor.close()
            if conn and conn.open:
                conn.close()

    def insert_one(self, table: str, data: Dict) -> Optional[int]:
        """简化插入单条数据（无需手写SQL）"""
        if not data:
            print("❌ 插入数据为空")
            return None

        fields = ", ".join(data.keys())
        placeholders = ", ".join(["%s"] * len(data))
        sql = f"INSERT INTO {table} ({fields}) VALUES ({placeholders})"
        params = tuple(data.values())

        # 调用execute执行插入
        if self.execute(sql, params) > 0:
            # 重新查询获取自增ID（因为连接已关闭，需二次查询）
            last_id_sql = f"SELECT LAST_INSERT_ID() as id FROM {table} LIMIT 1"
            result = self.query(last_id_sql)
            return result[0]['id'] if result else None
        return None


# ------------------- 使用示例 -------------------
if __name__ == "__main__":
    # 初始化（仅配置参数，不创建连接）
    db = MySQLDB(
        host="localhost",
        port=3306,
        user="root",
        password="123456",
        database="marine_news_db"
    )


    results = db.query("select * from marine_news where publish_date = '2025-01-01'")
    print("30秒后查询结果：", results)
    for res in results:
        print(res)