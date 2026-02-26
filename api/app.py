# app.py
from flask import Flask
from flask_cors import CORS
from api.chat_routes import chat_bp


def create_app():
    app = Flask(__name__)

    # 允许跨域请求，Vue 前端（通常在 5173 或 8080 端口）才能正常访问后端
    # 在生产环境中，建议将 origins 限制为你实际的前端域名
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # 注册路由蓝图
    app.register_blueprint(chat_bp, url_prefix='/api')

    @app.route('/health', methods=['GET'])
    def health_check():
        return {"status": "ok", "message": "海洋新闻分析系统后端正常运行中"}

    return app


if __name__ == '__main__':
    app = create_app()
    # 开启 debug 模式方便开发调试，默认运行在 5000 端口
    app.run(host='0.0.0.0', port=5000, debug=True)