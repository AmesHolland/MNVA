# api/data_routes.py
import os
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from agent.tools.data_manager import process_uploaded_file, load_registry

# 创建数据模块的Blueprint实例（仿照chat_bp的命名和格式）
data_bp = Blueprint('data', __name__)

# 配置上传目录（保持原有逻辑，适配Blueprint场景）
UPLOAD_FOLDER = './temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@data_bp.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    """接收前端上传的数据集文件，完成清洗和入库"""
    if 'file' not in request.files:
        return jsonify({"error": "没有找到文件对象"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "未选择文件"}), 400

    dataset_name = request.form.get('dataset_name', None)

    # 1. 保存到临时目录
    filename = secure_filename(file.filename)
    temp_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(temp_path)

    # 2. 调用核心处理模块进行清洗和入库
    result = process_uploaded_file(temp_path, file.filename, dataset_name)

    if "error" in result:
        return jsonify(result), 500
    return jsonify(result), 200


@data_bp.route('/list_datasets', methods=['GET'])
def list_datasets():
    """提供数据源列表接口，供前端下拉框渲染使用"""
    try:
        datasets = load_registry()
        # 按照上传时间倒序排列，最新的在最上面
        datasets.sort(key=lambda x: x['upload_time'], reverse=True)
        return jsonify({"datasets": datasets}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500