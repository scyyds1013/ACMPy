import json
from datetime import datetime

from flask import Flask, request, jsonify
import os
from flask import Flask
from videohash import VideoHash
from werkzeug.utils import secure_filename
import redis

from VGG16 import extract_features, calculate_similarity, VGGmodel
from Transformer import convert_text_to_bert_format, calculate_similarity_text, model, tokenizer
from Resemblyzer import my_preprocess_wav, encoder
import numpy as np

app = Flask(__name__)

# 配置Redis连接
redis_host = '127.0.0.1'  # Redis服务器地址，默认为localhost
redis_port = 6379  # Redis服务器端口，默认为6379
redis_db = 3  # 使用Redis的哪个数据库，默认为0

# 创建Redis连接对象
redis_client = redis.StrictRedis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)

# 设置上传文件的保存目录（确保这个目录存在并且应用有写入权限）
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 允许上传的文件扩展名（可以根据需要修改）
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'wav', 'mp4'}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 格式化时间戳为“年月日时分秒”
def format_timestamp(timestamp):
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')


def set_redis_value(key, value, tips):
    # 获取当前时间戳并格式化
    current_time = datetime.now()
    formatted_timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')

    # 创建一个包含所有相关信息的字典
    data1 = {
        'name': key,
        'img_path': value,
        'timestamp': formatted_timestamp
    }

    data2 = {
        'name': key,
        'text': value,
        'timestamp': formatted_timestamp
    }

    data3 = {
        'name': key,
        'wav': value,
        'timestamp': formatted_timestamp
    }

    data4 = {
        'name': key,
        'video': value,
        'timestamp': formatted_timestamp
    }

    # 将字典序列化为 JSON 字符串
    if tips == 1:
        data_json = json.dumps(data1)
        list_name = 'entries'
        redis_client.rpush(list_name, data_json)
    if tips == 2:
        data_json = json.dumps(data2)
        list_name = 'textList'
        redis_client.rpush(list_name, data_json)
    if tips == 3:
        data_json = json.dumps(data3)
        list_name = 'wavList'
        redis_client.rpush(list_name, data_json)
    if tips == 4:
        data_json = json.dumps(data4)
        list_name = 'videoList'
        redis_client.rpush(list_name, data_json)

    # 返回 JSON 响应
    return jsonify({'message': 'Data added to list "entries" successfully'})


def get_img_paths():
    list_name = 'entries'
    # 从 Redis 列表中获取所有元素
    entries = redis_client.lrange(list_name, 0, -1)
    other_img_paths = []
    # 解析每个 JSON 字符串并提取 img_path
    for entry in entries:
        data = json.loads(entry)
        other_img_paths.append(data['img_path'])
    # 返回所有 img_path 的列表（这里以 JSON 响应的形式返回，但您也可以根据需要以其他方式处理）
    return other_img_paths


def get_text():
    list_name = 'textList'
    # 从 Redis 列表中获取所有元素
    entries = redis_client.lrange(list_name, 0, -1)
    other_img_paths = []
    # 解析每个 JSON 字符串并提取 img_path
    for entry in entries:
        data = json.loads(entry)
        other_img_paths.append(data['text'])
    # 返回所有 img_path 的列表（这里以 JSON 响应的形式返回，但您也可以根据需要以其他方式处理）
    return other_img_paths


def get_wavs():
    list_name = 'wavList'

    entries = redis_client.lrange(list_name, 0, -1)
    other_img_paths = []

    for entry in entries:
        data = json.loads(entry)
        other_img_paths.append(data['wav'])
    return other_img_paths


def get_video():
    list_name = 'videoList'

    entries = redis_client.lrange(list_name, 0, -1)
    other_img_paths = []

    for entry in entries:
        data = json.loads(entry)
        other_img_paths.append(data['video'])
    return other_img_paths


@app.route('/1', methods=['POST'])
def create_channel():
    # 检查请求中是否包含文件
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']
    name = request.form['name']
    # 如果用户没有选择文件，浏览器会提交一个空的文件名而没有文件内容
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        # 使用 secure_filename 来获取安全的文件名
        filename = secure_filename(file.filename)

        other_img_paths = get_img_paths()

        if filename in [os.path.basename(path) for path in other_img_paths]:
            return jsonify({'error': 'File already exists'}), 400

        img_path1 = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # 先保存至本地 后续逻辑判断后决定是否删除
        file.save(img_path1)

        # 提取特征和计算相似度的代码（与 feature_extraction.py 中的相同）
        features1 = extract_features(img_path1, VGGmodel)
        highest_similarity = -1
        highest_similarity_path = ''

        for img_path2 in other_img_paths:
            features2 = extract_features(img_path2, VGGmodel)
            similarity = calculate_similarity(features1, features2)
            if similarity > highest_similarity:
                highest_similarity = similarity
                highest_similarity_path = img_path2

        print(f"Highest Similarity Path: {highest_similarity_path}")
        print(f"Highest Similarity: {highest_similarity}")
        if highest_similarity >= 0.8:
            os.remove(img_path1)
            return jsonify({'error': 'File type not allowed'}), 400
        else:
            set_redis_value(name, img_path1, 1)
            return jsonify({'message': 'File uploaded successfully', 'filename': filename}), 201


@app.route('/2', methods=['POST'])
def textCheck():
    try:
        textList = get_text()
        data = request.get_json()  # 获取 JSON 数据
        text1 = data['content']
        name = data['username']

        highest_similarity = -1
        for text2 in textList:
            similarity = calculate_similarity_text(text1, text2)
            if similarity > highest_similarity:
                highest_similarity = similarity

        print(f"Highest Similarity: {highest_similarity}")

        if highest_similarity >= 0.95:
            return jsonify({'error': 'File type not allowed'}), 400
        else:
            set_redis_value(name, text1, 2)
        return jsonify(
            {"message": f"The cosine similarity between the two texts is: {highest_similarity:.4f}"}), 200  # 修改响应格式和状态码
    except Exception as e:
        return jsonify({"error": str(e)}), 400  # 返回错误信息和 400 状态码


sampling_rate = 16000  # 假设这是模型要求的采样率
audio_norm_target_dBFS = -3.0  # 假设的归一化目标


@app.route('/3', methods=['POST'])
def musicCheck():
    # 检查请求中是否包含文件部分
    if 'file' not in request.files:
        return jsonify({"message": "No file part in the request"}), 400

    file = request.files['file']
    name = request.form['name']

    # 如果用户没有选择文件，浏览器也会提交一个空的文件名，没有文件内容
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    other_wavs = get_wavs()

    if file.filename in [os.path.basename(path) for path in other_wavs]:
        return jsonify({'error': 'File already exists'}), 400
    # 如果文件是允许的类型，保存它
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # 预处理第一个音频文件并获取嵌入
        wav1 = my_preprocess_wav(filepath)
        embedding1 = encoder.embed_utterance(wav1)  # 不需要传递采样率

        highest_similarity = -1
        highest_similarity_path = ''

        for wav2_path in other_wavs:
            # 预处理第二个音频文件并获取嵌入
            wav2 = my_preprocess_wav(wav2_path)
            embedding2 = encoder.embed_utterance(wav2)  # 不需要传递采样率
            # 计算余弦相似度
            cosine_similarity = np.dot(embedding1, embedding2) / (
                    np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            if cosine_similarity > highest_similarity:
                highest_similarity = cosine_similarity
                highest_similarity_path = wav2_path

        print(f"Highest Similarity Path: {highest_similarity_path}")
        print(f"Highest Similarity: {highest_similarity}")

        if highest_similarity >= 0.8:
            os.remove(filepath)
            return jsonify({'error': 'File type not allowed'}), 400
        else:
            set_redis_value(name, filepath, 3)
            return jsonify(f"两段语音的余弦相似度为: {highest_similarity:.4f}"), 200
    else:
        return jsonify({"message": "Allowed file types are wav"}), 400


@app.route('/4', methods=['POST'])
def videoCheck():
    file = request.files['file']
    name = request.form['name']

    other_video = get_video()

    if file.filename in [os.path.basename(path) for path in other_video]:
        return jsonify({'error': 'File already exists'}), 400

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # 创建VideoHash实例
        vh1 = VideoHash(filepath)
        # # 生成哈希值
        hash_value1 = vh1.hash

        highest_similarity = 100
        highest_similarity_path = ''

        for video2_path in other_video:
            vh2 = VideoHash(video2_path)
            hash_value2 = vh2.hash
            different_bits_count = sum(a != b for a, b in zip(hash_value1, hash_value2))
            if different_bits_count < highest_similarity:
                highest_similarity = different_bits_count
                highest_similarity_path = video2_path

        print(f"Highest Similarity Path: {highest_similarity_path}")
        print(f"Highest Similarity: {highest_similarity}")

        if highest_similarity <= 10:
            os.remove(filepath)
            return jsonify({'error': 'File type not allowed'}), 400
        else:
            set_redis_value(name, filepath, 4)
            return jsonify(f"两段音频的不同哈希位为: {highest_similarity}"), 200
    else:
        return jsonify({"message": "Allowed file types are wav"}), 400


if __name__ == '__main__':
    app.run()
