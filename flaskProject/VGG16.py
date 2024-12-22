from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np

# 加载预训练的VGG16模型，不包括最后的全连接层
base_model = VGG16(weights='imagenet', include_top=False)

# 定义新的模型，将原VGG16模型的输出直接作为新模型的输出
VGGmodel = Model(inputs=base_model.input, outputs=base_model.output)


def extract_features(img_path, model):
    """从指定路径的图像中提取特征"""
    # 加载图像，调整大小为224x224，VGG16模型要求的输入大小
    img = image.load_img(img_path, target_size=(224, 224))

    # 将PIL图像转换为numpy数组，并添加一个维度表示批大小
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # 预处理图像
    img_array = preprocess_input(img_array)

    # 通过模型获取图像的特征
    features = model.predict(img_array)

    # 扁平化特征使其成为一维数组
    flatten_features = features.flatten()

    # 归一化特征向量以比较它们的相似性
    normalized_features = flatten_features / np.linalg.norm(flatten_features)

    return normalized_features


def calculate_similarity(features1, features2):
    """计算两组特征之间的相似度"""
    similarity = np.dot(features1, features2)
    return similarity


# # 图片路径
# img_path1 = 'uploads/1733990356929.png'
# # 存储其他图片路径的列表
# other_img_paths = ['uploads/222.jpg']
#
# # 初始化最高相似度和对应的图片路径
# highest_similarity = -1
# highest_similarity_path = ''
#
# # 提取特征
# features1 = extract_features(img_path1, VGGmodel)
# # features2 = extract_features(img_path2, model)
#
# # 遍历列表中的每个图片路径
# for img_path2 in other_img_paths:
#     # 提取当前图片的特征
#     features2 = extract_features(img_path2, VGGmodel)
#
#     # 计算相似度
#     similarity = calculate_similarity(features1, features2)
#
#     # 检查当前相似度是否高于之前的最高相似度
#     if similarity > highest_similarity:
#         highest_similarity = similarity
#         highest_similarity_path = img_path2
#
# # 打印最高相似度的图片路径和相似度
# print(f"Highest Similarity Path: {highest_similarity_path}")
# print(f"Highest Similarity: {highest_similarity}")
