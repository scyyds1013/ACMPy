import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


def convert_text_to_bert_format(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    return input_ids, attention_mask


def calculate_similarity_text(text1, text2):
    input_ids1, attention_mask1 = convert_text_to_bert_format([text1])
    input_ids2, attention_mask2 = convert_text_to_bert_format([text2])

    # 获取BERT模型的输出
    outputs1 = model(input_ids1, attention_mask=attention_mask1)
    outputs2 = model(input_ids2, attention_mask=attention_mask2)

    # 从最后一层提取[CLS]标记的嵌入
    cls_embedding1 = outputs1.last_hidden_state[:, 0, :]  # [batch_size, sequence_length, hidden_size]
    cls_embedding2 = outputs2.last_hidden_state[:, 0, :]

    # 将嵌入转换为NumPy数组并计算余弦相似度
    cls_embedding1_np = cls_embedding1.detach().cpu().numpy()
    cls_embedding2_np = cls_embedding2.detach().cpu().numpy()

    # similarity = cosine_similarity([cls_embedding1_np], [cls_embedding2_np])[0][0]
    # 保持二维数组形状，但只包含一个样本
    similarity = cosine_similarity(cls_embedding1_np[:1], cls_embedding2_np[:1])[0][0]
    return similarity

#
# text1 = "你好"
# text2 = "你好吗"
#
#
# similarity_score = calculate_similarity_text(text1, text2)
# print(f"The cosine similarity between the two texts is: {similarity_score:.4f}")