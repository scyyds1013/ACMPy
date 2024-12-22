from resemblyzer import VoiceEncoder, preprocess_wav, \
    normalize_volume, trim_long_silences  # 注意：这里的 preprocess_wav 可能不是 resemblyzer 库的一部分，需要确认
from pathlib import Path  # 用于处理文件路径
import numpy as np
import librosa  # 确保安装了 librosa 库

# 定义全局变量（或者将它们作为参数传递给 preprocess_wav）
sampling_rate = 16000  # 假设这是模型要求的采样率
audio_norm_target_dBFS = -3.0  # 假设的归一化目标

# 加载预训练的语音编码器（确保 VoiceEncoder 是从 resemblyzer 或其他正确库中导入的）
encoder = VoiceEncoder()


# 修改 preprocess_wav 函数（如果它不在 resemblyzer 库中，则需要您自己实现）
# 或者，如果它已经在库中并且行为不同，请相应地调整下面的代码
def my_preprocess_wav(fpath_or_wav, target_sr=sampling_rate):
    if isinstance(fpath_or_wav, (str, Path)):
        wav, sr = librosa.load(str(fpath_or_wav), sr=None)
        if sr != target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    else:
        wav = fpath_or_wav
        # 如果传递的是波形数组，这里假设它已经是以目标采样率采样的
        # 如果不是，则需要在这里添加重采样的代码（但通常这不是必要的，因为
        # 调用者应该负责传递正确采样率的波形）

    # 注意：这里缺少了 normalize_volume 和 trim_long_silences 的实现
    # 这些函数需要您自己实现，或者从 resemblyzer 或其他库中获取（如果它们存在）
    # 假设它们已经被正确实现并导入
    wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    wav = trim_long_silences(wav)

    return wav


# # 预处理第一个音频文件并获取嵌入
# wav1 = my_preprocess_wav("uploads/")
# embedding1 = encoder.embed_utterance(wav1)  # 不需要传递采样率
#
# # 预处理第二个音频文件并获取嵌入
# wav2 = my_preprocess_wav("")
# embedding2 = encoder.embed_utterance(wav2)  # 不需要传递采样率
#
#
# # 计算余弦相似度
# cosine_similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
#
# print(f"两段语音的余弦相似度为: {cosine_similarity:.4f}")
