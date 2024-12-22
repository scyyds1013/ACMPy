from videohash import VideoHash
import videohash
# 指定两个视频文件的路径
video_path1 = '89d60284d6fba1e38e727d9269b73704.mp4'
video_path2 = 'c2963866d6f6151e1cead7b2db2ddc77.mp4'

# 创建VideoHash实例
vh1 = VideoHash(video_path1)
vh2 = VideoHash(video_path2)

# # 生成哈希值
hash_value1 = vh1.hash
hash_value2 = vh2.hash

# 打印哈希值
print(f"Hash value for video 1: {hash_value1}")
print(f"Hash value for video 2: {hash_value2}")

# 比较哈希值是否相同
if hash_value1 == hash_value2:
    print("The two videos are identical.")
else:
    # 比较哈希值的相似度（这里简单用不同位数的数量来表示）
    different_bits_count = sum(a != b for a, b in zip(hash_value1, hash_value2))
    print(f"The two videos are not identical, with {different_bits_count} different bits in their hash values.")

    # 根据实际需求设定一个相似度阈值
    similarity_threshold = 10  # 假设我们认为不同位数少于10的视频是近似重复的
    if different_bits_count <= similarity_threshold:
        print("The two videos are similar (approximately duplicate).")
    else:
        print("The two videos are not similar.")