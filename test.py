import faiss
import numpy as np

# 假设你的向量维度
d = 1024  

# IVF 参数
nlist = 32    # 越大越准，但训练越慢，index越大
# PQ 参数
m = 32           # 把1024维拆成32个子空间 → 每维 = 1024/32 = 32
nbits = 8        # 每个子向量 8bit（典型设置）

# 1. 构建 IVF+PQ index
quantizer = faiss.IndexFlatL2(d)  # IVF 用的 coarse quantizer
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)

print("Need training:", not index.is_trained)   # True

# ---------------------------------------
# 2. 生成训练数据（一定要独立的训练数据）
#    注意训练数据最好 10万条以上
# ---------------------------------------
train_num = 1600
train_data = np.random.random((train_num, d)).astype("float32")

print("Training...")
index.train(train_data)
print("Trained:", index.is_trained)

# ---------------------------------------
# 3. 添加数据库向量（你的检索库）
# ---------------------------------------
nb = 1000
xb = np.random.random((nb, d)).astype("float32")

index.add(xb)
print("ntotal:", index.ntotal)

# ---------------------------------------
# 4. 检索
# ---------------------------------------
xq = np.random.random((5, d)).astype("float32")  # 5 个 query

# 控制搜索精度：nprobe
index.nprobe = 16   # 常用值：8～64

k = 10
dist, ids = index.search(xq, k)
print("Result IDs:\n", ids)
