import cupy as cp
import numpy as np
import time

# 高维矩阵/数组：
gpu = cp.ones( (1024,512,4,4) )
cpu = np.ones( (1024,512,4,4) )

# 纯numpy的cpu测试：
ctime1 = time.time()
for c in range(1024):
    cpu = np.add(cpu,cpu)   # 这里用np.add()和直接用 + 一样！内核都是cpu来算
ctime2 = time.time()
ctotal = ctime2 - ctime1
print('纯cpu计算时间：', ctotal)

# 纯cupy的gpu测试：
gtime1 = time.time()
for g in range(1024):
    gpu = cp.add(gpu,gpu)   # 自带的加法函数
gtime2 = time.time()
gtotal = gtime2 - gtime1
print('纯gpu计算时间：', gtotal)

# gpu和cpu混合编程：
ggtime1 = time.time()
for g in range(1024):
    gpu = gpu + gpu         # 手工加法：+ 默认回到cpu计算！！！
ggtime2 = time.time()
ggtotal = ggtime2 - ggtime1
print('混合的计算时间：', ggtotal)