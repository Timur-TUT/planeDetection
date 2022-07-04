import numpy as np
import math
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

data1 = np.array([[5417, 5465, 5514, 5514, 5514, 5564, 5514, 5465, 5465, 5417],
       [5417, 5465, 5514, 5514, 5514, 5514, 5514, 5514, 5417, 5417],
       [5465, 5514, 5514, 5514, 5514, 5514, 5514, 5514, 5465, 5417],
       [5465, 5514, 5564, 5564, 5564, 5564, 5514, 5514, 5514, 5417],
       [5465, 5514, 5564, 5564, 5564, 5564, 5564, 5564, 5514, 5465],
       [5465, 5514, 5514, 5564, 5564, 5564, 5564, 5564, 5514, 5465],
       [5417, 5465, 5514, 5514, 5564, 5564, 5564, 5564, 5514, 5465],
       [5465, 5514, 5514, 5514, 5514, 5514, 5514, 5514, 5465, 5417],
       [5465, 5514, 5514, 5514, 5514, 5514, 5465, 5465, 5417, 5370],
       [5514, 5514, 5514, 5514, 5465, 5417, 5417, 5417, 5370, 5370]])

# 平均二乗誤差の計算
def mse(array):
    if np.any(array == 0):
        return math.inf
    else:
        pca = PCA()
        mse = mean_squared_error(array, pca.fit_transform(array))
        print(f"mse: {mse}")
        return mse  #合ってるか不安

# ノードの除去
def reject_node(data, threshold=999):
    # データが欠落していたら    
    if np.any(data == 0):
        print("has 0")
        return True

    # 周囲4点との差
    v_diff = np.abs(np.diff(data, axis=0))  #縦方向の差
    h_diff = np.abs(np.diff(data, axis=1))          #横方向の差

    print(v_diff)
    print(h_diff)

    if v_diff.max() > threshold or h_diff.max() > threshold:
        print("too big diff")
        return True

    # まだ決めていない
    if mse(data) > 999999:
        print("too big mse")
        return True

    return False

if reject_node(data1):
    print("rejected")
else:
    print("not rejected")