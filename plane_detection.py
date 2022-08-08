# @author Timur
# @author Yasu

from collections import deque
import math
import cv2
import numpy as np
import random
import pyrealsense2 as rs
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

# 面番号
NUMBERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Reject理由
MISSING_DATA = 0
BIG_DIFF = 1
BIG_MSE = 2

# 処理の概形\
# 入力は2次元になった
def fast_plane_extraction(verts):
    # データ構造構築(ステップ1)
    nodes, edges = init_graph(verts)

    # 粗い平面検出(ステップ2)
    boudaries, pai = ahcluster(nodes, edges)
    # # 粗い平面検出を精緻化(ステップ3)
    # cluster, pro_pai = refine(boudaries, pai)
    # return cluster, pro_pai
    return nodes, None

# グループのクラス
class Node:
    def __init__(self, data):
        self.data = data    # np.array型の深さの集合(10×10)
        self.g_num = 0 # グループの番号,初期値0
        self.reject_type = None #Rejectされた理由　Noneの場合はされていない
        self.center = [np.average(data[..., 0]), np.average(data[..., 1]), np.average(data[..., 2])]    # 平均(重心)
        self.cov = np.cov(data[..., :2].reshape(100, 2).T, data[..., 2].reshape(100,), rowvar=True)     # 分散共分散行列

        eig = np.linalg.eig(self.cov)   
        ind = np.argmin(eig[0])
        self.mse = eig[0][ind] * data.size    # 最小の固有値

        if eig[1][ind,0]*self.center[0] + eig[1][ind,1]*self.center[1] + eig[1][ind,2]*self.center[2] <= 0:
            self.normal = eig[1][ind]  #固有ベクトル(法線)
        else:
            self.normal = -eig[1][ind]
        
        self.t_mse = (1.6e-6*self.center[2]**2+5)**2

        # 上下左右のノード
        self.left = None
        self.right = None
        self.up = None
        self.down = None

# データ構造構築
def init_graph(verts, h=10, w=10):
    nodes = []
    rejected_nodes = []
    edges = []

    # 10×10が横にいくつあるかの数
    width = len(verts[0]) // w 
    height = len(verts) // h

    for i in range(height):
        nodes_line = []
        for j in range(width):
            # node は論文の v
            node = Node(verts[i*h:i*h+h,j*w:j*w+w])
            # nodeの除去の判定
            if reject_node(node):
                rejected_nodes.append(node)
            nodes_line.append(node)
        nodes.append(nodes_line)

    # 右・下の繋がりを調べる
    for i in range(len(nodes)):
        for j in range(len(nodes[0])):
            if nodes[i][j].reject_type == None:
                if j+1<len(nodes[0]) and not rejectedge(nodes[i][j], nodes[i][j+1]):
                    nodes[i][j].right = nodes[i][j+1]
                    nodes[i][j+1].left = nodes[i][j]
                    edges.append(nodes[i][j])
                if i+1<len(nodes) and not rejectedge(nodes[i][j], nodes[i+1][j]):
                    nodes[i][j].down = nodes[i+1][j]
                    nodes[i+1][j].up = nodes[i][j]
                    edges.append(nodes[i][j])

    edges = list(set(edges))
    
    print(f"rejectされていないノード数: {sum(len(v) for v in nodes) - len(rejected_nodes)}")
    print(f"エッジ数: {len(edges)}")
    # rand = random.randint(0, len(rejected_nodes))
    # print(f"rejectされたノードの例: {[node.data for node in rejected_nodes[rand:rand+1]]}")
    print("-------------------------------------------------------------------------------")

    return nodes, edges

# ノードの除去
def reject_node(node, threshold=0.1):
    data = node.data

    # データが欠落していたら
    if np.any(data[..., -1] == 0):
        node.reject_type = MISSING_DATA
        return True

    # 周囲4点との差
    v_diff = np.linalg.norm(data[:-1] - data[1:], axis=2)
    h_diff = np.linalg.norm(data[:, :-1] - data[:, 1:], axis=2)

    if v_diff.max() > threshold or h_diff.max() > threshold:
        node.reject_type = BIG_DIFF
        return True

    # print(f"mse = {node.mse}    t = {node.t_mse}")

    if node.mse > 0.01:   #ここではじかれるノードがない...
        node.reject_type = BIG_MSE
        return True

    return False

# 連結関係の除去
def rejectedge(node1, node2):
    # 欠落していなければ
    if node2.reject_type != None:
        return True 

    pfn = np.abs(np.dot(node1.normal, node2.normal))

    if  pfn <= 0.7:
        return True

    return False

# 平均二乗誤差の計算
def mse(array):
    pca = PCA()
    if array.size == 1:
        return math.inf
    pca.fit(array)
    # print(f"array: {array}")
    # print(f"fit: {pca.components_}")
    # print("-----------------------------------------------------------")
    return mean_squared_error(array, pca.fit_transform(array))  # 合ってるか不安

# 粗い平面検出
def ahcluster(nodes, edges):

    # MSEの昇順のヒープを作る
    queue = build_mse_heap(nodes)
    boudaries = np.array([])
    pai = np.array([])

    # # queueの中身がある限り
    # while queue != []:
    #     suf = popmin(queue)

    #     # 周りが同じ面ならばみない
    #     if (suf.g_num == suf.left.g_num) and (suf.g_num == suf.right.g_num) and (suf.g_num == suf.up.g_num) and (suf.down.g_num):
    #         continue

    #     # vがマージされているならば
    #     """
    #     if suf not in nodes:
    #         continue
    #     """

    #     # 面番号がなければ定義する
    #     if suf.g_num == 0:
    #         suf.g_num = NUMBERS.pop()

    #     # vと連結関係にあるuを取り出して
    #     # 連結関係のノードをマージする
    #     suf_test = np.hstack[suf.left.data, suf.data]
    #     suf_best = np.zeros(1)
    #     suf_choice = None
    #     # 一番MSEが小さいノードを選ぶ
    #     if mse(suf_test) < mse(suf_best):
    #             # 連結しているノードの中で一番優秀なものをbest,くっつけた状態のものをchoiceへ暫定的に
    #             if suf.left.g_num == suf.g_num:
    #                 continue
    #             else:
    #                 suf_best = suf_test
    #                 suf_choice = suf.left

    #     suf_test = np.hstack[suf.data, suf.right.data]

    #     if mse(suf_test) < mse(suf_best):
    #             # 連結しているノードの中で一番優秀なものをbest,くっつけた状態のものをchoiceへ暫定的に
    #             if suf.right.g_num == suf.g_num:
    #                 continue
    #             else:
    #                 suf_best = suf_test
    #                 suf_choice = suf.right

    #     suf_test = np.vstack[suf.up.data, suf.data]

    #     if mse(suf_test) < mse(suf_best):
    #             # 連結しているノードの中で一番優秀なものをbest,くっつけた状態のものをchoiceへ暫定的に
    #             if suf.up.g_num == suf.g_num:
    #                 continue
    #             else:
    #                 suf_best = suf_test
    #                 suf_choice = suf.up
        
    #     suf_test = np.vstack[suf.data, suf.down.data]

    #     if mse(suf_test) < mse(suf_best):
    #             # 連結しているノードの中で一番優秀なものをbest,くっつけた状態のものをchoiceへ暫定的に
    #             if suf.down.g_num == suf.g_num:
    #                 continue
    #             else:
    #                 suf_best = suf_test
    #                 suf_choice = suf.down

    #     # マージ失敗
    #     if mse(suf_best) >= 999999:
    #             #boudaries = boudaries.append(suf)
    #             pai = pai.append(plane(suf_best))

    #     # マージ成功
    #     else:
    #             queue.append(suf)
    #             suf_choice.g_num = suf.g_num
    # return pai
    return None, None

# 平面近似PCA
def plane(v):
    pca = PCA()
    return pca.fit(v)

# ヒープの作成
def build_mse_heap(nodes):
    # 1つずつmseを計算し直して並べ替える
    queue = sorted(sum(nodes, []), key=lambda node: node.mse)
    print(queue[len(queue)//2].mse)
    print(queue[10].mse)
    print(queue[-1].mse)
    return queue

# 先頭を取り出す
def popmin(queue):
    choice = queue.popleft()
    return choice

# 粗い平面検出の精緻化
# Bk,Rk,Rlは任意の順番のという意味かもしれない
def refine(boundaries, pai):
    # キュー
    queue = deque()
    refine = np.array()
    rf_nodes = np.array()
    rf_edges = np.array()
    for k, boundary in enumerate(boundaries):
        refine_k = np.array()
        # 謎ポイント
        refine = refine.append(refine_k)
        # フチを除く
        for v in boundary:
            # 上下左右のノードが面内ではないならば
            if v not in boundary:
                # 境界点判定
                boundary = boundary.remove(v)
        # 除去した境界のポイントを個別でみる
        for p in v:
            # kとは何の値？→インデックス？pとのタプルで追加しろといっている？
            queue.append({p, k})
        if boundary != None:
            rf_nodes = rf_nodes.appned(boundary)
    while queue != None:
        points = queue.popleft()
        k = points[1]
        for p in points[0]:
            if (p in boundary) or (p in refine_k) or (math.dist(p, pai[k]) >= 9 * mse(boundary[k])):
                continue
            # lが存在するならば？l=k+1？
            if (k+1 <= len(boundaries)) or (p in refine):
                rf_edges = rf_edges.append({boundary_k,boundary_l})
                if math.dist(p, pai[k]) < math.dist(p, pai[k+1]):
                    refine_l = refine_l.remove(p)
                    refine_k = refine_k.append(p)
            else:
                refine_k = refine_k.append(p)
                queue.append({p, k})
                # enqueue
    for refine_k in refine:
        boundary_k = boundary_k.append(refine_k)
    cluster, pro_pai = ahcluster(rf_nodes, rf_edges)
    return cluster, pro_pai

def visualization(color_image, nodes):
    color_image = np.copy(color_image)
    BLACK = [0, 0, 0]
    RED = [0, 0, 255]
    ORANGE = [17, 144, 255]
    PURPLE = [255, 0, 127]
    PINK = [102, 0, 255]
    GRAY = [128, 128, 128]
    GREEN = [51, 255, 51]
    BLUE = [255, 90, 0]

    color_image[::10] = BLACK
    color_image[:, ::10] = BLACK    
    
    for h in range(0, len(color_image), 10):
        for w in range(0, len(color_image[0]), 10):
            if nodes[h//10][w//10].reject_type == MISSING_DATA:
                color_image[h+4:h+7, w+4:w+7] = BLACK
            elif nodes[h//10][w//10].reject_type == BIG_DIFF:
                color_image[h+4:h+7, w+4:w+7] = ORANGE
            elif nodes[h//10][w//10].reject_type == BIG_MSE:
                color_image[h+4:h+7, w+4:w+7] = RED

            if nodes[h//10][w//10].up:
                color_image[h-5:h+5, w+5] = PURPLE
            if nodes[h//10][w//10].down:
                color_image[h+5:h+15, w+5] = PURPLE
            if nodes[h//10][w//10].left:
                color_image[h+5, w-5:w+5] = PURPLE
            if nodes[h//10][w//10].right:
                color_image[h+5, w+5:w+15] = PURPLE

    return color_image

if __name__ == '__main__':
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Convert images to numpy arrays
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    
    pc = rs.pointcloud()

    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data()) #計算に使用
        color_image = np.asanyarray(color_frame.get_data())

        points = pc.calculate(depth_frame)
        v = points.get_vertices()
        verts = np.asanyarray(v).view(np.float32).reshape(480, 640, 3)  # xyz

        cluster, pro_pai = fast_plane_extraction(verts)

        color_image = visualization(color_image, cluster)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_scale = cv2.convertScaleAbs(depth_image, alpha=0.03)  #0～255の値にスケール変更している

        depth_colormap = cv2.applyColorMap(depth_scale, cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        key = cv2.waitKey(1)
        prop_val = cv2.getWindowProperty('RealSense', cv2.WND_PROP_ASPECT_RATIO)

        if key == ord("s"):
            cv2.imwrite('./out.png', images)
        if key == ord("q") or (prop_val < 0):
            # Stop streaming
            pipeline.stop()
            cv2.destroyAllWindows()
            break
