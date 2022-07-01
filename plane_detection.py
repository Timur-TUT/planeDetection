# @author Timur
# @author Yasu

from collections import deque
import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

# 処理の概形
# 入力は2次元になった
def fast_plane_extraction(depth_image):
    # データ構造構築(ステップ1)
    nodes, edges = initgraph(depth_image)
    # # 粗い平面検出(ステップ2)
    # boudaries, pai = ahcluster(nodes, edges)
    # # 粗い平面検出を精緻化(ステップ3)
    # cluster, pro_pai = refine(boudaries, pai)
    # return cluster, pro_pai
    return None, None

# グループのクラス
class Node:
    def __init__(self, node):
        self.node = node    # np.array型の点群の集合(10×10)
        self.group_num = 0 # グループの番号,初期値0
        self.left = None
        self.right = None
        self.up = None
        self.down = None

    # 上下左右をみる
    def look_around(self, nodes, index_line, index):
        if index - 1 >= 0:
            self.left = nodes[index_line][index - 1]
        else:
            self.left = None
        if index + 1 < len(nodes[0]):
            self.right = nodes[index_line][index + 1]
        else:
            self.right = None
        if index_line - 1 >= 0:
            self.up = nodes[index_line - 1][index]
        else:
            self.up = None
        if index_line < len(nodes):
            self.down = nodes[index_line + 1][index]
        else:
            self.down = None

# データ構造構築
def initgraph(depth_image, h=10, w=10):
    nodes = []

    # 10×10が横にいくつあるかの数
    width = len(depth_image[0]) // w 
    height = len(depth_image) // h
    for i in range(height):
        nodes_line = []
        for j in range(width):
            # node は論文の v
            node = Node(depth_image[i*h:i*h+h-1,j*w:j*w+w-1])
            # nodeの除去の判定
            if rejectnode(node.node):
                node = Node(None)
            nodes_line.append(node)
        nodes.append(nodes_line)
    # 上下左右のノードを見つける
    for index_line in range(nodes):
        for index in range(index_line):
            if node[index_line][index] != None:
                node[index_line][index].look_around(nodes, index_line, index)
    for node in nodes:
        if not rejectedge(nodes[node.left].node, node.node, nodes[node.right].node):
            edges.push()
            # edges = edges.append([nodes[i-1], nodes[i], nodes[i+1]])
        if not rejectedge(nodes[node.up].node, node.node, nodes[node.down].node):
            edges.push()
            # edges = edges.append([nodes[i-num], nodes[i], nodes[i+num]])
    return nodes

# ノードの除去
def rejectnode(node):
    # 周囲4点との差
    for i in range(len(node)):
        if (i - 1 < 0) or (i + 1 > len(node) - 1):
            continue
        for j in range(len(node[0])):
            if (j - 1 < 0) or (j + 1 > len(node) - 1):
                continue
            if (abs(node[i][j] - node[i-1][j]) > 999) or (abs(node[i][j] - node[i+1][j]) > 999) or (abs(node[i][j] - node[i][j-1]) > 999) or (abs(node[i][j] - node[i][j+1]) > 999):
                return True
    # データが欠落していたら    
    if node == None:
        return True
    # まだ決めていない
    elif mse(node) > 999999:
        return True
    else:
        return False

# 連結関係の除去
def rejectedge(node1, node2, node3):
    # 欠落していなければ
    if (node1 == None) or (node2 == None) or (node3 == None):
        return True 
    # 法線のなす角
    # 一定値（まだ決めていない）
    elif np.cross(node1, node2) > 999999:
        return True
    elif np.cross(node2, node3) > 999999:
        return True
    else:
        return False

# 平均二乗誤差の計算
def mse(node):
    if node == None:
        return math.inf
    else:
        pca = PCA()
        return mean_squared_error(node, pca.fit(node))


# 粗い平面検出
def ahcluster(nodes, edges):
    # MSEの昇順のヒープを作る
    queue = buildminmseheap(nodes)
    boudaries = np.array()
    pai = np.array()
    # queueの中身がある限り
    while queue != []:
        suf = popmin(queue)
        # vがマージされているならば
        if suf not in nodes:
            continue
        u_best = np.array()
        u_merge = np.array()
        # vと連結関係にあるuを取り出して
        x_node, index = edges.catch(suf)
        for cand in suf.links:
            # 連結関係のノードをマージする
            # 縦なら1行目,横なら2行目
            u_test = np.append(u, v, axis=0)
            u_test = np.append(u, v, axis=1)
            # 一番MSEが小さいノードを選ぶ
            if mse(u_test) < mse(u_merge):
                # 連結しているノードの中で一番優秀なものをbest,くっつけた状態のものをmargeへ暫定的に
                u_best = u
                u_merge = u_test
        # マージ失敗
        if mse(u_merge) >= 2500:
            # vの大きさが一定以上ならば
            if abs(v) >= 100:
                # 平面とみなす
                boudaries = boudaries.append(v)
                pai = pai.append(plane(v))
                # 差集合
                # 連結関係の部分を削除？
                # uvの連結関係を追加
                edges = edges.remove()
                nodes = nodes.remove(v)
        # マージ成功
        # マージ後のu_mergeをそれぞれに追加しvとu_bestをそれぞれ削除する
        # エッジ収縮
        else:
            queue.append(u_merge)
            edges = edges.append()
            edges = edges.remove(u_best)
            edges = edges.remove(v)
            nodes = nodes.append(u_merge)
            nodes = nodes.remove(v)
            nodes = nodes.remove(u_best)
    return boudaries, pai

# 平面近似PCA
def plane(v):
    pca = PCA()
    return pca.fit(v)

# ヒープの作成
def buildminmseheap(nodes):
    # 1つずつmseを計算し直して並べ替える
    queue = sorted(nodes, key=mse)
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

if __name__ == '__main__':
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Convert images to numpy arrays
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data()) #計算に使用
        color_image = np.asanyarray(color_frame.get_data())

        cluster, pro_pai = fast_plane_extraction(depth_image)

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
        if key == ord("q") or (key != -1) or (prop_val < 0):
            # Stop streaming
            pipeline.stop()
            cv2.destroyAllWindows()
            break