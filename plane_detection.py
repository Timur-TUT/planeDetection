# @author Timur
# @author Yasu

from collections import deque
import math
import cv2
import numpy as np
import scipy.io
import random
import sys
import pyrealsense2 as rs
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from functools import singledispatch

# Nodeのタイプ
TYPE_NORMAL = 0
TYPE_MISSING_DATA = 1
TYPE_DEPTH_DISCONTINUE = 2
TYPE_BIG_MSE = 3

REAL_TIME = False
sys.setrecursionlimit(10000) 

# 処理の概形\
# 入力は2次元になった
def fast_plane_extraction(verts):
    global NODES
    # データ構造構築(ステップ1)
    NODES, edges = init_graph(verts)

    # 粗い平面検出(ステップ2)
    boudaries, pai = ahcluster(NODES, edges)
    # # 粗い平面検出を精緻化(ステップ3)
    # cluster, pro_pai = refine(boudaries, pai)
    # return cluster, pro_pai
    return NODES, None

# Nodeのクラス
class Node:
    def __init__(self, data, index):        #data: 点群の配列   ind: Nodeの座標
        self.data = np.nan_to_num(data)     # 点群の配列　np.array型(10×10)
        self.g_num = 0                      # グループの番号,初期値0
        self.node_type = TYPE_NORMAL        # Nodeのタイプ
        self.ind = index                    # 左上からの座標

        # つながりがある上下左右のNode
        self.left = None
        self.right = None
        self.up = None
        self.down = None
        self.edges = []

    def set_params(self, phase):   # 閾値定義
        if phase == 0:
            self.t_mse = (1.6e-6*self.center[2]**2+5)**2
        else:
            self.t_mse = (1.6e-6*self.center[2]**2+8)**2
        self.t_mse = 15

    def compute(self):
        self.center = [np.average(self.data[..., 0]), np.average(self.data[..., 1]), np.average(self.data[..., 2])]    # 平均(重心)
        self.cov = np.cov(self.data[..., :2].reshape(self.data.size//3, 2).T, self.data[..., 2].reshape(self.data.size//3,), rowvar=True)     # 分散共分散行列

        eig = np.linalg.eig(self.cov)   
        ind = np.argmin(eig[0])
        self.mse = eig[0][ind]  # 最小の固有値

        if eig[1][ind,0]*self.center[0] + eig[1][ind,1]*self.center[1] + eig[1][ind,2]*self.center[2] <= 0:
            self.normal = eig[1][ind]  #固有ベクトル(法線)
        else:
            self.normal = -eig[1][ind]

    def calculate_pf_mse(self, data):
        cov = np.cov(data[..., :2].reshape(data.size//3, 2).T, data[..., 2].reshape(data.size//3,), rowvar=True)     # 分散共分散行列

        eig = np.linalg.eig(cov)   
        ind = np.argmin(eig[0])
        mse = eig[0][ind]  # 最小の固有値

        return mse

class Plane(Node):
    def __init__(self, node_a, node_b):
        self.data = np.concatenate([node_a.data, node_b.data])
        self.nodes = [node_a, node_b]
        self.g_num = node_a.g_num = node_b.g_num =  NUMBERS.pop(0)
        self.add_edges(node_a, node_b)
        self.compute()

    def add_edges(self, node_a, node_b):
        self.edges = list(set(node_a.edges + node_b.edges))
        for rm in self.nodes:
            if rm in self.edges:
                self.edges.remove(rm)

    def push(self, new_node):
        self.data = np.concatenate([self.data, new_node.data])
        self.nodes.append(new_node)
        new_node.g_num = self.g_num
        self.add_edges(self, new_node)
        self.compute()

    def clear(self):
        self.g_num = 0
        # self.nodes = list(map(lambda node: node.g_num * 0, self.nodes))
        for node in self.nodes:
            node.g_num = 0

# データ構造構築
def init_graph(verts, h=10, w=10):
    global NUMBERS, colors
    NUMBERS = list(range(1,3001))    # 面番号
    colors = []
    for i in range(3000):
        colors.append(list(np.random.choice(range(256), size=3)))

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
            node = Node(verts[i*h:i*h+h,j*w:j*w+w], [i, j])
            # nodeの除去の判定
            if reject_node(node):
                rejected_nodes.append(node)
            nodes_line.append(node)
        nodes.append(nodes_line)

    # 右・下の繋がりを調べる
    for i in range(len(nodes)):
        for j in range(len(nodes[0])):
            if nodes[i][j].node_type == TYPE_NORMAL:
                if j+1<len(nodes[0]) and not rejectedge(nodes[i][j], nodes[i][j+1]):
                    nodes[i][j].right = nodes[i][j+1]
                    nodes[i][j].edges.append(nodes[i][j+1])

                    nodes[i][j+1].left = nodes[i][j]
                    nodes[i][j+1].edges.append(nodes[i][j])

                    edges.append(nodes[i][j])
                if i+1<len(nodes) and not rejectedge(nodes[i][j], nodes[i+1][j]):
                    nodes[i][j].down = nodes[i+1][j]
                    nodes[i][j].edges.append(nodes[i+1][j])

                    nodes[i+1][j].up = nodes[i][j]
                    nodes[i+1][j].edges.append(nodes[i][j])

                    edges.append(nodes[i][j])

    edges = list(set(edges))
    
    print(f"rejectされていないノード数: {sum(len(v) for v in nodes) - len(rejected_nodes)}")
    print(f"エッジ数: {len(edges)}")
    # rand = random.randint(0, len(rejected_nodes))
    # print(f"rejectされたノードの例: {[node.data for node in rejected_nodes[rand:rand+1]]}")
    # print(f"reject理由: {[node.node_type for node in rejected_nodes[rand:rand+1]]}")
    # print("-------------------------------------------------------------------------------")

    return nodes, edges

# ノードの除去
def reject_node(node, threshold=25):
    data = node.data

    # データが欠落していたら
    if np.any(data[..., -1] == 0):
        node.node_type = TYPE_MISSING_DATA
        return True

    # 周囲4点との差
    v_diff = np.linalg.norm(data[:-1] - data[1:], axis=2)
    h_diff = np.linalg.norm(data[:, :-1] - data[:, 1:], axis=2)

    if v_diff.max() > threshold or h_diff.max() > threshold:
        node.node_type = TYPE_DEPTH_DISCONTINUE
        return True

    node.compute()
    node.set_params(0)
    # print(f"mse = {node.mse}    t = {node.t_mse}")
    # with open("mse_my.txt", mode="a") as f:
    #     f.write(str(node.mse) + "\n")
    #     f.write(str(node.t_mse) + "\n" + "\n")
    if node.mse > node.t_mse:
        node.node_type = TYPE_BIG_MSE
        return True

    return False

# 連結関係の除去
def rejectedge(node1, node2):
    # 欠落していなければ
    if node2.node_type != TYPE_NORMAL:
        return True 

    pfn = np.abs(np.dot(node1.normal, node2.normal))

    if  pfn <= 0.7:
        return True

    return False

# 粗い平面検出
def ahcluster(nodes, edges):
    global color_image, NODES, data

    # MSEの昇順のヒープを作る
    queue = build_mse_heap(edges)
    boudaries = []
    flatten_nodes = sum(nodes, [])
    pai = np.array([])

    # queueの中身がある限り
    while queue != [] and NUMBERS:
        suf = queue.pop(0)
        print(len(queue))
        # print(suf.data.size//3)

        # if suf.g_num != 0:
        #     # 周りが同じ面ならばみない
        #     if (suf.left and suf.g_num == suf.left.g_num) and (suf.right and suf.g_num == suf.right.g_num) and (suf.up and suf.g_num == suf.up.g_num) and (suf.down and suf.g_num == suf.down.g_num):
        #         continue

        # vがマージされているならば
        if suf not in flatten_nodes:
            continue
        if len(suf.edges) <= 0:
            if len(suf.nodes) >= 10:
                boudaries.append(suf)
            else:
                suf.clear()
                
            continue

        # vと連結関係にあるuを取り出して
        # 連結関係のノードをマージする
        best_mse = [math.inf, None]
        # print(len(suf.edges))
        for edg in suf.edges:
            current_mse = suf.calculate_pf_mse(edg.data)
            if current_mse < best_mse[0]:
                best_mse = [current_mse, edg]

        # print(len(suf.edges))

        # マージ失敗
        suf.set_params(1)
        # print(best_mse[0])
        if best_mse[0] >= suf.t_mse:
            # print("aaaaaaaaaaaaaaaaaaaaaaaaaaa")
            flatten_nodes.remove(suf)
            if suf.data.size >= 800:
                boudaries.append(Plane(suf, best_mse[1]))

        # マージ成功
        else:
            for rm in [suf, best_mse[1]]:
                if rm in flatten_nodes:
                    flatten_nodes.remove(rm)
            if type(suf) == Node:
                suf = Plane(suf, best_mse[1])
            else:
                suf.push(best_mse[1])
            queue.insert(0, suf)
            # queue.append(suf)
            flatten_nodes.append(suf)
            # if best_mse[1] in queue:
            #     queue.remove(best_mse[1])

        # color_image = data[0, 0][1]
        # color_image = visualization(color_image, NODES)

        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('RealSense', color_image)
        # cv2.waitKey(50)


    # for i in range(10, 100, 5):
    #     print(f"members: {len(boudaries[i].nodes)}")
    #     print(f"group: {boudaries[i].g_num}")
    #     print(f"edges: {len(boudaries[i].edges)}")
    #     print("------------------------")
    return boudaries, pai

# ヒープの作成
def build_mse_heap(edges):
    # 1つずつmseを計算し直して並べ替える)
    queue = sorted(edges, key=lambda node: node.mse)
    return queue

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
    global colors
    color_image = np.copy(color_image)
    

    BLACK = [0, 0, 0]
    RED = [0, 0, 255]
    ORANGE = [17, 144, 255]
    PURPLE = [255, 0, 127]
    PINK = [102, 0, 255]
    GRAY = [128, 128, 128]
    GREEN = [51, 255, 51]
    BLUE = [255, 90, 0]

    # color_image[::10] = BLACK
    # color_image[:, ::10] = BLACK    
    
    for h in range(0, len(color_image), 10):
        for w in range(0, len(color_image[0]), 10):
            # if nodes[h//10][w//10].node_type == TYPE_MISSING_DATA:
            #     color_image[h+4:h+7, w+4:w+7] = BLACK
            # elif nodes[h//10][w//10].node_type == TYPE_DEPTH_DISCONTINUE:
            #     color_image[h+4:h+7, w+4:w+7] = ORANGE
            # elif nodes[h//10][w//10].node_type == TYPE_BIG_MSE:
            #     color_image[h+4:h+7, w+4:w+7] = RED

            # if nodes[h//10][w//10].up:
            #     color_image[h-5:h+5, w+5] = PURPLE
            # if nodes[h//10][w//10].down:
            #     color_image[h+5:h+15, w+5] = PURPLE
            # if nodes[h//10][w//10].left:
            #     color_image[h+5, w-5:w+5] = PURPLE
            # if nodes[h//10][w//10].right:
            #     color_image[h+5, w+5:w+15] = PURPLE

            if nodes[h//10][w//10].g_num != 0:
                color_image[h+1:h+10, w+1:w+10] = colors[nodes[h//10][w//10].g_num+1]

    return color_image

if __name__ == '__main__':
    if not REAL_TIME:
        data = scipy.io.loadmat('frame.mat')["frame"]
        verts = data[0, 0][0]
        color_image = data[0, 0][1]

        cluster, pro_pai = fast_plane_extraction(verts)

        color_image = visualization(color_image, cluster)

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        key = cv2.waitKey(0)
        if key == ord("s"):
                cv2.imwrite('./out_nr.png', color_image)
        cv2.destroyAllWindows()
    
    else:
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
