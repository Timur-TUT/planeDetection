# @author Timur
# @author Yasu

from ast import Num
from collections import deque
import math
import queue
from xml.dom.minicompat import NodeList
import cv2
import numpy as np
import scipy.io
import random
import sys
import pyrealsense2 as rs
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

# Reject理由
MISSING_DATA = 0
BIG_DIFF = 1
BIG_MSE = 2

REAL_TIME = False

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
        self.data = np.nan_to_num(data)    # np.array型の深さの集合(10×10)
        self.g_num = 0 # グループの番号,初期値0
        self.reject_type = None #Rejectされた理由　Noneの場合はされていない
        self.compute()
        
        # self.t_mse = (1.6e-6*self.center[2]**2+5)**2

        # 上下左右のノード
        self.left = None
        self.right = None
        self.up = None
        self.down = None
        self.edge = []

    def compute(self):
        self.center = [np.average(self.data[..., 0]), np.average(self.data[..., 1]), np.average(self.data[..., 2])]    # 平均(重心)
        self.cov = np.cov(self.data[..., :2].reshape(self.data.size//3, 2).T, self.data[..., 2].reshape(self.data.size//3,), rowvar=True)     # 分散共分散行列

        eig = np.linalg.eig(self.cov)   
        ind = np.argmin(eig[0])
        self.mse = eig[0][ind] * self.data.size    # 最小の固有値

        if eig[1][ind,0]*self.center[0] + eig[1][ind,1]*self.center[1] + eig[1][ind,2]*self.center[2] <= 0:
            self.normal = eig[1][ind]  #固有ベクトル(法線)
        else:
            self.normal = -eig[1][ind]
    
    def push(self, node, success=False):
        self.prev_data = np.copy(self.data)
        self.data = np.concatenate([self.data, node.data])
        self.compute()
        if success:
            self.edge.remove(node)
            self.edge += node.edge
            node.g_num = self.g_num

    def undo(self):
        self.data = np.copy(self.prev_data)
        # self.compute()


# データ構造構築
def init_graph(verts, h=10, w=10):
    global NUMBERS
    nodes = []
    rejected_nodes = []
    edges = []
        
    NUMBERS = list(range(1,10000))    # 面番号

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
                    nodes[i][j].edge.append(nodes[i][j+1])

                    nodes[i][j+1].left = nodes[i][j]
                    nodes[i][j+1].edge.append(nodes[i][j])

                    edges.append(nodes[i][j])
                if i+1<len(nodes) and not rejectedge(nodes[i][j], nodes[i+1][j]):
                    nodes[i][j].down = nodes[i+1][j]
                    nodes[i][j].edge.append(nodes[i+1][j])

                    nodes[i+1][j].up = nodes[i][j]
                    nodes[i+1][j].edge.append(nodes[i][j])

                    edges.append(nodes[i][j])

    edges = list(set(edges))
    
    print(f"rejectされていないノード数: {sum(len(v) for v in nodes) - len(rejected_nodes)}")
    print(f"エッジ数: {len(edges)}")
    rand = random.randint(0, len(rejected_nodes))
    # print(f"rejectされたノードの例: {[node.data for node in rejected_nodes[rand:rand+1]]}")
    # print(f"reject理由: {[node.reject_type for node in rejected_nodes[rand:rand+1]]}")
    # print("-------------------------------------------------------------------------------")

    return nodes, edges

# ノードの除去
def reject_node(node, threshold=100):
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

    if node.mse > 2000:   #閾値適当
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
    queue = build_mse_heap(edges)
    boudaries = np.array([])
    pai = np.array([])

    # queueの中身がある限り
    while queue != []:
        suf = queue.pop()

        # 周りが同じ面ならばみない
        if (suf.left and suf.g_num == suf.left.g_num) and (suf.right and suf.g_num == suf.right.g_num) and (suf.up and suf.g_num == suf.up.g_num) and (suf.down and suf.g_num == suf.down.g_num):
            continue

        # vがマージされているならば
        """
        if suf not in nodes:
            continue
        """

        # 面番号がなければ定義する
        if suf.g_num == 0:
            suf.g_num = NUMBERS.pop()

        # vと連結関係にあるuを取り出して
        # 連結関係のノードをマージする
        best_mse = [math.inf, None]
        for edg in suf.edge:
            suf.push(edg)
            if suf.mse < best_mse[0]:
                best_mse = [suf.mse, edg]
            suf.undo()

        # マージ失敗
        if best_mse[0] >= 0.001:
            continue
            boudaries = boudaries.append(suf)
            pai = pai.append(plane(suf_best))

        # マージ成功
        else:
            suf.push(best_mse[1], True)
            """
            # くっつけた相手の情報をqueueから消して、dataをくっつけた状態にした方が速い？
            queue.remove(best_mse[1])
            best_mse[1].push(suf, True)
            """
            queue.append(suf)
            # suf_choice.g_num = suf.g_num
    return boudaries, pai

# 平面近似PCA
def plane(v):
    pca = PCA()
    return pca.fit(v)

# ヒープの作成
def build_mse_heap(edges):
    # 1つずつmseを計算し直して並べ替える)
    queue = sorted(edges, key=lambda node: node.mse)
    print(queue[len(queue)//2].mse)
    print(queue[10].mse)
    print(queue[-1].mse)
    return queue

# 先頭を取り出す
def popmin(queue):
    choice = queue.popleft()
    return choice

# 粗い平面検出の精緻化
def refine(boundaries, pai):
    # 境界面のpointのみを集める
    queue = build_boud_heap(boundaries)
    while queue != None:
        point = queue.popleft()
        # やりたいことだけを書く(理想)
        # pointの上下左右をみたい ← クラス化？
        check_ref_points(point, boundaries, queue)

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

def build_boud_heap(nodes):
    queue = deque()
    # 上下左右のどれか一つでも隣接するものが無ければ平面から除去
    for node in nodes:
        if (node.left == None) or (node.right == None) or (node.up == None) or (node.down == None):
            node.g_num = 0
    # 除去した上で平面境界nodeをpointまで分割してqueueに追加
    for node in nodes:
        num = node.g_num 
        # 面に属していて
        if num != 0:
            # 境界ならば
            if (node.left.g_num != num) or (node.right.g_num != num) or (node.up.g_num != num) or (node.down.g_num != num):
                # pointまで分割したもののqueueを作る
                for point in node.data: # ←初期の10×10のポイントが欲しい(現状だとマージ後の大きさになっている？)
                    queue.append(point)
    return queue

# pointクラスがあること前提
def check_ref_points(point, nodes, queue):
    ref_edges = []
    count = 0
    # 左右上下の順で取り出す
    adj = point.left
    if adj != None:
        meke_refine(adj, point, nodes, queue, 1)
        
        # 同じnode内か
        if adj.node == point.node:
            continue
        # node(k)平面との距離がnode(k)平面のMSEの9倍以上か
        elif (abs(adj.coordinate - nodes[point.node].center) >= nodes[point.node].mse*9) :
            continue
        # 同じ平面内か
        if adj.g_num == point.g_num:
            continue
        # そもそも平面に属していない場合
        elif adj.g_num == 0:
            append_refine(adj, point)
            queue.aapend(adj)
        # 違う平面だった場合
        else:
            adj_node = nodes[adj.node]
            point_node = nodes[point.node]
            # 中心からの距離を
            make_edges(adj, point, queue, adj_node.center, point_node.center)

    return True

def make_adj

# 平面に属していないpointを平面に追加する
def append_refine(adj, point):
    # 同じ平面とみなす
    adj.g_num = point.g_num
    # nodeに追加する(10×10じゃなくなるが)
    # やり方要相談
    return True

# 違う平面に属していた場合
def  make_edges(adj, point, queue, adj_center, point_center):
    adj_dis = abs(adj.coordinate - adj_center)
    point_dis = abs(adj.coordinate - point_center)
    # 面からの中心距離がpointの属するnodeの方が近いならば
    if adj_dis > point_dis:
        # 同じ平面とみなす
        adj.g_num = point.num
        # nodeの所属を変更する(削除して追加する)
        # やり方要相談
        queue.append(adj)
    return True

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

            # if nodes[h//10][w//10].g_num != 0:
            #     color_image[h+1:h+10, w+1:w+10] = [nodes[h//10][w//10].g_num+100]

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