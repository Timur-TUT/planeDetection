# @author Timur
# @author Yasu

from collections import deque
import math
from platform import node
import cv2
import numpy as np
import pyrealsense2 as rs
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

# 面番号
NUMBERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 処理の概形
# 入力は2次元になった
def fast_plane_extraction(depth_image):
    # データ構造構築(ステップ1)
    nodes, edges = init_graph(depth_image)
    print(f"num of edges: {len(edges)}")

    # # 粗い平面検出(ステップ2)
    # boudaries, pai = ahcluster(nodes, edges)
    # # 粗い平面検出を精緻化(ステップ3)
    # cluster, pro_pai = refine(boudaries, pai)
    # return cluster, pro_pai
    return nodes, None

# グループのクラス
class Node:
    def __init__(self, data):
        self.data = data    # np.array型の深さの集合(10×10)
        self.g_num = 0 # グループの番号,初期値0
        self.rejected = False #reject_nodeに当てはまったかどうか
        self.center = np.average(data)
        self.t_mse = (1.6e-6*self.center**2+5)**2

        # 上下左右のノード
        self.left = None
        self.right = None
        self.up = None
        self.down = None

# データ構造構築
def init_graph(depth_image, h=10, w=10):
    nodes = []
    rejected_nodes = []
    edges = []

    # 10×10が横にいくつあるかの数
    width = len(depth_image[0]) // w 
    height = len(depth_image) // h

    for i in range(height):
        nodes_line = []
        for j in range(width):
            # node は論文の v
            node = Node(depth_image[i*h:i*h+h,j*w:j*w+w])
            # nodeの除去の判定
            if reject_node(node):
                node.rejected = True
                rejected_nodes.append(node)
            nodes_line.append(node)
        nodes.append(nodes_line)

    # print(f"rejectされていないノード数: {sum(len(v) for v in nodes) - len(rejected_nodes)}")
    # rand = random.randint(0, len(rejected_nodes))
    # print(f"rejectされたノードの例: {[node.data for node in rejected_nodes[rand:rand+1]]}")
    # print("-------------------------------------------------------------------------------")

    # 右・下の繋がりを調べる
    for i in range(len(nodes)):
        for j in range(len(nodes[0])):
            if not nodes[i][j].rejected:
                if j+1<len(nodes[0]) and not rejectedge(nodes[i][j], nodes[i][j+1]): #書き方不安
                    nodes[i][j].right = nodes[i][j+1]
                    nodes[i][j+1].left = nodes[i][j]
                    edges.append(nodes[i][j])
                if i+1<len(nodes) and not rejectedge(nodes[i][j], nodes[i+1][j]):
                    nodes[i][j].down = nodes[i+1][j]
                    nodes[i+1][j].up = nodes[i][j]
                    edges.append(nodes[i][j])

    return nodes, list(set(edges))

# ノードの除去
def reject_node(node, threshold=999):
    data = node.data.astype(np.int32)
    # データが欠落していたら
    if np.any(data == 0):
        return True

    # 周囲4点との差
    v_diff = np.abs(np.diff(data, axis=0))  #縦方向の差
    h_diff = np.abs(np.diff(data))          #横方向の差

    if v_diff.max() > threshold or h_diff.max() > threshold:
        return True

    # まだ決めていない
    if mse(data) > node.t_mse:
        return True

    return False

# 連結関係の除去
def rejectedge(node1, node2):
    # 欠落していなければ
    if np.any(node2.data == 0):
        return True 

    normals = []

    for data in [node1.data, node2.data]:
        eig = np.linalg.eig(data)   # [0]: 固有値 shape(10, )      [1]: 固有ベクトル shape(10, 10)
        normals.append(eig[1][:, np.argmin(eig[0])])

    # pfn = np.dot(normals[0], normals[1])
    pfn = np.abs(np.sum(normals[0]*normals[1]))
    # print(f"pfn: {eig[0]}")
    # print("-----------------------------------------------------------")

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
    # queue = build_mse_heap(nodes)
    queue = edges
    boudaries = np.array()
    pai = np.array()
    suf_choice = None
    suf_best = np.zeros(1)

    # queueの中身がある限り
    while queue != []:
        suf = popmin(queue)
        # vがマージされているならば
        """
        if suf not in nodes:
            continue
        """

        # 面番号がなければ定義する
        if suf.g_num == 0:
            suf.g_num = NUMBERS.pop()

        # 連結関係のノードをマージする
        try:
            suf_test = np.hstack[suf.left.data, suf.data]
        except:
            pass

        # 一番MSEが小さいノードを選ぶ
        if mse(suf_test) < mse(suf_best):
            # 連結しているノードの中で一番優秀なものをbest,くっつけた状態のものをchoiceへ暫定的に
            if suf.left.g_num == suf.g_num:
                pass
            else:
                suf_best = suf_test
                suf_choice = suf.left

        try:
            suf_test = np.hstack[suf.data, suf.right.data]
        except:
            pass

        if mse(suf_test) < mse(suf_best):
            # 連結しているノードの中で一番優秀なものをbest,くっつけた状態のものをchoiceへ暫定的に
            if suf.right.g_num == suf.g_num:
                pass
            else:
                suf_best = suf_test
                suf_choice = suf.right

        try:
            suf_test = np.vstack[suf.up.data, suf.data]
        except:
            pass

        if mse(suf_test) < mse(suf_best):
            # 連結しているノードの中で一番優秀なものをbest,くっつけた状態のものをchoiceへ暫定的に
            if suf.up.g_num == suf.g_num:
                pass
            else:
                suf_best = suf_test
                suf_choice = suf.up
        
        try:
            suf_test = np.vstack[suf.data, suf.down.data]
        except:
            pass

        if mse(suf_test) < mse(suf_best):
            # 連結しているノードの中で一番優秀なものをbest,くっつけた状態のものをchoiceへ暫定的に
            if suf.down.g_num == suf.g_num:
                pass
            else:
                suf_best = suf_test
                suf_choice = suf.down

        # マージ失敗
        if mse(suf_best) >= 999999:
            #boudaries = boudaries.append(suf)
            pai = pai.append(plane(suf_best))

        # マージ成功
        else:
            if not suf_choice != None:
                queue.append(suf)
                suf_choice.g_num = suf.g_num
    return pai

# 平面近似PCA
def plane(v):
    pca = PCA()
    return pca.fit(v)

# ヒープの作成
def build_mse_heap(nodes):
    # 1つずつmseを計算し直して並べ替える
    # クラスのメンバーとしてmseの値が必要？
    queue = sorted(nodes, key=mse)
    return queue

# 先頭を取り出す
def popmin(queue):
    choice = queue.popleft()
    return choice

# 粗い平面検出の精緻化
def refine(edges, pai):
    queue = edges
    refine = np.array()
    rf_nodes = np.array()
    rf_edges = np.array()
    suf_num = 0

    for node in edges:
        suf_num = node.g_num
        # 上下左右のノードと番号が違ったら境界
        if (suf_num != node.left.g_num) or (suf_num != node.right.g_num) or (suf_num != node.up.g_num) or (suf_num != node.down.g_num):
            
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
    color_image[::10] = [0, 0, 0]
    color_image[:, ::10] = [0, 0, 0]
    
    for h in range(0, len(color_image), 10):
        for w in range(0, len(color_image[0]), 10):
            if nodes[h//10][w//10].rejected:
                color_image[h+4:h+7, w+4:w+7] = [0, 0, 255]

            if nodes[h//10][w//10].up:
                color_image[h-5:h+5, w+5] = [102, 0, 255]
            if nodes[h//10][w//10].down:
                color_image[h+5:h+15, w+5] = [102, 0, 255]
            if nodes[h//10][w//10].left:
                color_image[h+5, w-5:w+5] = [102, 0, 255]
            if nodes[h//10][w//10].right:
                color_image[h+5, w+5:w+15] = [102, 0, 255]

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

        cluster, pro_pai = fast_plane_extraction(depth_image)

        color_image = visualization(color_image, cluster)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_scale = cv2.convertScaleAbs(depth_image, alpha=0.03)  #0～255の値にスケール変更している

        depth_colormap = cv2.applyColorMap(depth_scale, cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        cv2.imwrite("debug_image.png", images)

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