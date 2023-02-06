# @author Timur
# @author Yasu

"""深さ情報を持つ映像・画像における面検知

深さ情報を一定の大きさのごとに区切る(以後ノードと表す)
ノード同士の深さや角度により同一面とみなせるか判定する
一定の大きさになったものは面として抽出する
その後面の境界から未検出領域に点単位で深さを比較する
ランダムな色をつけて表示する

"""

from collections import deque
from turtle import distance
from tqdm import tqdm
import math
import cv2
import numpy as np
import scipy.io
import random
import sys
import pyrealsense2 as rs
import time

# Nodeのタイプ
TYPE_NORMAL = 0
TYPE_MISSING_DATA = 1
TYPE_DEPTH_DISCONTINUE = 2
TYPE_BIG_MSE = 3

# マージのタイプ
PUSH = 0
POP = 1

START = "CALLED"
END = "DONE"

# 表示用の色
BLACK = [0, 0, 0]
RED = [0, 0, 255]
ORANGE = [17, 144, 255]
PURPLE = [255, 0, 127]

# 表示用フラグ
REAL_TIME = False
DEFAULT = 0
INIT_GRAPH = 1

# 設定
sys.setrecursionlimit(10000)
np.set_printoptions(threshold=np.inf)

class Global:
    """グローバルで使う関数

    """
    def __init__(self, h=480, w=640, number_of_groups=3000):
        self.group_numbers = list(range(1,number_of_groups+1))    # 面番号
        self.colors = []
        for i in range(number_of_groups):
            self.colors.append(list(np.random.choice(range(256), size=3)))
        self.refine_edges = set()

        set_time()
        self.log_file = open('log.txt', 'x')

    def detect(self):
        """面検知の3ステップを行う
        """
        # データ構造構築(ステップ1)
        self.nodes, self.edges = init_graph(points, verts)

        # 粗い平面検出(ステップ2)
        self.planes = ahcluster(self.nodes, self.edges)

        # 粗い平面検出を精緻化(ステップ3)
        self.planes = refine(self.planes)

        for node in sum(glob.nodes, []):
            for point in node.members:
                point.g_num = node.g_num

        self.log_file.close()

    def visualize(self, show_depth=False, distinguish=None, mode=DEFAULT):

        """画像を表示する

        Args:
            show_depth (bool, optional): _description_. Defaults to False.
            distinguish (_type_, optional): _description_. Defaults to None.
            mode (_type_, optional): _description_. Defaults to DEFAULT.

        Returns:
            image : 色をつけた後の画像
        """

        image = np.copy(color_image)
        # image[::10] = BLACK
        # image[:, ::10] = BLACK    
        
        if mode == INIT_GRAPH:
            for h in range(0, len(image), 10):
                for w in range(0, len(image[0]), 10):
                    if mode == INIT_GRAPH:
                        if self.nodes[h//10][w//10].node_type == TYPE_MISSING_DATA:
                            image[h+4:h+7, w+4:w+7] = BLACK
                        elif self.nodes[h//10][w//10].node_type == TYPE_DEPTH_DISCONTINUE:
                            image[h+4:h+7, w+4:w+7] = ORANGE
                        elif self.nodes[h//10][w//10].node_type == TYPE_BIG_MSE:
                            image[h+4:h+7, w+4:w+7] = RED

                        if self.nodes[h//10][w//10].up:
                            image[h-5:h+5, w+5] = PURPLE
                        if self.nodes[h//10][w//10].down:
                            image[h+5:h+15, w+5] = PURPLE
                        if self.nodes[h//10][w//10].left:
                            image[h+5, w-5:w+5] = PURPLE
                        if self.nodes[h//10][w//10].right:
                            image[h+5, w+5:w+15] = PURPLE

                    # continue

                # if self.nodes[h//10][w//10].g_num != 0:
                #     image[h+1:h+10, w+1:w+10] = self.colors[self.nodes[h//10][w//10].g_num+1]
        else:
            for h in range(0, len(image)):
                for w in range(0, len(image[0])):
                    if points[h][w].g_num != 0:
                        image[h, w] = self.colors[points[h][w].g_num+1]

        if distinguish:
            image[distinguish[0]*10:distinguish[0]*10+11:10] = RED
            image[:, distinguish[1]*10:distinguish[1]*10+11:10] = RED

        if show_depth:
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_scale = cv2.convertScaleAbs(depth_image, alpha=0.03)  #0～255の値にスケール変更している

            depth_colormap = cv2.applyColorMap(depth_scale, cv2.COLORMAP_JET)

            # Stack both images horizontally
            image = np.hstack((image, depth_colormap))

        return image

class Point:
    """画像の点一つが持つ情報を作成するクラス

    """
    def __init__(self, data, cord):

        """init関数

        クラス作成時に定義される

        Args:
            data (_type_): _description_
            cord (_type_): _description_

        """

        self.data = data
        self.cord = cord
        self.node = None
        self.refined = None
        self.g_num = 0

class Stats():
    def __init__(self):
        self.sx, self.sy, self.sz = 0, 0, 0
        self.sxx, self.syy, self.szz = 0, 0, 0
        self.sxy, self.syz, self.sxz = 0, 0, 0
        self.N = 0

    def clear(self):
        self.sx = self.sy = self.sz = self.sxx = self.syy = self.szz = self.sxy = self.syz = self.sxz = self.N = 0

    def push_point(self, x, y, z):
        self.sx+=x; self.sy+=y; self.sz+=z
        self.sxx+=x*x; self.syy+=y*y; self.szz+=z*z
        self.sxy+=x*y; self.syz+=y*z; self.sxz+=x*z
        self.N += 1

    def push_stats(self, stats):
        self.sx+=stats.x; self.sy+=stats.y; self.sz+=stats.z
        self.sxx+=stats.sxx; self.syy+=stats.syy; self.szz+=stats.szz
        self.sxy+=stats.sxy; self.syz+=stats.syz; self.sxz+=stats.sxz
        self.N += stats.N

    def pop_point(self, x, y, z):
        self.sx-=x; self.sy-=y; self.sz-=z
        self.sxx-=x*x; self.syy-=y*y; self.szz-=z*z
        self.sxy-=x*y; self.syz-=y*z; self.sxz-=x*z
        self.N -= 1

    def pop_stats(self, stats):
        self.sx-=stats.x; self.sy-=stats.y; self.sz-=stats.z
        self.sxx-=stats.sxx; self.syy-=stats.syy; self.szz-=stats.szz
        self.sxy-=stats.sxy; self.syz-=stats.syz; self.sxz-=stats.sxz
        self.N -= stats.N

    def compute(self):
        self.center = [self.sx/self.N, self.sy/self.N, self.sz/self.N]

        self.K = [[self.sxx-self.sx*self.sx/self.N, self.sxy-self.sx*self.sy/self.N, self.sxz-self.sx*self.sz/self.N],
                   [                             0, self.syy-self.sy*self.sy/self.N, self.syz-self.sy*self.sz/self.N],
                   [                             0,                               0, self.szz-self.sz*self.sz/self.N]]
        self.K[1][0] = self.K[0][1]; self.K[2][0]=self.K[0][2]; self.K[2][1]=self.K[1][2]

        eig = np.linalg.eig(self.K)   
        ind = np.argmin(eig[0])
        self.mse = eig[0][ind]/self.N  # 最小の固有値
        self.curvature = eig[0][0]/(eig[0][0]+eig[0][1]+eig[0][2])

        if eig[1][ind,0]*self.center[0] + eig[1][ind,1]*self.center[1] + eig[1][ind,2]*self.center[2] <= 0:
            self.normal = eig[1][ind]  #固有ベクトル(法線)
        else:
            self.normal = -eig[1][ind]

def aaa(point, node):
    point.node = node

# とりあえず定義
class PlaneSeg():
    """ノードのクラス"""

    def __init__(self, points, data, index):
        
        """_summary_

        Args:
            points (_type_): _description_
            data (_type_): 点群の配列
            index (_type_): ノードの座標
        """

        self.data = np.reshape(np.nan_to_num(data), (100,3))     # 点群の配列　np.array型(10×10)
        self.members = set(points)
        for point in self.members:
            point.node = self
        
        self.g_num = 0                      # グループの番号,初期値0
        self.node_type = TYPE_NORMAL        # Nodeのタイプ
        self.ind = index                    # 左上からの座標
        self.refine_pc = set()
        self.plane = None

        # つながりがある上下左右のNode
        self.left = None
        self.right = None
        self.up = None
        self.down = None
        self.edges = set()

    def set_params(self, phase):

        """閾値を定義する

        Args:
            phase (_type_): _description_
        """

        if phase == 0:
            self.t_mse = (1.6e-6*self.center[2]**2+5)**2
        else:
            self.t_mse = (1.6e-6*self.center[2]**2+8)**2
        self.t_mse = 15

    def push(self):

        """点をノードに追加する

        ノードを構成する要素としてポイントを追加し
        データ配列にポイント要素を加え,ポイントの親ノード情報を更新する
    
        """

        for point in self.refine_pc:
            self.members.add(point)
            point.node = self
            if np.any(point.data == 0):
                continue
            self.data = np.vstack([self.data, point.data])

        # if self.node_type != TYPE_MISSING_DATA:
        self.compute()

    def compute(self):

        """ノードクラスの各値をそれぞれ定義する
        """

        self.center = [np.average(self.data[..., 0]), np.average(self.data[..., 1]), np.average(self.data[..., 2])]    # 平均(重心)
        self.cov = np.cov(self.data[..., :2].reshape(self.data.size//3, 2).T, self.data[..., 2].reshape(self.data.size//3,), rowvar=True)     # 分散共分散行列

        eig = np.linalg.eig(self.cov)   
        ind = np.argmin(eig[0])
        self.mse = eig[0][ind]  # 最小の固有値

        if eig[1][ind,0]*self.center[0] + eig[1][ind,1]*self.center[1] + eig[1][ind,2]*self.center[2] <= 0:
            self.normal = eig[1][ind]  #固有ベクトル(法線)
        else:
            self.normal = -eig[1][ind]
        self.set_color_vec()

    def calculate_pf_mse(self, new_data):

        """マージしたノードの分散共分散行列から平均二乗誤差を計算する

        Returns:
            _type_: _description_

        """
        
        data = np.concatenate([self.data, new_data])
        cov = np.cov(data[..., :2].reshape(data.size//3, 2).T, data[..., 2].reshape(data.size//3,), rowvar=True)     # 分散共分散行列

        eig = np.linalg.eig(cov)   
        ind = np.argmin(eig[0])
        mse = eig[0][ind]  # 最小の固有値

        return mse

    def set_color_vec(self):

        """ベクトルから固有値を求める
        """

        clx = ((self.normal[0] + 1.0) * 0.5 * 255.0) # unsigned char (0~255)
        cly = ((self.normal[1] + 1.0) * 0.5 * 255.0) # unsigned char (0~255)
        clz = ((self.normal[2] + 1.0) * 0.5 * 255.0) # unsigned char (0~255)
        self.normal_clr = [clx, cly, clz] # 例では発見できていない
        self.clr = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] # 表示で使ってる

class Plane(Node):
    """ノードをマージするためのクラス

    Args:
        Node (_tyape_): マージさせたいノード

    """

    def __init__(self, node_a, node_b):

        """オブジェクト作成時に実行される

        Args:
            node_a (_type_): マージ元のノード
            node_b (_type_): マージさせたいノード

        """

        self.data = np.concatenate([node_a.data, node_b.data])
        self.members = {node_a, node_b}
        node_a.plane = node_b.plane = self
        self.g_num = node_a.g_num = node_b.g_num = glob.group_numbers.pop(0)
        for node in self.members:
            for point in node.members:
                point.g_num = self.g_num
        self.add_edges(node_a, node_b)
        self.compute()
        self.calc_plane_d(self.data[0])

    def add_edges(self, node_a, node_b):

        """マージ後に連結情報を更新する

        マージ元のノードにマージしたノードの連結情報を追加し
        重複部分を消去する

        """

        self.edges = node_a.edges | node_b.edges
        for rm in self.members:
            if rm in self.edges:
                self.edges.remove(rm)

    def push(self, subject, has_plane=False):

        """ノードをマージする

        Args:
            subject (_type_): _description_
            has_plane (bool, optional): _description_. Defaults to False.

        """

        self.data = np.concatenate([self.data, subject.data])
        self.add_edges(self, subject)
        if has_plane:
            self.members.add(subject.members)
            subject.g_num = self.g_num
            for point in subject.members:
                point.g_num = self.g_num
        else:
            self.members.add(subject)
            subject.g_num = self.g_num
            for point in subject.members:
                point.g_num = self.g_num
        # self.compute()

    def pop(self, node):

        """ノードのマージを取り消す

        Args:
            node (_type_): マージを取り消したいノード

        """

        self.data= np.delete(self.data, np.argwhere(self.data == node.data))
        self.members.remove(node)
        node.g_num = 0
        # self.compute()

    def clear(self):

        """マージ関係を全て消去する

        面を構成するマージしたノードの情報を消去する
        それぞれのg_numを0にする

        """

        self.g_num = 0
        # self.members = list(map(lambda node: node.g_num * 0, self.members))   #出来なかったやつ
        for node in self.members:
            node.g_num = 0
            for point in node.members:
                point.g_num = 0

    def calc_plane_d(self, point_cord):

        """面の平面方程式を求める

        Args:
            point_cord (_type_): 面の中の任意の1点の座標(x, y, z)


        """

        # 法線ベクトルと通る点から求める
        self.d = -1*((self.normal[0]*point_cord[0]) + (self.normal[1]*point_cord[1]) + (self.normal[2]*point_cord[2]))

# データ構造構築
def init_graph(points, verts, h=10, w=10):

    """深さデータからノードとそれらのつながりを作成する

    深さデータを引数h,wで指定した値ごとに区切ったノード(オブジェクト)を作る
    その際にデータの欠損や極端な値がないかを調べあるものは欠損ノードとする
    作成したノードのつながりを調べクラスの繋がりの情報(エッジ)を更新する
    作成したノードの集合とエッジを持ったノードの集合を作成し戻り値とする

    Args:
        points (_type_): _description_
        verts (_type_): _description_
        h (int, optional): _description_. Defaults to 10.
        w (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_

    """

    log(sys._getframe().f_code.co_name, START)
    nodes = []
    rejected_nodes = set()
    edges = set()
        
    # 10×10が横にいくつあるかの数
    width = len(verts[0]) // w 
    height = len(verts) // h

    for i in range(height):
        nodes_line = []
        for j in range(width):
            # node は論文の v
            node = Node(sum([row[j*w:j*w+w] for row in points[i*h:i*h+h]], []), verts[i*h:i*h+h,j*w:j*w+w], [i, j])
            # nodeの除去の判定
            if reject_node(node):
                rejected_nodes.add(node)
            nodes_line.append(node)
        nodes.append(nodes_line)

    # 右・下の繋がりを調べる
    for i in range(len(nodes)):
        for j in range(len(nodes[0])):
            if nodes[i][j].node_type == TYPE_NORMAL:
                if j+1<len(nodes[0]) and not rejectedge(nodes[i][j], nodes[i][j+1]):
                    nodes[i][j].right = nodes[i][j+1]
                    nodes[i][j].edges.add(nodes[i][j+1])

                    nodes[i][j+1].left = nodes[i][j]
                    nodes[i][j+1].edges.add(nodes[i][j])

                    edges.add(nodes[i][j])
                if i+1<len(nodes) and not rejectedge(nodes[i][j], nodes[i+1][j]):
                    nodes[i][j].down = nodes[i+1][j]
                    nodes[i][j].edges.add(nodes[i+1][j])

                    nodes[i+1][j].up = nodes[i][j]
                    nodes[i+1][j].edges.add(nodes[i][j])

                    edges.add(nodes[i][j])
    
    print(f"rejectされていないノード数: {sum(len(v) for v in nodes) - len(rejected_nodes)}")
    print(f"エッジ数: {len(edges)}")
    # rand = random.randint(0, len(rejected_nodes))
    # print(f"rejectされたノードの例: {[node.data for node in rejected_nodes[rand:rand+1]]}")
    # print(f"reject理由: {[node.node_type for node in rejected_nodes[rand:rand+1]]}")
    # print("-------------------------------------------------------------------------------")

    log(sys._getframe().f_code.co_name, END)
    return nodes, edges

# ノードの除去
def reject_node(node, threshold=25):

    """_summary_

    Args:
        node (_type_): _description_
        threshold (int, optional): _description_. Defaults to 25.

    Returns:
        _type_: _description_

    """

    data = np.reshape(node.data, (10, 10, 3))

    # データが欠落していたら
    if np.any(data[..., -1] == 0):
        node.node_type = TYPE_MISSING_DATA
        return True

    node.compute()
    node.set_params(0)

    # 周囲4点との差
    v_diff = np.linalg.norm(data[:-1] - data[1:], axis=2)
    h_diff = np.linalg.norm(data[:, :-1] - data[:, 1:], axis=2)

    if v_diff.max() > threshold or h_diff.max() > threshold:
        node.node_type = TYPE_DEPTH_DISCONTINUE
        return True

    # print(f"mse = {node.mse}    t = {node.t_mse}")
    # with open("mse_my.txt", mode="a") as f:
    #     f.write(str(node.mse) + "\n")
    #     f.write(str(node.t_mse) + "\n" + "\n")
    if node.mse > node.t_mse:
        node.node_type = TYPE_BIG_MSE
        return True

    return False

def rejectedge(node1, node2):

    """連結関係を消去する

    Args:
        node1 (_type_): _description_
        node2 (_type_): _description_

    Returns:
        bool: _description_

    """

    # 欠落していなければ
    if node2.node_type != TYPE_NORMAL:
        return True 

    pfn = np.abs(np.dot(node1.normal, node2.normal))

    if  pfn <= 0.7:
        return True

    return False

def ahcluster(nodes, edges):

    """粗い平面検出を行う

    Args:
        nodes (_type_): ノードが含まれる集合
        edges (_type_): 連結関係をもったノードが含まれる集合

    Returns:
        _type_: _description_

    """

    log(sys._getframe().f_code.co_name, START)

    # MSEの昇順のヒープを作る
    queue = build_mse_heap(edges)
    planes = []
    flatten_nodes = sum(nodes, [])

    pbar = tqdm(total=len(queue))

    # queueの中身がある限り
    while len(queue) > 0:
        log(sys._getframe().f_code.co_name, None, "loop_start")
        suf = queue.popleft()
        log(sys._getframe().f_code.co_name, None, f"len of edges: {str(len(suf.edges))}")

        # vがマージされているならば
        if suf not in flatten_nodes:
            pbar.update(1)
            continue
        if len(suf.edges) <= 0:
            pbar.update(1)
            if len(suf.members) >= 10:
                planes.append(suf)
            else:
                suf.clear()
                
            continue

        # vと連結関係にあるuを取り出して
        # 連結関係のノードをマージする
        log(sys._getframe().f_code.co_name, None, "trying merge")
        best_mse = [math.inf, None]
        for edg in suf.edges:
            current_mse = suf.calculate_pf_mse(edg.data)
            if current_mse < best_mse[0]:
                best_mse = [current_mse, edg]

        # マージ失敗
        suf.set_params(1)
        # print(best_mse[0])
        if best_mse[0] >= suf.t_mse:
            log(sys._getframe().f_code.co_name, None, "merge failed")
            flatten_nodes.remove(suf)
            pbar.update(1)
            if len(suf.members) >= 10:
                planes.append(Plane(suf, best_mse[1]))

        # マージ成功
        # ここでループしているかも?
        else:
            log(sys._getframe().f_code.co_name, None, "merge successed")
            for rm in [suf, best_mse[1]]:
                if rm in flatten_nodes:
                    flatten_nodes.remove(rm)
            if type(suf) == Node:
                suf = Plane(suf, best_mse[1])
            else:
                if best_mse[1].plane:
                    best_mse[1].plane.push(suf)
                else:
                    suf.push(best_mse[1])
            queue.appendleft(suf)
            flatten_nodes.append(suf)

        log(sys._getframe().f_code.co_name, None, "loop_end")

    """
    # 演出用
    # image = glob.visualize()
    # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('RealSense', image)
    # cv2.waitKey(5)

    # for i in range(10, 100, 5):
    #     print(f"members: {len(planes[i].members)}")
    #     print(f"group: {planes[i].g_num}")
    #     print(f"edges: {len(planes[i].edges)}")
    #     print("------------------------")
    """
    pbar.close()

    log(sys._getframe().f_code.co_name, END)
    return planes

def set_time():

    """時間を設定する
    """

    global start_time
    start_time = time.time()

def log(func_name, flag=None, message=None):

    """経過時間を別ファイルに書き込む

    Args:
        func_name (_type_): 実行する関数名
        flag (_type_, optional): _description_. Defaults to None.
        message (_type_, optional): _description_. Defaults to None.

    """

    # calculate elapsed time
    elapsed_time = int(time.time() - start_time)

    # convert second to hour, minute and seconds
    elapsed_hour = elapsed_time // 3600
    elapsed_minute = (elapsed_time % 3600) // 60
    elapsed_second = (elapsed_time % 3600 % 60)

    glob.log_file.write(f"{func_name} {str(elapsed_hour).zfill(2)}:{str(elapsed_minute).zfill(2)}:{str(elapsed_second).zfill(2)}  :  ")

    if flag:
        glob.log_file.write(f"{flag}\n")
    if message:
        glob.log_file.write(f"{message}\n")

def build_mse_heap(edges):

    """mseの値で並び替えたキュー(ヒープ)を作成する

    Args:
        edges (_type_): 連結情報をもつノードの集合

    Returns:
        _type_: 作成したキュー

    """

    log(sys._getframe().f_code.co_name, START)

    # 1つずつmseを計算し直して並べ替える)
    queue = deque(sorted(edges, key=lambda node: node.mse))

    log(sys._getframe().f_code.co_name, END)
    return queue


def refine(planes):

    """粗い平面検出の精緻化

    Args:
        planes (_type_): 粗く検出した面の集合

    Returns:
        _type_: _description_

    """

    # 境界面のpointのみを集める
    glob.boud_queue = build_boud_heap(planes)
    # glob.pbar = tqdm(total=len(glob.boud_queue))

    while len(glob.boud_queue) > 0:
        # print(len(glob.boud_queue))
        point = glob.boud_queue.popleft()
        check_ref_points(point)
    
    for node in sum(glob.nodes, []):
        node.push()
        for point in node.members:
            point.g_num = node.g_num
    # glob.pbar.close()
    return ahcluster(glob.nodes, glob.refine_edges | glob.edges)

def build_boud_heap(planes):

    """

    Args:
        planes (_type_): _description_

    Returns:
        _type_: _description_
    
    """
    
    queue = deque()
    # queue.extend(sum(points, []))
    # 上下左右のどれかが欠落nodeまたは端ならば面から除去
    for plane in planes:
        nodes_to_pop = []
        for i in range(2):
            for node in plane.members:
                y, x = node.ind
                try:
                    if glob.nodes[y][x-1].g_num != node.g_num or glob.nodes[y][x+1].g_num != node.g_num or glob.nodes[y-1][x].g_num != node.g_num or glob.nodes[y+1][x].g_num != node.g_num:
                        if node not in nodes_to_pop:
                            nodes_to_pop.append(node)
                except IndexError:
                    if node not in nodes_to_pop:
                        nodes_to_pop.append(node)
                if i == 1:  #２回目の時はノードのポイント達をキューに追加
                    queue.extend(node.members)
                    pass

        for node in nodes_to_pop:
            plane.pop(node)

    print("build_boud_heap: end")
    log()

    return queue

def check_ref_points(point):

    """ポイントの上下左右を確認しノードの連結情報を定義し直す関数

    Args:
        point (_type_): _description_

    Note:
        point_position = np.argwhere((verts==point).all()) できてない

    """

    # 左右上下の順で取り出す
    for dir in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        try:
            adjacent = points[point.cord[0] + dir[0]][point.cord[1] + dir[1]]
            result = make_refine(point, adjacent)
            # if result:
            #     glob.pbar.update(1)
        except IndexError:
            pass

    print("check_ref_point: end")
    log()

def connect(node_a, node_b):
    
    """

    Args:
        node_a (_type_): _description_
        node_b (_type_): _description_
    
    """
    
    node_a.edges.add(node_b)
    node_b.edges.add(node_a)
    glob.refine_edges.add(node_a)
    glob.refine_edges.add(node_b)

def make_refine(point, adjacent):
    
    """2つの点が同じ面か違う面か判定し,結果によってそれぞれの処理を行う

    Args:
        point (_type_): _description_
        adjacent (_type_): _description_


    Returns:
        _type_: _description_
    
    """
    
    # 取り出したポイントと同一node(k)に属している
    if point.node == adjacent.node != None:
        return True

    # 同一のrefineポイントクラウド(k)に属している
    elif adjacent in point.node.refine_pc:
        return True

    # node(k)平面との距離がnode(k)平面のMSEの9倍以上か
    elif point.node.plane and calc_distance(point.node.plane, point) >= point.node.plane.mse*9 :
        return True

    # ポイントが他のrefineポイントクラウド(l)に属している場合
    if adjacent.refined:
        connect(point.node, adjacent.refined)

        if point.node.plane and adjacent.node.plane and calc_distance(point.node.plane, adjacent) <= calc_distance(adjacent.node.plane, adjacent):
            point.node.refine_pc.add(adjacent)
            adjacent.refined.refine_pc.remove(adjacent)
            glob.boud_queue.append(adjacent)
        else:
            return True

    # ポイントが他のrefineポイントクラウド(l)に属していない場合
    else:
        adjacent.refined = point.node
        point.node.refine_pc.add(adjacent)
        glob.boud_queue.append(adjacent)

    print("make_refine: end")
    log()       

def calc_distance(suf, point):

    """面と点の距離を計算する関数

    Args:
        suf (_type_): _description_
        point (_type_): _description_

    Returns:
        _type_: _description_
    
    """
    
    dist = (abs(suf.normal[0]*point.data[0] + suf.normal[1]*point.data[1] + suf.normal[2]*point.data[2] + suf.d)) / (math.sqrt(suf.normal[0]**2 + suf.normal[1]**2 + suf.normal[2]**2))
    
    print("calc_distance: end")
    log()
    
    return dist

if __name__ == '__main__':   
    if REAL_TIME: # カメラ映像
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
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            verts = np.asanyarray(pc.calculate(depth_frame).get_vertices()).view(np.float32).reshape(480, 640, 3)  # xyz
            points = []

            for h in range(len(verts)):
                line = []
                for w in range (len(verts[0])):
                    line.append(Point(verts[h,w], (h,w)))
                points.append(line)

            glob = Global()
            glob.detect()

            image = glob.visualize(True)

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', image)
            key = cv2.waitKey(1)
            prop_val = cv2.getWindowProperty('RealSense', cv2.WND_PROP_ASPECT_RATIO)

            if key == ord("s"):
                cv2.imwrite('./out.png', image)
            if key == ord("q") or (prop_val < 0):
                # Stop streaming
                pipeline.stop()
                cv2.destroyAllWindows()
                break

    else: # Matファイル読み込み
        data = scipy.io.loadmat('frame.mat')["frame"]
        verts = data[0, 0][0]
        color_image = data[0, 0][1]
        points = []

        for h in range(len(verts)):
            line = []
            for w in range(len(verts[0])):
                line.append(Point(verts[h,w], (h,w)))
            points.append(line)

        glob = Global()
        glob.detect()

        image = glob.visualize()

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', image)

        key = cv2.waitKey(0)
        if key == ord("s"):
                cv2.imwrite('./out_nr.png', image)
        cv2.destroyAllWindows()
