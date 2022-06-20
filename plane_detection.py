# @author Timur
# @author Yasu

from contextlib import nullcontext
from curses import pair_content
from collections import deque
from email import message
from hashlib import new
from lib2to3.pygram import python_grammar_no_print_statement
import queue
from re import L, U
from ssl import VERIFY_X509_TRUSTED_FIRST
from tkinter import E, INSERT, W
from typing import final
from xml.sax.handler import property_declaration_handler
import graphlib
import math
from platform import java_ver, node
import time
from turtle import right
from typing_extensions import Self
import cv2
import numpy as np
import pyrealsense2 as rs
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error



class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)


state = AppState()

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Get stream profile and camera intrinsics
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

# Processing blocks
pc = rs.pointcloud()
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
colorizer = rs.colorizer()


def mouse_cb(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        state.mouse_btns[0] = True

    if event == cv2.EVENT_LBUTTONUP:
        state.mouse_btns[0] = False

    if event == cv2.EVENT_RBUTTONDOWN:
        state.mouse_btns[1] = True

    if event == cv2.EVENT_RBUTTONUP:
        state.mouse_btns[1] = False

    if event == cv2.EVENT_MBUTTONDOWN:
        state.mouse_btns[2] = True

    if event == cv2.EVENT_MBUTTONUP:
        state.mouse_btns[2] = False

    if event == cv2.EVENT_MOUSEMOVE:

        h, w = out.shape[:2]
        dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]

        if state.mouse_btns[0]:
            state.yaw += float(dx) / w * 2
            state.pitch -= float(dy) / h * 2

        elif state.mouse_btns[1]:
            dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
            state.translation -= np.dot(state.rotation, dp)

        elif state.mouse_btns[2]:
            dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
            state.translation[2] += dz
            state.distance -= dz

    if event == cv2.EVENT_MOUSEWHEEL:
        dz = math.copysign(0.1, flags)
        state.translation[2] += dz
        state.distance -= dz

    state.prev_mouse = (x, y)


cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow(state.WIN_NAME, w, h)
cv2.setMouseCallback(state.WIN_NAME, mouse_cb)


def project(v):
    """project 3d vector array to 2d"""
    h, w = out.shape[:2]
    view_aspect = float(h)/w

    # ignore divide by zero for invalid depth
    with np.errstate(divide='ignore', invalid='ignore'):
        proj = v[:, :-1] / v[:, -1, np.newaxis] * \
            (w*view_aspect, h) + (w/2.0, h/2.0)

    # near clipping
    znear = 0.03
    proj[v[:, 2] < znear] = np.nan
    return proj


def view(v):
    """apply view transformation on vector array"""
    return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation


def line3d(out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
    """draw a 3d line from pt1 to pt2"""
    p0 = project(pt1.reshape(-1, 3))[0]
    p1 = project(pt2.reshape(-1, 3))[0]
    if np.isnan(p0).any() or np.isnan(p1).any():
        return
    p0 = tuple(p0.astype(int))
    p1 = tuple(p1.astype(int))
    rect = (0, 0, out.shape[1], out.shape[0])
    inside, p0, p1 = cv2.clipLine(rect, p0, p1)
    if inside:
        cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)


def grid(out, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)):
    """draw a grid on xz plane"""
    pos = np.array(pos)
    s = size / float(n)
    s2 = 0.5 * size
    for i in range(0, n+1):
        x = -s2 + i*s
        line3d(out, view(pos + np.dot((x, 0, -s2), rotation)),
               view(pos + np.dot((x, 0, s2), rotation)), color)
    for i in range(0, n+1):
        z = -s2 + i*s
        line3d(out, view(pos + np.dot((-s2, 0, z), rotation)),
               view(pos + np.dot((s2, 0, z), rotation)), color)


def axes(out, pos, rotation=np.eye(3), size=0.075, thickness=2):
    """draw 3d axes"""
    line3d(out, pos, pos +
           np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
    line3d(out, pos, pos +
           np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
    line3d(out, pos, pos +
           np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)


def frustum(out, intrinsics, color=(0x40, 0x40, 0x40)):
    """draw camera's frustum"""
    orig = view([0, 0, 0])
    w, h = intrinsics.width, intrinsics.height

    for d in range(1, 6, 2):
        def get_point(x, y):
            p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
            line3d(out, orig, view(p), color)
            return p

        top_left = get_point(0, 0)
        top_right = get_point(w, 0)
        bottom_right = get_point(w, h)
        bottom_left = get_point(0, h)

        line3d(out, view(top_left), view(top_right), color)
        line3d(out, view(top_right), view(bottom_right), color)
        line3d(out, view(bottom_right), view(bottom_left), color)
        line3d(out, view(bottom_left), view(top_left), color)


def pointcloud(out, verts, texcoords, color, painter=True):
    """draw point cloud with optional painter's algorithm"""
    if painter:
        # Painter's algo, sort points from back to front

        # get reverse sorted indices by z (in view-space)
        # https://gist.github.com/stevenvo/e3dad127598842459b68
        v = view(verts)
        s = v[:, 2].argsort()[::-1]
        proj = project(v[s])
    else:
        proj = project(view(verts))

    if state.scale:
        proj *= 0.5**state.decimate

    h, w = out.shape[:2]

    # proj now contains 2d image coordinates
    j, i = proj.astype(np.uint32).T

    # create a mask to ignore out-of-bound indices
    im = (i >= 0) & (i < h)
    jm = (j >= 0) & (j < w)
    m = im & jm

    cw, ch = color.shape[:2][::-1]
    if painter:
        # sort texcoord with same indices as above
        # texcoords are [0..1] and relative to top-left pixel corner,
        # multiply by size and add 0.5 to center
        v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
    else:
        v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
    # clip texcoords to image
    np.clip(u, 0, ch-1, out=u)
    np.clip(v, 0, cw-1, out=v)

    # perform uv-mapping
    out[i[m], j[m]] = color[u[m], v[m]]


out = np.empty((h, w, 3), dtype=np.uint8)

'''
# 5/23まで
# nodeクラスの作成
class node:
    def __init__(self, points):
        self.node = points
        self.mse = None

    def make_mse(self):
        self.mse = 0

#nodeを作成
def make_grouping(points, x, y):
    
    return node

#nodeを除去
def dele_node(node):
    for i in range(10):
        for j in range(10):
            node[i][j] = 0

def remove_miss(node):
    #誤差の値
    mse_value = 100
    sur_value = 3
    #主成分分析
    for i in range(10):
        for j in range(10):
            if node[i][j] == None:
                dele_node(node)
                return


    node_pca = PCA(n_components=2).fit(node)
    #誤差の大きさが一定以上ならば
    if mean_squared_error(node_pca, node) >= mse_value:
        #除去する
        dele_node(node)
        return
    
    #上下左右の仮値
    up = 0
    down = 0
    right = 0
    left = 0
    center = 0

    #上下左右の値をとる
    for i in range(10):
        for j in range(10):
            center = node[i][j]
            try:
                up = node[i-1][j]
            except:
                up = center
            if abs(center - up) >= sur_value:
                dele_node(node)
                return
            try:
                down = node[i+1][j]
            except:
                down = center
            if abs(center - down) >= sur_value:
                dele_node(node)
                return
            try:
                right = node[i][j+1]
            except:
                right = center
            if abs(center - right) >= sur_value:
                dele_node(node)
                return
            try:
                left = node[i][j-1]
            except:
                left = center
            if abs(center - left) >= sur_value:
                dele_node(node)
                return
'''

# 5/30まで
# 処理の概形
# 入力は2次元になった
def fastplaneextraction(point_cloud):
    # データ構造構築(ステップ1)
    nodes, edges = initgraph(point_cloud)
    # 粗い平面検出(ステップ2)
    boudaries, pai = ahcluster(nodes, edges)
    # 粗い平面検出を精緻化(ステップ3)
    cluster, pro_pai = refine(boudaries, pai)
    return cluster, pro_pai

class Node:
    def __init__(self, node, index):
        self.node = node    # np.array型の点群の集合(10×10)
        self.i, self.j = index[0], index[1]  # 次元上のインデックス
        self.left = 0
        self.right = 0
        self.up = 0
        self.down = 0
        self.links = []
    
    # 上下左右をみる
    def look_araund(self, nodes, index, width):
        if index - 1 >= 0:
            self.left = nodes[index-1]
        if index + 1 <= len(nodes):
            self.right = nodes[index+1]
        if index - width >= 0:
            self.up = nodes[index-width]
        if index + width <= len(nodes):
            self.down = nodes[index+width]

    # 連結情報の登録
    def make_links(self, node_1, node_2):
        self.links.append(node_1)
        self.links.append(node_2)

# データ構造構築
def initgraph(point_cloud, h=10, w=10):
    nodes = []
    edges = []
    # 10×10が横にいくつあるかの数
    num = 8
    for i in range(len(point_cloud)/h):
        for j in range(len(point_cloud[0])/W):
            # node は論文の v
            node = Node(point_cloud[i*h:i*h+h-1,j*w:j*w+w-1], (i,j))
            # nodeの除去の判定
            if rejectnode(node.node):
                node = Node(None, (i,j))
            nodes = nodes.append(node)
    # 連結関係
    for i in range(len(nodes)):
        if (i == 0) or (i == len(nodes)-1):
            continue
        if not rejectedge(nodes[i-1].node, nodes[i].node, nodes[i+1].node):
            # 追加形式が謎
            edges = edges.append([nodes[i-1], nodes[i], nodes[i+1]])
        if (i-num < 0) or (i+num > len(nodes)-1):
            continue
        if not rejectedge(nodes[i-num].node, nodes[i].nodes, nodes[i+num].node):
            edges = edges.append([nodes[i-num], nodes[i], nodes[i+num]])
    return nodes, edges

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
    elif np.cross(node1, node3) > 999999:
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
        v = popmin(queue)
        # vがマージされているならば
        if v not in nodes:
            continue
        u_best = np.array()
        u_merge = np.array()
        # vと連結関係にあるuを取り出して
        for u in edges:
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

    # C++を参考にしながら



while True:
    # Grab camera data
    if not state.paused:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_frame = decimate.process(depth_frame)

        # Grab new intrinsics (may be changed by decimation)
        depth_intrinsics = rs.video_stream_profile(
            depth_frame.profile).get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_colormap = np.asanyarray(
            colorizer.colorize(depth_frame).get_data())

        if state.color:
            mapped_frame, color_source = color_frame, color_image
        else:
            mapped_frame, color_source = depth_frame, depth_colormap

        points = pc.calculate(depth_frame)
        pc.map_to(mapped_frame)

        # Pointcloud data to arrays
        v, t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
        texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

        # vertsが引数(中身をみてみる)
        cluster, pai =  fastplaneextraction(verts)
        
    # 3dを描画しないでcluster,paiを描画する

    # Render
    now = time.time()

    out.fill(0)

    grid(out, (0, 0.5, 1), size=1, n=10)
    frustum(out, depth_intrinsics)
    axes(out, view([0, 0, 0]), state.rotation, size=0.1, thickness=1)

    if not state.scale or out.shape[:2] == (h, w):
        pointcloud(out, verts, texcoords, color_source)
    else:
        tmp = np.zeros((h, w, 3), dtype=np.uint8)
        pointcloud(tmp, verts, texcoords, color_source)
        tmp = cv2.resize(
            tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        np.putmask(out, tmp > 0, tmp)

    if any(state.mouse_btns):
        axes(out, view(state.pivot), state.rotation, thickness=4)

    dt = time.time() - now

    cv2.setWindowTitle(
        state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
        (w, h, 1.0/dt, dt*1000, "PAUSED" if state.paused else ""))

    cv2.imshow(state.WIN_NAME, out)
    key = cv2.waitKey(1)

    if key == ord("r"):
        state.reset()

    if key == ord("p"):
        state.paused ^= True

    if key == ord("d"):
        state.decimate = (state.decimate + 1) % 3
        decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)

    if key == ord("z"):
        state.scale ^= True

    if key == ord("c"):
        state.color ^= True

    if key == ord("s"):
        cv2.imwrite('./out.png', out)

    if key == ord("e"):
        points.export_to_ply('./out.ply', mapped_frame)

    if key in (27, ord("q")) or cv2.getWindowProperty(state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
        break
# Stop streaming
pipeline.stop()