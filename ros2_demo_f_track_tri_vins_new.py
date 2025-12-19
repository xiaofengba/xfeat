#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud, ChannelFloat32
from geometry_msgs.msg import Point32
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import message_filters

from modules.xfeat import XFeat

class XFeatVinsFrontEnd(Node):
    def __init__(self):
        super().__init__('xfeat_vins_frontend')

        # 1. 初始化 XFeat (只用于提取关键点，不需要描述子匹配)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.xfeat = XFeat().to(self.device).eval()

        # 2. 状态维护
        self.prev_gray_l = None
        self.curr_pts_l = np.empty((0, 2)) 
        self.curr_ids = np.array([], dtype=int)   
        self.curr_ages = np.array([], dtype=int)   
        self.prev_un_pts_l = {} # {id: (u, v)}
        self.prev_stamp = None
        
        self.next_id = 0
        self.max_pts_count = 200  # VINS 通常 150-200 就够了
        self.min_pts_threshold = 120 # 低于此值才补点
        
        self.cam_info = {'left': None, 'right': None}
        self.is_calibrated = False

        # 3. ROS 通信
        self.bridge = CvBridge()
        
        # 使用 QoS profile 保证通信顺畅
        qos = rclpy.qos.QoSProfile(depth=10)
        
        self.sub_info_l = self.create_subscription(CameraInfo, '/cam0/image_raw_info', 
                                                 lambda msg: self.info_callback(msg, 'left'), qos)
        self.sub_info_r = self.create_subscription(CameraInfo, '/cam1/image_raw_info', 
                                                 lambda msg: self.info_callback(msg, 'right'), qos)

        self.sub_img_l = message_filters.Subscriber(self, Image, '/cam0/image_raw')
        self.sub_img_r = message_filters.Subscriber(self, Image, '/cam1/image_raw')
        self.sync = message_filters.ApproximateTimeSynchronizer([self.sub_img_l, self.sub_img_r], 10, 0.05)
        self.sync.registerCallback(self.stereo_callback)

        self.pub_feature = self.create_publisher(PointCloud, '/vins_feature', 10)
        self.pub_vis = self.create_publisher(Image, 'xfeat/vins_vis', 10)

        # LK 参数：对于 Fisheye 这种大畸变，winSize 稍微大一点更好
        self.lk_params = dict(winSize=(21, 21), maxLevel=3, 
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))

    def info_callback(self, msg, side):
        if self.cam_info[side] is not None: return
        K = np.array(msg.k).reshape(3, 3)
        D = np.array(msg.d)
        # 注意：如果发布的是去畸变后的图，R 通常是单位阵；如果是原始图，R 是极线矫正旋转矩阵
        R = np.array(msg.r).reshape(3, 3) 
        P = np.array(msg.p).reshape(3, 4)
        
        # 简单判断畸变模型
        model = "fisheye" if "equi" in msg.distortion_model.lower() or "kannala" in msg.distortion_model.lower() else "pinhole"
        self.cam_info[side] = {'K': K, 'D': D, 'R': R, 'P': P, 'model': model}
        if self.cam_info['left'] and self.cam_info['right']: 
            self.get_logger().info("Stereo Camera Calibrated!")
            self.is_calibrated = True

    def undistort_to_norm(self, pts, side):
        """像素坐标 -> 归一化平面坐标 (z=1)"""
        if len(pts) == 0: return pts
        info = self.cam_info[side]
        pts_reshaped = pts.reshape(-1, 1, 2).astype(np.float32)
        
        if info['model'] == "fisheye":
            # P=None 或者是 np.eye(3) 都可以让 fisheye.undistortPoints 返回归一化坐标
            undistorted = cv2.fisheye.undistortPoints(pts_reshaped, info['K'], info['D'], R=info['R'], P=None)
        else:
            undistorted = cv2.undistortPoints(pts_reshaped, info['K'], info['D'], R=info['R'], P=None)
        return undistorted.reshape(-1, 2)

    def stereo_callback(self, msg_l, msg_r):
        if not self.is_calibrated: return
        
        img_l = self.bridge.imgmsg_to_cv2(msg_l, 'mono8') # 直接读灰度，节省转换时间
        img_r = self.bridge.imgmsg_to_cv2(msg_r, 'mono8')
        
        # 1. 左目帧间追踪 (Temporal Tracking)
        self.track_left_temporal(img_l)
        
        # 2. XFeat 补点 (只在点少时运行，且只在左目)
        self.detect_new_features(img_l)
        
        # 3. 双目匹配 (Stereo Tracking - 使用光流更稳)
        pts_r_final, st_stereo = self.track_stereo_lk(img_l, img_r)
        
        # 4. 发布消息
        self.publish_vins_feature(msg_l.header, pts_r_final, st_stereo)

        # 5. 更新状态
        self.prev_gray_l = img_l.copy()
        self.prev_stamp = msg_l.header.stamp
        
        # 6. 可视化
        if self.pub_vis.get_subscription_count() > 0:
            self.visualize(img_l, img_r, pts_r_final, st_stereo)

    def track_left_temporal(self, gray_l):
        """使用 LK 光流追踪上一帧特征到当前帧"""
        if self.prev_gray_l is not None and len(self.curr_pts_l) > 0:
            # 前向光流
            pts_next, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray_l, gray_l, self.curr_pts_l.astype(np.float32), None, **self.lk_params)
            
            # 反向光流检查 (可选，增强鲁棒性)
            pts_back, _, _ = cv2.calcOpticalFlowPyrLK(gray_l, self.prev_gray_l, pts_next, None, **self.lk_params)
            dist = np.sum((pts_back - self.curr_pts_l)**2, axis=1)
            status_back = dist < 0.5 # 0.5 像素误差以内
            
            # 越界检查
            h, w = gray_l.shape
            in_bound = (pts_next[:,0]>=0) & (pts_next[:,0]<w) & (pts_next[:,1]>=0) & (pts_next[:,1]<h)
            
            valid = status.reshape(-1).astype(bool) & status_back & in_bound
            
            # 减少 id 列表
            self.curr_pts_l = pts_next[valid]
            self.curr_ids = self.curr_ids[valid]
            self.curr_ages = self.curr_ages[valid] + 1
        else:
            self.curr_pts_l = np.empty((0, 2))
            self.curr_ids = np.array([], dtype=int)
            self.curr_ages = np.array([], dtype=int)

    def detect_new_features(self, gray_l):
        """使用 XFeat 提取新特征"""
        current_cnt = len(self.curr_pts_l)
        if current_cnt >= self.min_pts_threshold:
            return

        n_new = self.max_pts_count - current_cnt
        h, w = gray_l.shape
        
        # 构建 Mask，避免在已有特征点附近提取
        mask = np.ones((h, w), dtype=np.uint8) * 255
        for pt in self.curr_pts_l:
            cv2.circle(mask, (int(pt[0]), int(pt[1])), 15, 0, -1) # 半径15像素抑制

        # 准备数据给 XFeat
        img_tensor = torch.tensor(cv2.cvtColor(gray_l, cv2.COLOR_GRAY2RGB)).float().permute(2,0,1).unsqueeze(0).to(self.device)/255.0
        
        with torch.no_grad():
            # top_k 可以设置大一点，然后通过 mask 筛选
            out = self.xfeat.detectAndCompute(img_tensor, top_k=self.max_pts_count*2)[0]
            
        new_kpts = out['keypoints'].cpu().numpy()
        
        # 筛选
        added_pts = []
        for p in new_kpts:
            if len(added_pts) >= n_new: break
            ix, iy = int(p[0]), int(p[1])
            if 0 <= ix < w and 0 <= iy < h and mask[iy, ix] > 0:
                added_pts.append(p)
                cv2.circle(mask, (ix, iy), 15, 0, -1) # 动态更新 mask
        
        if added_pts:
            added_pts = np.array(added_pts)
            self.curr_pts_l = np.vstack([self.curr_pts_l, added_pts]) if len(self.curr_pts_l) > 0 else added_pts
            self.curr_ids = np.concatenate([self.curr_ids, np.arange(self.next_id, self.next_id + len(added_pts))])
            self.curr_ages = np.concatenate([self.curr_ages, np.zeros(len(added_pts), dtype=int)])
            self.next_id += len(added_pts)

    def track_stereo_lk(self, gray_l, gray_r):
        """双目匹配：使用光流从左图追踪到右图"""
        if len(self.curr_pts_l) == 0:
            return np.empty((0, 2)), np.zeros(0, dtype=bool)

        pts_r, status, _ = cv2.calcOpticalFlowPyrLK(gray_l, gray_r, self.curr_pts_l.astype(np.float32), None, **self.lk_params)
        status = status.reshape(-1).astype(bool)
        
        # 极线约束剔除 (Epipolar Check)
        # 如果相机做了极线矫正，特征点应该在同一行 (y 坐标相近)
        if self.is_calibrated:
            # 转换到归一化平面检查更准确
            pts_l_norm = self.undistort_to_norm(self.curr_pts_l, 'left')
            pts_r_norm = self.undistort_to_norm(pts_r, 'right')
            
            # 阈值：归一化平面下的 y 差值，0.02 大约对应图像上的 10-20 像素(取决于焦距)
            # 如果你的图像已经做过 undistort+rectify，可以直接比较 pixel y
            dy = np.abs(pts_l_norm[:, 1] - pts_r_norm[:, 1])
            valid_epi = dy < 0.1 # 稍微放宽一点，防止误杀
            
            # 双向光流检查右目 (可选，耗时)
            # pts_l_back, _, _ = cv2.calcOpticalFlowPyrLK(gray_r, gray_l, pts_r, None, **self.lk_params)
            # dist_stereo = np.sum((pts_l_back - self.curr_pts_l)**2, axis=1)
            # valid_fb = dist_stereo < 1.0
            
            status = status & valid_epi

        return pts_r, status

    def publish_vins_feature(self, header, pts_r, st_stereo):
        if len(self.curr_pts_l) == 0: return
        
        un_l = self.undistort_to_norm(self.curr_pts_l, 'left')
        un_r = self.undistort_to_norm(pts_r, 'right')
        
        dt = 0.0
        if self.prev_stamp is not None:
            dt = (header.stamp.sec - self.prev_stamp.sec) + (header.stamp.nanosec - self.prev_stamp.nanosec)*1e-9
        
        feat_msg = PointCloud()
        feat_msg.header = header
        
        ids, u, v, vx, vy = [], [], [], [], []
        # 构建通道... (省略部分重复代码，逻辑与你原代码一致，只是简化写法)
        # 重点：确保 channels 名字和 VINS 接收端一致
        
        new_prev_un = {}
        
        points = []
        c_id_vals = []
        c_u_vals = []
        c_v_vals = []
        c_vx_vals = []
        c_vy_vals = []
        
        for i in range(len(self.curr_pts_l)):
            fid = int(self.curr_ids[i])
            ux, uy = un_l[i]
            
            vel_x, vel_y = 0.0, 0.0
            if fid in self.prev_un_pts_l and dt > 0.001:
                p_prev = self.prev_un_pts_l[fid]
                vel_x = (ux - p_prev[0]) / dt
                vel_y = (uy - p_prev[1]) / dt
            
            new_prev_un[fid] = (ux, uy)
            
            # Left point info
            # VINS 要求 Point32(x, y, z) 也是归一化坐标
            points.append(Point32(x=float(ux), y=float(uy), z=1.0))
            c_id_vals.append(float(fid))
            c_u_vals.append(float(self.curr_pts_l[i][0])) # 像素坐标
            c_v_vals.append(float(self.curr_pts_l[i][1]))
            c_vx_vals.append(float(vel_x))
            c_vy_vals.append(float(vel_y))

            # Right point info (same ID, velocity=0)
            if st_stereo[i]:
                points.append(Point32(x=float(un_r[i][0]), y=float(un_r[i][1]), z=1.0))
                c_id_vals.append(float(fid))
                c_u_vals.append(float(pts_r[i][0]))
                c_v_vals.append(float(pts_r[i][1]))
                c_vx_vals.append(0.0)
                c_vy_vals.append(0.0)

        self.prev_un_pts_l = new_prev_un
        
        feat_msg.points = points
        feat_msg.channels = [
            ChannelFloat32(name="id", values=c_id_vals),
            ChannelFloat32(name="u", values=c_u_vals), # 像素u
            ChannelFloat32(name="v", values=c_v_vals), # 像素v
            ChannelFloat32(name="velocity_x", values=c_vx_vals),
            ChannelFloat32(name="velocity_y", values=c_vy_vals)
        ]
        self.pub_feature.publish(feat_msg)

    def visualize(self, img_l, img_r, pts_r, st_stereo):
        # 简单可视化，转为彩色以便画图
        vis = cv2.cvtColor(img_l, cv2.COLOR_GRAY2BGR)
        for i in range(len(self.curr_pts_l)):
            pt = self.curr_pts_l[i]
            # 追踪次数越多颜色越深
            color = (0, 255, 0) if st_stereo[i] else (0, 0, 255)
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, color, -1)
            # 如果匹配成功，画线
            if st_stereo[i]:
                pt_r = pts_r[i]
                cv2.line(vis, (int(pt[0]), int(pt[1])), (int(pt_r[0]), int(pt_r[1])), (0, 255, 255), 1)
        
        self.pub_vis.publish(self.bridge.cv2_to_imgmsg(vis, "bgr8"))

def main():
    rclpy.init(); node = XFeatVinsFrontEnd()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()