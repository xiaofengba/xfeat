#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud, ChannelFloat32
from geometry_msgs.msg import Point32
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import time
import message_filters

from modules.xfeat import XFeat

class XFeatVinsFrontEnd(Node):
    def __init__(self):
        super().__init__('xfeat_vins_frontend')

        # 1. 初始化 XFeat
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.xfeat = XFeat().to(self.device).eval()

        # 2. 状态维护
        self.prev_gray_l = None
        self.curr_pts_l = None 
        self.curr_ids = None    
        self.curr_ages = None   
        self.prev_un_pts_l = {} # 用于计算速度 {id: (u, v)}
        self.prev_stamp = None
        
        self.next_id = 0
        self.max_pts_count = 300
        self.min_pts_threshold = 150
        
        self.cam_info = {'left': None, 'right': None}
        self.is_calibrated = False

        # 3. ROS 工具
        self.bridge = CvBridge()
        self.sub_info_l = self.create_subscription(CameraInfo, '/cam0/image_raw_info', 
                                                 lambda msg: self.info_callback(msg, 'left'), 10)
        self.sub_info_r = self.create_subscription(CameraInfo, '/cam1/image_raw_info', 
                                                 lambda msg: self.info_callback(msg, 'right'), 10)

        self.sub_img_l = message_filters.Subscriber(self, Image, '/cam0/image_raw')
        self.sub_img_r = message_filters.Subscriber(self, Image, '/cam1/image_raw')
        self.sync = message_filters.ApproximateTimeSynchronizer([self.sub_img_l, self.sub_img_r], 10, 0.05)
        self.sync.registerCallback(self.stereo_callback)

        # 发布给 VINS-Fusion 的话题
        self.pub_feature = self.create_publisher(PointCloud, '/vins_feature', 10)
        self.pub_vis = self.create_publisher(Image, 'xfeat/vins_vis', 10)

        self.prev_time = time.time()
        self.fps = 0.0
        self.lk_params = dict(winSize=(21, 21), maxLevel=3, 
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def info_callback(self, msg, side):
        K = np.array(msg.k).reshape(3, 3)
        D = np.array(msg.d)
        R = np.array(msg.r).reshape(3, 3)
        P = np.array(msg.p).reshape(3, 4)
        model = "fisheye" if "equi" in msg.distortion_model.lower() or "kannala" in msg.distortion_model.lower() else "pinhole"
        self.cam_info[side] = {'K': K, 'D': D, 'R': R, 'P': P, 'model': model}
        if self.cam_info['left'] and self.cam_info['right']: self.is_calibrated = True

    def undistort_to_norm(self, pts, side):
        """转换到归一化平面 (u, v)，不应用投影矩阵 P，仅去畸变并校正 R"""
        info = self.cam_info[side]
        pts_reshaped = pts.reshape(-1, 1, 2).astype(np.float32)
        if info['model'] == "fisheye":
            # 注意：这里 P=None 返回归一化坐标
            undistorted = cv2.fisheye.undistortPoints(pts_reshaped, info['K'], info['D'], R=info['R'])
        else:
            undistorted = cv2.undistortPoints(pts_reshaped, info['K'], info['D'], R=info['R'])
        return undistorted.reshape(-1, 2)

    def stereo_callback(self, msg_l, msg_r):
        if not self.is_calibrated: return
        t_start = time.time()
        img_l = self.bridge.imgmsg_to_cv2(msg_l, 'bgr8')
        img_r = self.bridge.imgmsg_to_cv2(msg_r, 'bgr8')
        gray_l, gray_r = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        # 1. 左目追踪
        self.track_left_temporal(gray_l)
        # 2. XFeat 补点
        out_l, out_r = self.extract_features(img_l, img_r, gray_l)
        # 3. 跨目匹配
        pts_r_final, st_stereo = self.match_stereo_reinforced(gray_l, gray_r, out_l, out_r)
        # 4. 对极约束筛选
        st_stereo = self.reject_with_epipolar(self.curr_pts_l, pts_r_final, st_stereo)

        # 5. 构造 VINS PointCloud 消息
        self.publish_vins_feature(msg_l.header, pts_r_final, st_stereo)

        # 6. 状态更新
        self.prev_gray_l = gray_l
        self.prev_stamp = msg_l.header.stamp
        self.visualize(img_l, img_r, pts_r_final, st_stereo)

    def track_left_temporal(self, gray_l):
        h, w = gray_l.shape
        if self.prev_gray_l is not None and self.curr_pts_l is not None and len(self.curr_pts_l) > 0:
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray_l, gray_l, self.curr_pts_l.astype(np.float32), None, **self.lk_params)
            status = status.reshape(-1).astype(bool)
            valid = status & (new_pts[:,0]>5) & (new_pts[:,0]<w-5) & (new_pts[:,1]>5) & (new_pts[:,1]<h-5)
            self.curr_pts_l, self.curr_ids, self.curr_ages = new_pts[valid], self.curr_ids[valid], self.curr_ages[valid]+1
        else:
            self.curr_pts_l, self.curr_ids, self.curr_ages = np.empty((0, 2)), np.array([], dtype=int), np.array([], dtype=int)

    def extract_features(self, img_l, img_r, gray_l):
        h, w = gray_l.shape
        ts_l = torch.tensor(cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)).float().permute(2,0,1).unsqueeze(0).to(self.device)/255.0
        ts_r = torch.tensor(cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)).float().permute(2,0,1).unsqueeze(0).to(self.device)/255.0
        with torch.no_grad():
            out_l = self.xfeat.detectAndCompute(ts_l, top_k=self.max_pts_count)[0]
            out_r = self.xfeat.detectAndCompute(ts_r, top_k=self.max_pts_count)[0]
        # 如果当前帧的特征点比较少， 为已经存在的特征点构建一个mask， 计算需要新增的特征点， 保证新特征点位于图像内，且特征点所在的mask可用，然后添加该特征点
        if len(self.curr_pts_l) < self.max_pts_count:
            mask = np.ones((h, w), dtype=np.uint8)*255
            for pt in self.curr_pts_l: cv2.circle(mask, (int(pt[0]), int(pt[1])), 25, 0, -1)
            new_kpts = out_l['keypoints'].cpu().numpy()
            filtered = [p for p in new_kpts if 5<p[0]<w-5 and 5<p[1]<h-5 and mask[int(p[1]), int(p[0])]>0]
            if filtered:
                add_pts = np.array(filtered[:self.max_pts_count - len(self.curr_pts_l)])
                self.curr_pts_l = np.vstack([self.curr_pts_l, add_pts]) if len(self.curr_pts_l)>0 else add_pts
                self.curr_ids = np.concatenate([self.curr_ids, np.arange(self.next_id, self.next_id+len(add_pts))])
                self.curr_ages = np.concatenate([self.curr_ages, np.ones(len(add_pts), dtype=int)])
                self.next_id += len(add_pts)
        return out_l, out_r
    
    def match_stereo_reinforced(self, gray_l, gray_r, out_l, out_r):
        pts_r, st = np.zeros_like(self.curr_pts_l), np.zeros(len(self.curr_pts_l), dtype=bool)
        if len(self.curr_pts_l) > 0:
            pk, sk, _ = cv2.calcOpticalFlowPyrLK(gray_l, gray_r, self.curr_pts_l.astype(np.float32), None, **self.lk_params)
            with torch.no_grad():
                il, ir = self.xfeat.match(out_l['descriptors'], out_r['descriptors'], min_cossim=0.82)
            dm = {tuple(out_l['keypoints'][idx].cpu().numpy().round(1)): out_r['keypoints'][jdx].cpu().numpy() for idx, jdx in zip(il, ir)}
            for i in range(len(self.curr_pts_l)):
                lookup = tuple(self.curr_pts_l[i].round(1))
                if lookup in dm: pts_r[i], st[i] = dm[lookup], True
                elif sk.reshape(-1)[i]: pts_r[i], st[i] = pk[i], True
        return pts_r, st

    def reject_with_epipolar(self, pts_l, pts_r, st):
        if len(pts_l) == 0 or not any(st): return st
        idx = np.where(st)[0]
        # 使用 P=info['K'] 映射到带畸变的相机平面进行 dy 检查更准确，或直接在归一化平面检查
        rl = self.undistort_to_norm(pts_l[idx], 'left')
        rr = self.undistort_to_norm(pts_r[idx], 'right')
        dy = np.abs(rl[:, 1] - rr[:, 1])
        new_st = np.zeros(len(st), dtype=bool)
        new_st[idx[dy < 0.05]] = True # 归一化坐标系下的阈值通常很小
        return new_st

    def publish_vins_feature(self, header, pts_r, st_stereo):
        if len(self.curr_pts_l) == 0: return
        
        # 1. 坐标去畸变转换到归一化平面 (z=1)
        un_l = self.undistort_to_norm(self.curr_pts_l, 'left')
        un_r = self.undistort_to_norm(pts_r, 'right')
        
        # 2. 计算速度
        dt = 0.1 # 默认值
        if self.prev_stamp is not None:
            dt = (header.stamp.sec + header.stamp.nanosec*1e-9) - (self.prev_stamp.sec + self.prev_stamp.nanosec*1e-9)
        
        feat_msg = PointCloud()
        feat_msg.header = header
        
        # channels 定义
        c_id = ChannelFloat32(name="id", values=[])
        c_cam = ChannelFloat32(name="camera_id", values=[])
        c_u = ChannelFloat32(name="p_u", values=[])
        c_v = ChannelFloat32(name="p_v", values=[])
        c_vx = ChannelFloat32(name="velocity_x", values=[])
        c_vy = ChannelFloat32(name="velocity_y", values=[])

        new_prev_un_pts = {}
        
        for i in range(len(self.curr_pts_l)):
            fid = int(self.curr_ids[i])
            ux, uy = un_l[i][0], un_l[i][1]
            
            # 速度计算
            vx, vy = 0.0, 0.0
            if fid in self.prev_un_pts_l and dt > 0:
                vx = (ux - self.prev_un_pts_l[fid][0]) / dt
                vy = (uy - self.prev_un_pts_l[fid][1]) / dt
            new_prev_un_pts[fid] = (ux, uy)

            # 左目数据填充
            feat_msg.points.append(Point32(x=float(ux), y=float(uy), z=1.0))
            c_id.values.append(float(fid))
            c_cam.values.append(0.0)
            c_u.values.append(float(self.curr_pts_l[i][0]))
            c_v.values.append(float(self.curr_pts_l[i][1]))
            c_vx.values.append(float(vx))
            c_vy.values.append(float(vy))

            # 右目数据填充 (如果跨目匹配成功)
            if st_stereo[i]:
                feat_msg.points.append(Point32(x=float(un_r[i][0]), y=float(un_r[i][1]), z=1.0))
                c_id.values.append(float(fid))
                c_cam.values.append(1.0)
                c_u.values.append(float(pts_r[i][0]))
                c_v.values.append(float(pts_r[i][1]))
                c_vx.values.append(0.0) # 右目速度通常不强制要求，VINS主要用左目速度
                c_vy.values.append(0.0)

        self.prev_un_pts_l = new_prev_un_pts
        feat_msg.channels = [c_id, c_cam, c_u, c_v, c_vx, c_vy]
        self.pub_feature.publish(feat_msg)

    def visualize(self, img_l, img_r, pts_r, st_stereo):
        canvas = np.hstack([img_l, img_r])
        w_off = img_l.shape[1]
        for i in range(len(self.curr_pts_l)):
            xl, yl = int(self.curr_pts_l[i][0]), int(self.curr_pts_l[i][1])
            cv2.circle(canvas, (xl, yl), 3, (0, 255, 0), -1)
            if st_stereo[i]:
                xr, yr = int(pts_r[i][0] + w_off), int(pts_r[i][1])
                cv2.circle(canvas, (xr, yr), 3, (255, 0, 0), -1)
        self.pub_vis.publish(self.bridge.cv2_to_imgmsg(canvas, "bgr8"))

def main():
    rclpy.init(); node = XFeatVinsFrontEnd()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()