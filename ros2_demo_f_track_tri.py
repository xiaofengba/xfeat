#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import time
import message_filters
from sensor_msgs_py import point_cloud2

from modules.xfeat import XFeat

class XFeatStereoTrackerNode(Node):
    def __init__(self):
        super().__init__('xfeat_stereo_tracker_node')

        # 1. 初始化 XFeat
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.xfeat = XFeat().to(self.device)
        self.xfeat.eval()

        # 2. 状态与相机内参维护
        self.prev_gray_l = None
        self.curr_pts_l = None 
        self.curr_ids = None    
        self.curr_ages = None   
        self.next_id = 0
        self.max_pts_count = 300
        self.min_pts_threshold = 150
        
        self.cam_info = {'left': None, 'right': None}
        self.is_calibrated = False

        # 3. ROS 工具
        self.bridge = CvBridge()
        
        # 订阅 CameraInfo
        self.sub_info_l = self.create_subscription(CameraInfo, '/cam0/image_raw_info', 
                                                 lambda msg: self.info_callback(msg, 'left'), 10)
        self.sub_info_r = self.create_subscription(CameraInfo, '/cam1/image_raw_info', 
                                                 lambda msg: self.info_callback(msg, 'right'), 10)

        # 同步订阅 raw 图像
        self.sub_img_l = message_filters.Subscriber(self, Image, '/cam0/image_raw')
        self.sub_img_r = message_filters.Subscriber(self, Image, '/cam1/image_raw')
        self.sync = message_filters.ApproximateTimeSynchronizer([self.sub_img_l, self.sub_img_r], 10, 0.05)
        self.sync.registerCallback(self.stereo_callback)

        self.pub_vis = self.create_publisher(Image, 'xfeat/stereo_tracking_vis', 10)
        self.pub_cloud = self.create_publisher(PointCloud2, 'xfeat/point_cloud', 10)

        self.prev_time = time.time()
        self.fps = 0.0
        self.lk_params = dict(winSize=(21, 21), maxLevel=3, 
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def info_callback(self, msg, side):
        """解析相机参数：支持 pinhole 和 fisheye"""
        K = np.array(msg.k).reshape(3, 3)
        D = np.array(msg.d)
        R = np.array(msg.r).reshape(3, 3)
        P = np.array(msg.p).reshape(3, 4)
        model = "fisheye" if "equi" in msg.distortion_model.lower() or "kannala" in msg.distortion_model.lower() else "pinhole"
        
        self.cam_info[side] = {'K': K, 'D': D, 'R': R, 'P': P, 'model': model, 'header': msg.header}
        if self.cam_info['left'] is not None and self.cam_info['right'] is not None:
            self.is_calibrated = True

    def preprocess_tensor(self, cv_img):
        """关键函数修复：将图像转换为 Tensor"""
        img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        x = torch.tensor(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        return x.to(self.device)

    def undistort_points(self, pts, side):
        """仅对点进行去畸变并校正 (Rectified)"""
        info = self.cam_info[side]
        pts_reshaped = pts.reshape(-1, 1, 2).astype(np.float32)
        if info['model'] == "fisheye":
            undistorted = cv2.fisheye.undistortPoints(pts_reshaped, info['K'], info['D'], R=info['R'], P=info['P'])
        else:
            undistorted = cv2.undistortPoints(pts_reshaped, info['K'], info['D'], R=info['R'], P=info['P'])
        return undistorted.reshape(-1, 2)

    def stereo_callback(self, msg_l, msg_r):
        if not self.is_calibrated: return

        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now
        if dt > 0: self.fps = self.fps * 0.9 + (1.0 / dt) * 0.1

        img_l = self.bridge.imgmsg_to_cv2(msg_l, 'bgr8')
        img_r = self.bridge.imgmsg_to_cv2(msg_r, 'bgr8')
        gray_l, gray_r = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        # 1. 左目时序追踪
        self.track_left_temporal(gray_l)

        # 2. 特征补充与提取
        out_l, out_r = self.extract_stereo_features(img_l, img_r, gray_l)

        # 3. 跨目强化匹配
        pts_r_final, st_stereo = self.match_stereo_reinforced(gray_l, gray_r, out_l, out_r)

        # 4. 去畸变并在校正平面进行对极约束筛选
        st_stereo = self.reject_with_epipolar_constraint(self.curr_pts_l, pts_r_final, st_stereo)

        # 5. 三角化生成 3D 点云
        self.triangulate_and_publish(self.curr_pts_l, pts_r_final, st_stereo, msg_l.header)

        # 6. 可视化
        self.prev_gray_l = gray_l
        self.visualize_stereo(img_l, img_r, self.curr_pts_l, pts_r_final, st_stereo, self.curr_ids, self.curr_ages)

    def track_left_temporal(self, gray_l):
        h, w = gray_l.shape
        if self.prev_gray_l is not None and self.curr_pts_l is not None and len(self.curr_pts_l) > 0:
            new_pts_l, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray_l, gray_l, self.curr_pts_l.astype(np.float32), None, **self.lk_params)
            status = status.reshape(-1).astype(bool)
            valid = status & (new_pts_l[:,0]>5) & (new_pts_l[:,0]<w-5) & (new_pts_l[:,1]>5) & (new_pts_l[:,1]<h-5)
            self.curr_pts_l, self.curr_ids, self.curr_ages = new_pts_l[valid], self.curr_ids[valid], self.curr_ages[valid]+1
        else:
            self.curr_pts_l, self.curr_ids, self.curr_ages = np.empty((0, 2)), np.array([], dtype=int), np.array([], dtype=int)

    def extract_stereo_features(self, img_l, img_r, gray_l):
        h, w = gray_l.shape
        input_l, input_r = self.preprocess_tensor(img_l), self.preprocess_tensor(img_r)
        with torch.no_grad():
            out_l = self.xfeat.detectAndCompute(input_l, top_k=self.max_pts_count)[0]
            out_r = self.xfeat.detectAndCompute(input_r, top_k=self.max_pts_count)[0]
        if len(self.curr_pts_l) < self.max_pts_count:
            mask = np.ones((h, w), dtype=np.uint8) * 255
            for pt in self.curr_pts_l: cv2.circle(mask, (int(pt[0]), int(pt[1])), 25, 0, -1)
            new_kpts = out_l['keypoints'].cpu().numpy()
            filtered = [p for p in new_kpts if 5<p[0]<w-5 and 5<p[1]<h-5 and mask[int(p[1]), int(p[0])]>0]
            if len(filtered) > 0:
                add_pts = np.array(filtered[:self.max_pts_count - len(self.curr_pts_l)])
                self.curr_pts_l = np.vstack([self.curr_pts_l, add_pts]) if len(self.curr_pts_l)>0 else add_pts
                self.curr_ids = np.concatenate([self.curr_ids, np.arange(self.next_id, self.next_id+len(add_pts))])
                self.curr_ages = np.concatenate([self.curr_ages, np.ones(len(add_pts), dtype=int)])
                self.next_id += len(add_pts)
        return out_l, out_r

    def match_stereo_reinforced(self, gray_l, gray_r, out_l, out_r):
        pts_r_final, st_stereo = np.zeros_like(self.curr_pts_l), np.zeros(len(self.curr_pts_l), dtype=bool)
        if len(self.curr_pts_l) > 0:
            pts_r_klt, st_klt, _ = cv2.calcOpticalFlowPyrLK(gray_l, gray_r, self.curr_pts_l.astype(np.float32), None, **self.lk_params)
            with torch.no_grad():
                idx_l, idx_r = self.xfeat.match(out_l['descriptors'], out_r['descriptors'], min_cossim=0.82)
            desc_matches = {tuple(out_l['keypoints'][il].cpu().numpy().round(1)): out_r['keypoints'][ir].cpu().numpy() 
                            for il, ir in zip(idx_l.cpu().numpy(), idx_r.cpu().numpy())}
            for i in range(len(self.curr_pts_l)):
                lookup = tuple(self.curr_pts_l[i].round(1))
                if lookup in desc_matches:
                    pts_r_final[i], st_stereo[i] = desc_matches[lookup], True
                elif st_klt.reshape(-1)[i]:
                    pts_r_final[i], st_stereo[i] = pts_r_klt[i], True
        return pts_r_final, st_stereo

    def reject_with_epipolar_constraint(self, pts_l, pts_r, status, threshold=5.5):
        if len(pts_l) == 0 or not any(status): return status
        idx = np.where(status)[0]
        rect_pts_l = self.undistort_points(pts_l[idx], 'left')
        rect_pts_r = self.undistort_points(pts_r[idx], 'right')
        dy = np.abs(rect_pts_l[:, 1] - rect_pts_r[:, 1])
        new_status = np.zeros(len(status), dtype=bool)
        new_status[idx[dy < threshold]] = True
        return new_status

    def triangulate_and_publish(self, pts_l, pts_r, status, header):
        """执行三角化并发布点云"""
        if len(pts_l) == 0 or not any(status): return
        idx = np.where(status)[0]
        # 获取去畸变且校正（Rectified）后的点
        rect_pts_l = self.undistort_points(pts_l[idx], 'left')
        rect_pts_r = self.undistort_points(pts_r[idx], 'right')
        
        # 使用投影矩阵 P 执行三角化
        # 结果在左相机校正后的坐标系下
        pts4d = cv2.triangulatePoints(self.cam_info['left']['P'], self.cam_info['right']['P'], 
                                      rect_pts_l.T, rect_pts_r.T)
        pts3d = (pts4d[:3, :] / pts4d[3, :]).T
        
        # 过滤距离太近或太远的无效深度
        valid = (pts3d[:, 2] > 0.1) & (pts3d[:, 2] < 20.0)
        if any(valid):
            pc2_msg = point_cloud2.create_cloud_xyz32(header, pts3d[valid].astype(np.float32))
            self.pub_cloud.publish(pc2_msg)

    def visualize_stereo(self, img_l, img_r, pts_l, pts_r, st_stereo, ids, ages):
        canvas = np.hstack([img_l, img_r]); w_off = img_l.shape[1]
        for i in range(len(pts_l)):
            xl, yl = int(round(pts_l[i][0])), int(round(pts_l[i][1]))
            age = int(ages[i]); color = (0, int(min(255, age*10)), int(max(0, 255-age*10)))
            cv2.circle(canvas, (xl, yl), 3, color, -1)
            if age > 5: cv2.putText(canvas, str(int(ids[i])), (xl+4, yl-4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
            if i < len(st_stereo) and st_stereo[i]:
                xr, yr = int(round(pts_r[i][0] + w_off)), int(round(pts_r[i][1]))
                cv2.circle(canvas, (xr, yr), 3, color, -1)
                if age > 5: cv2.putText(canvas, str(int(ids[i])), (xr+4, yr-4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
        cv2.putText(canvas, f"FPS: {self.fps:.1f} Pts: {len(pts_l)}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        self.pub_vis.publish(self.bridge.cv2_to_imgmsg(canvas, "bgr8"))

def main():
    rclpy.init(); node = XFeatStereoTrackerNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()