#!/usr/bin/env python3
import os
import sys
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

# ROS 消息类型
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2

# 工具库
import message_filters
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch

# 引入 XFeat
from modules.xfeat import XFeat

class XFeatStereoNode(Node):
    def __init__(self):
        super().__init__('xfeat_stereo_node')

        self.get_logger().info("初始化 XFeat Stereo (Raw模式 + 鱼眼矫正 + 极线约束)...")

        # ---------------------------------------------------------
        # 1. 初始化 XFeat 模型
        # ---------------------------------------------------------
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"XFeat 运行设备: {self.device}")
        
        self.xfeat = XFeat().to(self.device)
        self.xfeat.eval()

        # ---------------------------------------------------------
        # 2. ROS 工具 & 参数容器
        # ---------------------------------------------------------
        self.bridge = CvBridge()
        self.cam0_params = None
        self.cam1_params = None
        
        # ---------------------------------------------------------
        # 3. 创建发布者
        # ---------------------------------------------------------
        self.pub_matches = self.create_publisher(Image, 'xfeat/matches', 10)
        self.pub_cloud = self.create_publisher(PointCloud2, 'xfeat/point_cloud', 10)

        # ---------------------------------------------------------
        # 4. 创建订阅者 (CameraInfo)
        # ---------------------------------------------------------
        self.sub_info0 = self.create_subscription(
            CameraInfo, '/cam0/image_raw_info', self.info0_callback, 10)
        self.sub_info1 = self.create_subscription(
            CameraInfo, '/cam1/image_raw_info', self.info1_callback, 10)

        # ---------------------------------------------------------
        # 5. 创建订阅者 (Raw Image)
        # ---------------------------------------------------------
        sub_img0 = message_filters.Subscriber(self, Image, '/cam0/image_raw')
        sub_img1 = message_filters.Subscriber(self, Image, '/cam1/image_raw')

        queue_size = 10
        slop = 0.05 
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [sub_img0, sub_img1], queue_size, slop)
        
        self.sync.registerCallback(self.stereo_callback)

        self.get_logger().info("节点初始化完成，等待 CameraInfo 和 Raw Image...")

    def parse_camera_info(self, msg):
        return {
            'K': np.array(msg.k).reshape(3, 3).astype(np.float64),
            'D': np.array(msg.d).astype(np.float64),
            'R': np.array(msg.r).reshape(3, 3).astype(np.float64),
            'P': np.array(msg.p).reshape(3, 4).astype(np.float64),
            'model': msg.distortion_model,
            'header': msg.header
        }

    def info0_callback(self, msg):
        self.cam0_params = self.parse_camera_info(msg)
        self.destroy_subscription(self.sub_info0)
        self.get_logger().info(f"左相机内参已获取: {msg.distortion_model}")

    def info1_callback(self, msg):
        self.cam1_params = self.parse_camera_info(msg)
        self.destroy_subscription(self.sub_info1)
        self.get_logger().info(f"右相机内参已获取: {msg.distortion_model}")

    def preprocess_image(self, cv_img):
        if len(cv_img.shape) == 2: 
            img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        x = torch.tensor(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        return x.to(self.device)

    def undistort_points(self, keypoints, params):
        if len(keypoints) == 0:
            return keypoints

        pts_in = np.ascontiguousarray(keypoints[:, :2]).reshape(-1, 1, 2).astype(np.float64)
        
        K, D, R, P = params['K'], params['D'], params['R'], params['P']
        model = params['model']
        K_new = P[:3, :3]

        is_fisheye = ('fisheye' in model) or ('equidistant' in model)

        if is_fisheye:
            D_fish = D.flatten()[:4]
            try:
                pts_out = cv2.fisheye.undistortPoints(pts_in, K, D_fish, R=R, P=K_new)
            except cv2.error as e:
                self.get_logger().error(f"鱼眼矫正失败: {e}")
                return keypoints
        else:
            pts_out = cv2.undistortPoints(pts_in, K, D, R=R, P=K_new)
        
        return pts_out.reshape(-1, 2)

    def stereo_callback(self, msg_left, msg_right):
        if self.cam0_params is None or self.cam1_params is None:
            self.get_logger().warn("等待相机内参...", throttle_duration_sec=2.0)
            return

        t_start = time.time()

        try:
            # --- 阶段 1: 预处理 ---
            raw_img0 = self.bridge.imgmsg_to_cv2(msg_left, desired_encoding='bgr8')
            raw_img1 = self.bridge.imgmsg_to_cv2(msg_right, desired_encoding='bgr8')
            
            input_tensor0 = self.preprocess_image(raw_img0)
            input_tensor1 = self.preprocess_image(raw_img1)
            
            t_preprocess = time.time()

            # --- 阶段 2: 推理 ---
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            t_infer_start = time.time()
            
            with torch.no_grad():
                mkpts_0, mkpts_1 = self.xfeat.match_xfeat(input_tensor0, input_tensor1, top_k=2048)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()  
            t_infer_end = time.time()

            if len(mkpts_0) < 5:
                return

            if isinstance(mkpts_0, torch.Tensor):
                pts0_raw = mkpts_0.cpu().numpy()
                pts1_raw = mkpts_1.cpu().numpy()
            else:
                pts0_raw = mkpts_0
                pts1_raw = mkpts_1

            # --- 阶段 3: 点矫正 ---
            pts0_rect = self.undistort_points(pts0_raw, self.cam0_params)
            pts1_rect = self.undistort_points(pts1_raw, self.cam1_params)

            # =========================================================================
            # [新增] 极线约束过滤 (Epipolar Constraint)
            # =========================================================================
            # 在校正后的图像平面上，匹配点的 Y 坐标应该相同。
            # 设定一个阈值（例如 1.5 像素），如果 Y 偏差过大，认为是错误匹配。
            y_diff = np.abs(pts0_rect[:, 1] - pts1_rect[:, 1])
            epipolar_threshold = 5.5  # 像素阈值，可根据标定精度调整
            
            mask = y_diff < epipolar_threshold
            
            # 统计被剔除的点
            n_original = len(pts0_rect)
            n_kept = np.sum(mask)
            
            # 如果剩余点太少，跳过
            if n_kept < 5:
                self.get_logger().warn(f"极线约束剔除了几乎所有点! ({n_original} -> {n_kept})")
                return

            # 应用掩码过滤数据
            pts0_rect = pts0_rect[mask]
            pts1_rect = pts1_rect[mask]
            
            # 注意：同时也过滤 Raw 点，用于后续可视化展示正确的匹配
            pts0_raw = pts0_raw[mask]
            pts1_raw = pts1_raw[mask]
            # =========================================================================

            # --- 阶段 4: 三角化 (使用过滤后的校正点) ---
            points_4d = cv2.triangulatePoints(
                self.cam0_params['P'], 
                self.cam1_params['P'], 
                pts0_rect.T, pts1_rect.T
            )
            points_3d = points_4d[:3, :] / points_4d[3, :]
            points_3d = points_3d.T 

            # 深度过滤
            valid_mask = (points_3d[:, 2] > 0.1) & (points_3d[:, 2] < 50.0)
            valid_points = points_3d[valid_mask]

            t_end = time.time()

            # --- 发布数据 ---
            self.publish_pointcloud(valid_points, msg_left.header)
            
            # 发布可视化 (此时 pts0_raw 已经是经过极线约束筛选过的优质点了)
            self.publish_visual_matches(raw_img0, raw_img1, pts0_raw, pts1_raw)
            
            # --- 打印统计 ---
            dur_total = (t_end - t_start) * 1000
            
            # 计算过滤后的平均误差 (应该很小)
            y_diff_final = np.mean(np.abs(pts0_rect[:, 1] - pts1_rect[:, 1]))

            self.get_logger().info(
                f"耗时:{dur_total:.1f}ms | 匹配:{n_original}->{n_kept} | 极线误差:{y_diff_final:.2f}px"
            )

        except Exception as e:
            self.get_logger().error(f"处理出错: {e}")
            import traceback
            traceback.print_exc()

    def publish_pointcloud(self, points, header):
        if len(points) == 0:
            return
        pc2_msg = point_cloud2.create_cloud_xyz32(header, points)
        self.pub_cloud.publish(pc2_msg)

    def publish_visual_matches(self, img0, img1, pts0, pts1):
        kps0 = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=1) for p in pts0]
        kps1 = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=1) for p in pts1]
        matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(pts0))]
        vis_img = cv2.drawMatches(
            img0, kps0, img1, kps1, matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        # 可以在图上画上过滤信息
        cv2.putText(vis_img, f"Matches: {len(pts0)}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        vis_msg = self.bridge.cv2_to_imgmsg(vis_img, encoding="bgr8")
        self.pub_matches.publish(vis_msg)

def main(args=None):
    rclpy.init(args=args)
    node = XFeatStereoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()