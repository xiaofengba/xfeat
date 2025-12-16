#!/usr/bin/env python3
import os
import sys
import time  # 引入 time 模块
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

# ROS 消息类型
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Header

# 工具库
import message_filters
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from sensor_msgs_py import point_cloud2

# 引入 XFeat
from modules.xfeat import XFeat

class XFeatStereoNode(Node):
    def __init__(self):
        super().__init__('xfeat_stereo_node')

        self.get_logger().info("正在初始化 XFeat Stereo 节点...")

        # ---------------------------------------------------------
        # 1. 初始化 XFeat 模型 (GPU/CPU 配置)
        # ---------------------------------------------------------
        # 检查是否使用 GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"XFeat 运行设备: {self.device}")
        
        # 初始化模型并移动到指定设备 (关键修改)
        # 只有将模型 .to(device)，它的权重才会加载显存中
        self.xfeat = XFeat().to(self.device)
        self.xfeat.eval() # 设置为评估模式 (通常会关闭 dropout 等，虽然 XFeat 可能影响不大，但是个好习惯)

        # ---------------------------------------------------------
        # 2. ROS 工具初始化
        # ---------------------------------------------------------
        self.bridge = CvBridge()
        self.P0 = None 
        self.P1 = None 
        self.frame_id = "cam0_optical_frame"

        # ---------------------------------------------------------
        # 3. 创建发布者
        # ---------------------------------------------------------
        self.pub_matches = self.create_publisher(Image, 'xfeat/matches', 10)
        self.pub_cloud = self.create_publisher(PointCloud2, 'xfeat/point_cloud', 10)

        # ---------------------------------------------------------
        # 4. 创建订阅者
        # ---------------------------------------------------------
        self.sub_info0 = self.create_subscription(
            CameraInfo, '/cam0/image_raw_info', self.info0_callback, 10)
        self.sub_info1 = self.create_subscription(
            CameraInfo, '/cam1/image_raw_info', self.info1_callback, 10)

        sub_img0 = message_filters.Subscriber(self, Image, '/cam0/image_rect')
        sub_img1 = message_filters.Subscriber(self, Image, '/cam1/image_rect')

        queue_size = 10
        slop = 0.05 
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [sub_img0, sub_img1], queue_size, slop)
        
        self.sync.registerCallback(self.stereo_callback)

        self.get_logger().info("节点初始化完成，等待图像话题...")

    def info0_callback(self, msg):
        self.P0 = np.array(msg.p).reshape(3, 4)
        self.frame_id = msg.header.frame_id
        self.destroy_subscription(self.sub_info0)
        self.get_logger().info("已获取左相机内参")

    def info1_callback(self, msg):
        self.P1 = np.array(msg.p).reshape(3, 4)
        self.destroy_subscription(self.sub_info1)
        self.get_logger().info("已获取右相机内参")

    def preprocess_image(self, cv_img):
        """将 OpenCV 图像转换为 XFeat 需要的 Tensor 格式"""
        if len(cv_img.shape) == 2: 
            img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        x = torch.tensor(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        # 这里的 .to(device) 很重要，确保输入数据也在 GPU 上
        return x.to(self.device)

    def stereo_callback(self, msg_left, msg_right):
        if self.P0 is None or self.P1 is None:
            self.get_logger().warn("等待相机内参...", throttle_duration_sec=2.0)
            return

        # 记录总开始时间
        t_start = time.time()

        try:
            # --- 阶段 1: 预处理 ---
            cv_img0 = self.bridge.imgmsg_to_cv2(msg_left, desired_encoding='bgr8')
            cv_img1 = self.bridge.imgmsg_to_cv2(msg_right, desired_encoding='bgr8')
            
            input_tensor0 = self.preprocess_image(cv_img0)
            input_tensor1 = self.preprocess_image(cv_img1)
            
            t_preprocess = time.time()

            # --- 阶段 2: 推理 (核心耗时) ---
            # 如果是 GPU 模式，建议同步一下时间，保证计时准确
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            t_infer_start = time.time()
            
            with torch.no_grad():
                # 直接对两副图进行匹配
                # mkpts_0, mkpts_1 = self.xfeat.match_xfeat(input_tensor0, input_tensor1, top_k=2048)
                mkpts_0, mkpts_1 = self.xfeat.match_xfeat_star(input_tensor0, input_tensor1, top_k=2048)

                # 先提取描述子然后进行匹配
                # out0 = self.xfeat.detectAndCompute(input_tensor0, top_k=2048)[0]
                # out1 = self.xfeat.detectAndCompute(input_tensor1, top_k=2048)[0]
                # mkpts_0, mkpts_1 = self.xfeat.match_xfeat(out0, out1, top_k=2048)

                # print(out1)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
                
            t_infer_end = time.time()

            if len(mkpts_0) < 5:
                return

            # 数据回传 CPU (兼容性处理)
            if isinstance(mkpts_0, torch.Tensor):
                pts0_np = mkpts_0.cpu().numpy()
            else:
                pts0_np = mkpts_0
            
            if isinstance(mkpts_1, torch.Tensor):
                pts1_np = mkpts_1.cpu().numpy()
            else:
                pts1_np = mkpts_1

            # --- 阶段 3: 后处理 (三角化 + 发布) ---
            points_4d = cv2.triangulatePoints(self.P0, self.P1, pts0_np.T, pts1_np.T)
            points_3d = points_4d[:3, :] / points_4d[3, :]
            points_3d = points_3d.T 

            valid_mask = (points_3d[:, 2] > 0.1) & (points_3d[:, 2] < 50.0)
            valid_points = points_3d[valid_mask]

            self.publish_pointcloud(valid_points, msg_left.header)
            self.publish_visual_matches(cv_img0, cv_img1, pts0_np, pts1_np)
            
            t_end = time.time()

            # --- 打印耗时统计 ---
            dur_pre = (t_preprocess - t_start) * 1000
            dur_inf = (t_infer_end - t_infer_start) * 1000
            dur_post = (t_end - t_infer_end) * 1000
            dur_total = (t_end - t_start) * 1000

            self.get_logger().info(
                f"耗时统计 [ms] | 预处理: {dur_pre:.1f} | 推理: {dur_inf:.1f} | 后处理: {dur_post:.1f} | 总计: {dur_total:.1f} | 匹配点数: {len(pts0_np)}"
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