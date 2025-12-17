#!/usr/bin/env python3
import os
import sys
import time
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import torch
import message_filters
from cv_bridge import CvBridge

# ROS 消息
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2

# 引入 XFeat
from modules.xfeat import XFeat

def get_rotation_translation_from_matrix(matrix):
    """从4x4矩阵提取四元数和平移"""
    import tf_transformations
    trans = matrix[:3, 3]
    quat = tf_transformations.quaternion_from_matrix(matrix)
    return trans, quat

class XFeatStereoVO(Node):
    def __init__(self):
        super().__init__('xfeat_stereo_vo')

        self.get_logger().info("正在初始化 XFeat 双目 VO (Raw+鱼眼)...")

        # ---------------------------------------------------------
        # 1. 模型初始化
        # ---------------------------------------------------------
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.xfeat = XFeat().to(self.device).eval()

        # ---------------------------------------------------------
        # 2. 状态变量 (VO核心)
        # ---------------------------------------------------------
        # 存储上一帧的数据: {'keypoints': np, 'descriptors': torch, 'points_3d': np}
        self.prev_frame = None 
        
        # 全局位姿 (T_world_cam), 初始化为单位矩阵
        self.cur_pose = np.eye(4) 
        
        # ---------------------------------------------------------
        # 3. ROS 通信
        # ---------------------------------------------------------
        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)

        # 发布者
        self.pub_odom = self.create_publisher(Odometry, 'xfeat/odom', 10)
        self.pub_path = self.create_publisher(Path, 'xfeat/path', 10)
        self.pub_cloud = self.create_publisher(PointCloud2, 'xfeat/point_cloud', 10)
        # 用于可视化匹配追踪
        self.pub_matches = self.create_publisher(Image, 'xfeat/tracking', 10)

        # 路径消息缓存
        self.path_msg = Path()

        # 相机参数
        self.cam0_params = None
        self.cam1_params = None

        # 订阅者
        self.sub_info0 = self.create_subscription(CameraInfo, '/cam0/image_raw_info', self.info0_cb, 10)
        self.sub_info1 = self.create_subscription(CameraInfo, '/cam1/image_raw_info', self.info1_cb, 10)

        sub_img0 = message_filters.Subscriber(self, Image, '/cam0/image_raw')
        sub_img1 = message_filters.Subscriber(self, Image, '/cam1/image_raw')
        self.sync = message_filters.ApproximateTimeSynchronizer([sub_img0, sub_img1], 10, 0.05)
        self.sync.registerCallback(self.stereo_callback)

    def parse_camera_info(self, msg):
        """解析内参"""
        return {
            'K': np.array(msg.k).reshape(3, 3).astype(np.float64),
            'D': np.array(msg.d).astype(np.float64),
            'R': np.array(msg.r).reshape(3, 3).astype(np.float64),
            'P': np.array(msg.p).reshape(3, 4).astype(np.float64),
            'model': msg.distortion_model,
            'frame_id': msg.header.frame_id
        }

    def info0_cb(self, msg):
        self.cam0_params = self.parse_camera_info(msg)
        self.destroy_subscription(self.sub_info0)

    def info1_cb(self, msg):
        self.cam1_params = self.parse_camera_info(msg)
        self.destroy_subscription(self.sub_info1)

    def preprocess(self, cv_img):
        """转Tensor"""
        if len(cv_img.shape) == 2:
            img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        x = torch.tensor(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        return x.to(self.device)

    def undistort_keypoints(self, kps, params):
        """
        将 Raw 图上的 Keypoints (N, 2) 去畸变并转换到 Rectified 坐标系
        """
        if len(kps) == 0: return kps
        pts_in = np.ascontiguousarray(kps[:, :2]).reshape(-1, 1, 2).astype(np.float64)
        
        K, D, R, P = params['K'], params['D'], params['R'], params['P']
        K_new = P[:3, :3]
        
        is_fisheye = ('fisheye' in params['model']) or ('equidistant' in params['model'])

        if is_fisheye:
            D_fish = D.flatten()[:4]
            try:
                # 鱼眼去畸变 + 旋转 R + 投影 K_new
                pts_out = cv2.fisheye.undistortPoints(pts_in, K, D_fish, R=R, P=K_new)
            except:
                return kps
        else:
            # 普通去畸变
            pts_out = cv2.undistortPoints(pts_in, K, D, R=R, P=K_new)
            
        return pts_out.reshape(-1, 2)

    def match_descriptors(self, desc1, desc2, threshold=0.82):
        """
        简单的 MNN (Mutual Nearest Neighbor) 匹配
        desc1, desc2: (N, C) torch tensor
        """
        # 计算距离矩阵 (内积，假设描述子已归一化)
        # sim_mat = desc1 @ desc2.t()
        # 由于 XFeat 输出通常是 float32，直接矩阵乘法很快
        scores = torch.einsum('bd,nd->bn', desc1, desc2)

        # 双向寻找最大值
        # max_1_2: desc1 中每个点在 desc2 中的最佳匹配
        val_1, idx_1 = scores.max(dim=1)
        # max_2_1: desc2 中每个点在 desc1 中的最佳匹配
        val_2, idx_2 = scores.max(dim=0)

        # 相互匹配一致性检查 (Mutual Check)
        matches = []
        # 将 tensor 转到 CPU 处理索引
        idx_1_cpu = idx_1.cpu().numpy()
        idx_2_cpu = idx_2.cpu().numpy()
        
        for i, j in enumerate(idx_1_cpu):
            if idx_2_cpu[j] == i: # 互为最佳
                # 可选：增加相似度阈值过滤
                # if val_1[i] > threshold: 
                matches.append([i, j])
        
        return np.array(matches)

    def stereo_callback(self, msg_left, msg_right):
        if self.cam0_params is None or self.cam1_params is None:
            return

        t_start = time.time()

        # 1. 读取 Raw 图像
        img0 = self.bridge.imgmsg_to_cv2(msg_left, 'bgr8')
        img1 = self.bridge.imgmsg_to_cv2(msg_right, 'bgr8')
        
        x0 = self.preprocess(img0)
        x1 = self.preprocess(img1)

        # 2. XFeat 提取特征 (Detect & Compute)
        # 我们这里分别提取，以便复用 Left 的特征进行帧间匹配
        with torch.no_grad():
            # top_k 可以适当调大，保证跟踪稳定性
            out0 = self.xfeat.detectAndCompute(x0, top_k=2000)[0] 
            out1 = self.xfeat.detectAndCompute(x1, top_k=2000)[0]
        
        kps0 = out0['keypoints'].cpu().numpy()  # (N, 2)
        desc0 = out0['descriptors']             # (N, 64) GPU
        scores0 = out0['scores'].cpu().numpy()

        kps1 = out1['keypoints'].cpu().numpy()
        desc1 = out1['descriptors']

        print(desc1.shape)
        # ---------------------------------------------------------
        # 3. 双目匹配 (Stereo Matching) -> 恢复当前帧深度
        # ---------------------------------------------------------
        # 左图描述子 vs 右图描述子
        stereo_matches = self.match_descriptors(desc0, desc1)
        
        if len(stereo_matches) < 10:
            self.get_logger().warn("双目匹配点过少，跳过")
            return

        # 获取匹配点坐标 (Raw)
        idx0_stereo = stereo_matches[:, 0]
        idx1_stereo = stereo_matches[:, 1]
        
        pts0_raw_s = kps0[idx0_stereo]
        pts1_raw_s = kps1[idx1_stereo]

        # 去畸变 (Raw -> Rectified)
        pts0_rect = self.undistort_keypoints(pts0_raw_s, self.cam0_params)
        pts1_rect = self.undistort_keypoints(pts1_raw_s, self.cam1_params)

        # 三角化 -> 获取当前帧左相机坐标系下的 3D 点
        points_4d = cv2.triangulatePoints(
            self.cam0_params['P'], self.cam1_params['P'], 
            pts0_rect.T, pts1_rect.T
        )
        points_3d = (points_4d[:3] / points_4d[3]).T # (M, 3)

        # 过滤无效深度
        valid_mask = (points_3d[:, 2] > 0.1) & (points_3d[:, 2] < 50.0)
        
        # 构建当前帧的 "Map"
        # 这是一个稀疏结构：Current Left Frame 所有的 Keypoints 中，
        # 只有一部分(idx0_stereo)拥有了3D坐标。
        # 我们用一个数组存储：如果 kp[i] 没有3D点，设为 NaN 或 infinite
        curr_points_3d = np.full((len(kps0), 3), np.nan)
        # 只填充有效的3D点
        valid_indices = idx0_stereo[valid_mask]
        curr_points_3d[valid_indices] = points_3d[valid_mask]

        # ---------------------------------------------------------
        # 4. 帧间追踪 (VO Logic)
        # ---------------------------------------------------------
        T_curr = np.eye(4) # 相对运动
        is_tracking_good = False

        if self.prev_frame is not None:
            # 匹配: 上一帧左图 vs 当前帧左图
            # desc0 是当前帧, self.prev_frame['descriptors'] 是上一帧
            temporal_matches = self.match_descriptors(self.prev_frame['descriptors'], desc0)

            if len(temporal_matches) > 20:
                # 准备 PnP 数据
                # Obj Pts: 上一帧的 3D 点 (World frame or Prev Camera frame)
                # Img Pts: 当前帧的 2D 点 (Undistorted Keypoints)
                
                idx_prev = temporal_matches[:, 0] # 上一帧特征点索引
                idx_curr = temporal_matches[:, 1] # 当前帧特征点索引

                # 查找哪些匹配点在上一帧拥有有效的 3D 坐标
                prev_p3d = self.prev_frame['points_3d'][idx_prev]
                
                # 过滤掉 NaN 的点 (即上一帧虽有特征点，但双目没匹配上，所以没深度)
                has_depth_mask = ~np.isnan(prev_p3d[:, 0])
                
                if np.sum(has_depth_mask) > 10:
                    obj_pts = prev_p3d[has_depth_mask] # 3D点 (上一帧相机坐标系)
                    
                    # 对应的当前帧 2D 点 (Raw)
                    curr_p2d_raw = kps0[idx_curr][has_depth_mask]
                    
                    # 关键：PnP 必须使用 去畸变后的点 和 校正后的内参 (P[:3,:3])
                    curr_p2d_rect = self.undistort_keypoints(curr_p2d_raw, self.cam0_params)
                    
                    K_rect = self.cam0_params['P'][:3, :3]
                    
                    # Solve PnP (RANSAC)
                    # 求解的是: World(上一帧) 到 Camera(当前帧) 的变换
                    success, rvec, tvec, inliers = cv2.solvePnPRansac(
                        obj_pts, curr_p2d_rect, K_rect, None, 
                        iterationsCount=100, reprojectionError=2.0, confidence=0.99
                    )

                    if success and len(inliers) > 15: # 提高一点内点门槛
                        R, _ = cv2.Rodrigues(rvec)
                        t = tvec
                        
                        # T_p_c: 把点从上一帧变换到当前帧 (Point Transform)
                        T_p_c = np.eye(4)
                        T_p_c[:3, :3] = R
                        T_p_c[:3, 3] = t.flatten()
                        
                        # T_c_p: 相机在上一帧坐标系下的位姿变化 (Motion)
                        T_c_p = np.linalg.inv(T_p_c)
                        
                        # --- [关键修改] 运动阈值检测 (Keyframe Strategy) ---
                        # 计算相对位移距离和旋转角度
                        delta_trans = np.linalg.norm(T_c_p[:3, 3])
                        
                        # 计算相对旋转角 (Trace method)
                        trace_R = np.trace(T_c_p[:3, :3])
                        # clip防止数值误差导致acos越界
                        delta_angle = np.arccos(np.clip((trace_R - 1) / 2, -1.0, 1.0))
                        delta_angle_deg = np.degrees(delta_angle)

                        # 阈值：位移 < 2cm 且 旋转 < 0.2度，则认为是静止噪音/抖动，忽略
                        if delta_trans < 0.02 and delta_angle_deg < 0.2:
                            # 即使不更新位姿，也可以选择是否更新 prev_frame
                            # 这里我们选择不更新 prev_frame，这样下一帧还是和当前这个参考帧比
                            # 这就是最简单的 Keyframe 思想
                            return 

                        # 只有运动足够大，才更新全局位姿
                        self.cur_pose = self.cur_pose @ T_c_p
                        is_tracking_good = True
                        
                        # 更新上一帧 (设为关键帧)
                        self.prev_frame = {
                            'keypoints': kps0,
                            'descriptors': desc0,
                            'points_3d': curr_points_3d
                        }
                        
                        self.publish_tracking_img(img0, curr_p2d_raw, inliers)

                    else:
                        self.get_logger().warn(f"PnP 失败或内点不足 ({len(inliers)})")
            
            else:
                 # 第一帧，或者跟丢了，强制设为关键帧
                 self.prev_frame = {
                    'keypoints': kps0,
                    'descriptors': desc0,
                    'points_3d': curr_points_3d
                }

        # ---------------------------------------------------------
        # 5. 更新状态与发布
        # ---------------------------------------------------------
        # 更新上一帧数据
        self.prev_frame = {
            'keypoints': kps0,
            'descriptors': desc0,
            'points_3d': curr_points_3d # 保存当前帧计算出的3D点供下一帧用
        }

        # 如果追踪失败（或者第一帧），不发布里程计更新，或者重置
        if is_tracking_good:
            self.publish_odometry(msg_left.header)
            
            # 顺便发布当前帧生成的点云 (变换到全局坐标系用于可视化)
            # 仅发布 valid 的点
            valid_pts = curr_points_3d[~np.isnan(curr_points_3d[:,0])]
            if len(valid_pts) > 0:
                # 把点云转到全局坐标系
                R_w_c = self.cur_pose[:3, :3]
                t_w_c = self.cur_pose[:3, 3]
                pts_global = (R_w_c @ valid_pts.T).T + t_w_c
                self.publish_cloud(pts_global, msg_left.header)

        dur = (time.time() - t_start) * 1000
        self.get_logger().info(f"VO Step: {dur:.1f}ms | Tracked: {is_tracking_good}")

    def publish_odometry(self, header):
        # 1. 构造 Odometry 消息
        odom = Odometry()
        odom.header.stamp = header.stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link" # 或者 cam0_optical_frame

        # 从 4x4 矩阵提取位姿
        import tf_transformations
        trans = self.cur_pose[:3, 3]
        quat = tf_transformations.quaternion_from_matrix(self.cur_pose)

        odom.pose.pose.position.x = trans[0]
        odom.pose.pose.position.y = trans[1]
        odom.pose.pose.position.z = trans[2]
        odom.pose.pose.orientation.x = quat[0]
        odom.pose.pose.orientation.y = quat[1]
        odom.pose.pose.orientation.z = quat[2]
        odom.pose.pose.orientation.w = quat[3]

        self.pub_odom.publish(odom)

        # 2. 发布 TF (odom -> base_link)
        t = TransformStamped()
        t.header.stamp = header.stamp
        t.header.frame_id = "odom"
        t.child_frame_id = self.cam0_params['frame_id'] # 也就是相机坐标系
        t.transform.translation.x = trans[0]
        t.transform.translation.y = trans[1]
        t.transform.translation.z = trans[2]
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        self.tf_broadcaster.sendTransform(t)

        # 3. 发布 Path
        pose_stamped = PoseStamped()
        pose_stamped.header = odom.header
        pose_stamped.pose = odom.pose.pose
        self.path_msg.header = odom.header
        self.path_msg.poses.append(pose_stamped)
        self.pub_path.publish(self.path_msg)

    def publish_cloud(self, points, header):
        if len(points) == 0: return
        # 注意：这里我们发布到了 odom 坐标系
        h = Header()
        h.stamp = header.stamp
        h.frame_id = "odom" 
        pc2 = point_cloud2.create_cloud_xyz32(h, points)
        self.pub_cloud.publish(pc2)

    def publish_tracking_img(self, img, pts, inliers):
        """画出用于 PnP 的内点"""
        vis = img.copy()
        for i, idx in enumerate(inliers):
            pt = pts[idx[0]]
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
        self.pub_matches.publish(self.bridge.cv2_to_imgmsg(vis, 'bgr8'))

def main():
    rclpy.init()
    node = XFeatStereoVO()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()