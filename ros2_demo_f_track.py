#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import time

from modules.xfeat import XFeat

class XFeatKLTTrackerNode(Node):
    def __init__(self):
        super().__init__('xfeat_klt_tracker_node')

        # 1. 初始化 XFeat (仅用于特征提取)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.xfeat = XFeat().to(self.device)
        self.xfeat.eval()

        # 2. 状态维护
        self.prev_gray = None   # 上一帧灰度图
        self.curr_pts = None    # 当前追踪的点集 (N, 2)
        self.curr_ids = None    # 当前点的 ID 集 (N,)
        self.curr_ages = None   # 当前点的追踪次数 (N,)
        
        self.next_id = 0
        self.min_pts_threshold = 150  # 当追踪点少于此值时，触发 XFeat 补充新点
        self.max_pts_count = 300      # 维持的最大点数

        # KLT 光流参数
        self.lk_params = dict(winSize=(21, 21), 
                              maxLevel=3, 
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        # 3. ROS 工具
        self.bridge = CvBridge()
        self.sub_img = self.create_subscription(Image, '/cam0/image_raw', self.image_callback, 10)
        self.pub_vis = self.create_publisher(Image, 'xfeat/klt_vis', 10)

        self.prev_time = time.time()
        self.fps = 0.0

    def preprocess_tensor(self, cv_img):
        img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        return torch.tensor(img).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
    def image_callback(self, msg):
        # 计算 FPS (平滑处理)
        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now
        if dt > 0:
            current_fps = 1.0 / dt
            # 指数平滑，让显示的 FPS 波动不那么剧烈
            self.fps = self.fps * 0.9 + current_fps * 0.1
        
        t_start = time.time()
        cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        curr_gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        h, w = curr_gray.shape

        # --- 步骤 1: KLT 追踪 ---
        if self.prev_gray is not None and self.curr_pts is not None and len(self.curr_pts) > 0:
            new_pts, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, curr_gray, 
                self.curr_pts.astype(np.float32), None, **self.lk_params
            )
            
            status = status.reshape(-1).astype(bool)
            # 越界检查（保留画面边缘 5 像素缓冲区）
            in_bounds = (new_pts[:,0] > 5) & (new_pts[:,0] < w-5) & (new_pts[:,1] > 5) & (new_pts[:,1] < h-5)
            valid = status & in_bounds

            # 静止抑制：移动小于 0.1 像素则不更新坐标
            move_dist = np.linalg.norm(new_pts - self.curr_pts.astype(np.float32), axis=1)
            still_mask = move_dist < 0.1
            new_pts[still_mask] = self.curr_pts[still_mask]

            self.curr_pts = new_pts[valid]
            self.curr_ids = self.curr_ids[valid]
            self.curr_ages = self.curr_ages[valid] + 1
        else:
            self.curr_pts = np.empty((0, 2))
            self.curr_ids = np.array([], dtype=int)
            self.curr_ages = np.array([], dtype=int)

        # --- 步骤 2: 空间分布检查与特征补充 ---
        # 即使总数够，如果有点移出画面，也需要补充。这里将阈值设得更灵敏。
        if len(self.curr_pts) < self.max_pts_count:
            # 1. 创建掩码，遮住已有追踪点周围的区域（防止特征点扎堆）
            # 这种方式会强制 XFeat 在“空白区域”寻找新点
            mask = np.ones((h, w), dtype=np.uint8) * 255
            if len(self.curr_pts) > 0:
                for pt in self.curr_pts:
                    # 关键：在追踪点周围画黑圆，半径 25 像素内的区域不许提新点
                    cv2.circle(mask, (int(pt[0]), int(pt[1])), 25, 0, -1)

            # 2. 调用 XFeat 提取潜力点
            input_tensor = self.preprocess_tensor(cv_img)
            with torch.no_grad():
                results = self.xfeat.detectAndCompute(input_tensor, top_k=self.max_pts_count)[0]
            
            new_detected_pts = results['keypoints'].cpu().numpy()
            
            # 3. 筛选出位于空白区（mask > 0）的优质新点
            filtered_new_pts = []
            for pt in new_detected_pts:
                px, py = int(pt[0]), int(pt[1])
                # 再次检查越界，并根据 mask 筛选
                if 5 < px < w-5 and 5 < py < h-5:
                    if mask[py, px] > 0:
                        filtered_new_pts.append(pt)
            
            # 4. 补充新点到点集
            if len(filtered_new_pts) > 0:
                # 补充到最大上限
                num_to_add = self.max_pts_count - len(self.curr_pts)
                add_pts = np.array(filtered_new_pts[:num_to_add])
                
                add_ids = np.arange(self.next_id, self.next_id + len(add_pts))
                self.next_id += len(add_pts)
                
                if len(self.curr_pts) > 0:
                    self.curr_pts = np.vstack([self.curr_pts, add_pts])
                    self.curr_ids = np.concatenate([self.curr_ids, add_ids])
                    self.curr_ages = np.concatenate([self.curr_ages, np.ones(len(add_pts), dtype=int)])
                else:
                    self.curr_pts = add_pts
                    self.curr_ids = add_ids
                    self.curr_ages = np.ones(len(add_pts), dtype=int)

        # --- 步骤 3: 状态与可视化 ---
        self.prev_gray = curr_gray
        self.visualize_and_publish(cv_img, self.curr_pts, self.curr_ids, self.curr_ages)

    
    def visualize_and_publish(self, img, pts, ids, ages):
            vis_img = img.copy()
            
            # 1. 绘制特征点
            for i in range(len(pts)):
                x, y = int(round(pts[i][0])), int(round(pts[i][1]))
                age_val = int(ages[i])
                # 颜色：新点红色 -> 稳定点绿色
                green = int(max(0, min(255, age_val * 10)))
                red = int(max(0, min(255, 255 - age_val * 10)))
                color = (0, green, red) 
                
                cv2.circle(vis_img, (x, y), 3, color, -1)
                
                # 追踪超过10帧的点才显示ID，避免画面太乱
                if age_val > 10:
                    cv2.putText(vis_img, str(int(ids[i])), (x + 4, y - 4), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

            # 2. 绘制 UI 叠加层 (HUD)
            # 创建一个黑色半透明矩形区域作为背景，方便看清文字
            overlay = vis_img.copy()
            cv2.rectangle(overlay, (5, 5), (180, 75), (0, 0, 0), -1)
            # 混合透明度
            cv2.addWeighted(overlay, 0.5, vis_img, 0.5, 0, vis_img)

            # 3. 写入文本信息
            # 字体、大小、颜色设置
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            white = (255, 255, 255)
            
            pts_count = len(pts)
            fps_text = f"FPS: {self.fps:.1f}"
            pts_text = f"Features: {pts_count}"
            id_text = f"Total IDs: {self.next_id}"
            
            # 状态颜色：如果点数太少，显示红色警告
            status_color = (0, 255, 0) if pts_count > self.min_pts_threshold else (0, 0, 255)

            cv2.putText(vis_img, fps_text, (15, 25), font, font_scale, white, 1)
            cv2.putText(vis_img, pts_text, (15, 45), font, font_scale, status_color, 1)
            cv2.putText(vis_img, id_text, (15, 65), font, font_scale, white, 1)

            # 4. 发布
            vis_msg = self.bridge.cv2_to_imgmsg(vis_img, "bgr8")
            self.pub_vis.publish(vis_msg)

def main():
    rclpy.init()
    node = XFeatKLTTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()