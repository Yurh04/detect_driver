"""
YOLOv8 驾驶员行为检测引擎
支持三种行为检测：抽烟(Smoke)、用手机(Phone)、喝水(Drink)
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DetectionEngine:
    """YOLOv8检测引擎"""
    
    def __init__(self, model_path: str = "models/best.pt"):
        """
        初始化检测引擎
        
        Args:
            model_path: 模型权重文件路径
        """
        self.model_path = model_path
        self.model = None
        self.class_names = []  # 将从模型自动获取
        self.class_colors = {}  # 将根据类别数自动生成
        self.load_model()
    
    def load_model(self):
        """加载YOLOv8模型"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"模型文件不存在: {self.model_path}")
                raise FileNotFoundError(f"模型文件未找到: {self.model_path}")
            
            logger.info(f"正在加载模型: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # 自动获取类别信息
            if hasattr(self.model, 'names'):
                self.class_names = list(self.model.names.values())
                num_classes = len(self.class_names)
                logger.info(f"检测到 {num_classes} 个类别: {self.class_names[:5]}{'...' if num_classes > 5 else ''}")
                
                # 自动生成颜色（使用HSV色彩空间均匀分布）
                import colorsys
                self.class_colors = {}
                for i in range(num_classes):
                    hue = i / max(num_classes, 1)
                    rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
                    # 转换为BGR格式(OpenCV使用)
                    self.class_colors[i] = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))
            else:
                logger.warning("无法获取模型类别信息，使用默认设置")
                self.class_names = ['Object']
                self.class_colors = {0: (255, 255, 255)}
            
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise
    
    def detect_image(self, image: np.ndarray, conf_threshold: float = 0.25) -> Dict:
        """
        检测单张图像
        
        Args:
            image: 输入图像(BGR格式)
            conf_threshold: 置信度阈值
            
        Returns:
            检测结果字典
        """
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        try:
            # 运行推理
            results = self.model(image, conf=conf_threshold, verbose=False)[0]
            
            # 提取检测结果
            detections = []
            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                confidences = results.boxes.conf.cpu().numpy()
                class_ids = results.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    # 获取基本信息
                    class_name = self.class_names[cls_id]
                    
                    # 目标行为类别
                    target_behaviors = ['Smoke', 'Phone', 'Drink']
                    
                    # 过滤和映射逻辑
                    final_class_name = class_name
                    final_class_id = cls_id
                    
                    # 如果是通用模型(COCO)，尝试映射到目标行为
                    if 'person' in self.class_names: 
                        if class_name == 'cell phone':
                            final_class_name = 'Phone'
                            final_class_id = 1
                        elif class_name in ['bottle', 'cup', 'wine glass']:
                            final_class_name = 'Drink'
                            final_class_id = 2
                        elif class_name == 'person':
                            # 保留驾驶员检测，标记为 Driver
                            final_class_name = 'Driver'
                            final_class_id = 3
                        else:
                            continue
                    else:
                        # 正式模型逻辑
                        if class_name not in target_behaviors:
                            continue

                    detections.append({
                        'bbox': [float(x) for x in box.tolist()],
                        'confidence': float(conf),
                        'class_id': int(final_class_id),
                        'class_name': final_class_name
                    })
            
            # === 两阶段检测优化 (针对通用模型的小目标) ===
            # 如果是通用模型，对每个检测到的 Driver (person) 进行局部二次检测
            if 'person' in self.class_names and len(detections) > 0:
                img_h, img_w = image.shape[:2]
                
                # 收集所有Driver的框
                driver_boxes = [d['bbox'] for d in detections if d['class_name'] == 'Driver']
                
                for d_box in driver_boxes:
                    x1, y1, x2, y2 = [int(v) for v in d_box]
                    
                    # 扩大裁剪区域 (Margin)，包含手部活动范围
                    margin_x = int((x2 - x1) * 0.2)
                    margin_y = int((y2 - y1) * 0.2)
                    crop_x1 = max(0, x1 - margin_x)
                    crop_y1 = max(0, y1 - margin_y)
                    crop_x2 = min(img_w, x2 + margin_x)
                    crop_y2 = min(img_h, y2 + margin_y + int((y2 - y1) * 0.3)) # 下方多留点空间看手
                    
                    # 裁剪图像
                    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1: continue
                    crop_img = image[crop_y1:crop_y2, crop_x1:crop_x2]
                    
                    # 局部检测（使用更低的阈值）
                    try:
                        # 局部推理
                        sub_results = self.model(crop_img, conf=0.1, verbose=False)[0]
                        if sub_results.boxes is not None and len(sub_results.boxes) > 0:
                            sub_boxes = sub_results.boxes.xyxy.cpu().numpy()
                            sub_confs = sub_results.boxes.conf.cpu().numpy()
                            sub_cls_ids = sub_results.boxes.cls.cpu().numpy().astype(int)
                            
                            for sb, sc, sid in zip(sub_boxes, sub_confs, sub_cls_ids):
                                s_name = self.class_names[sid]
                                s_final_name = None
                                s_final_id = -1
                                
                                # 只关注局部图中的小物体
                                if s_name == 'cell phone':
                                    s_final_name = 'Phone'
                                    s_final_id = 1
                                elif s_name in ['bottle', 'cup', 'wine glass']:
                                    s_final_name = 'Drink'
                                    s_final_id = 2
                                elif s_name == 'cigarette': # 虽然COCO没有，但以防万一
                                    s_final_name = 'Smoke'
                                    s_final_id = 0
                                
                                if s_final_name:
                                    # 坐标还原到原图，并确保转换为Python原生float类型
                                    global_box = [
                                        float(sb[0] + crop_x1),
                                        float(sb[1] + crop_y1),
                                        float(sb[2] + crop_x1),
                                        float(sb[3] + crop_y1)
                                    ]
                                    
                                    # 添加到最终结果
                                    detections.append({
                                        'bbox': global_box,
                                        'confidence': float(sc),
                                        'class_id': int(s_final_id),
                                        'class_name': s_final_name
                                    })
                    except Exception as e:
                        logger.warning(f"局部检测失败: {e}")
            
            return {
                'success': True,
                'detections': detections,
                'num_detections': len(detections)
            }
        
        except Exception as e:
            logger.error(f"检测失败: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'detections': [],
                'num_detections': 0
            }
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        在图像上绘制检测框
        
        Args:
            image: 输入图像
            detections: 检测结果列表
            
        Returns:
            标注后的图像
        """
        annotated_image = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            class_id = det['class_id']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # 获取颜色
            color = self.class_colors.get(class_id, (255, 255, 255))
            
            # 绘制边界框
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签背景
            label = f"{class_name}: {confidence:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                annotated_image,
                (x1, y1 - label_height - 10),
                (x1 + label_width, y1),
                color,
                -1
            )
            
            # 绘制标签文字
            cv2.putText(
                annotated_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return annotated_image
    
    def process_video(
        self, 
        video_path: str, 
        output_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        process_every_n_frames: int = 1
    ) -> Dict:
        """
        处理视频文件
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径(可选)
            conf_threshold: 置信度阈值
            process_every_n_frames: 每N帧处理一次
            
        Returns:
            处理结果统计
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")
        
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"视频属性: {width}x{height}, {fps}fps, {total_frames}帧")
        
        # 准备输出视频
        out = None
        if output_path:
            # 尝试使用avc1编码(H.264)，浏览器兼容性更好
            try:
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            except Exception:
                # 如果失败回退到mp4v
                logger.warning("avc1编码失败，回退到mp4v")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 统计数据 - 初始化所有类别
        behavior_counts = {name: 0 for name in self.class_names}
        behavior_frames = {name: [] for name in self.class_names}
        all_detections = []
        
        frame_idx = 0
        processed_frames = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 按设定频率处理帧
                if frame_idx % process_every_n_frames == 0:
                    result = self.detect_image(frame, conf_threshold)
                    
                    if result['success'] and result['num_detections'] > 0:
                        detections = result['detections']
                        
                        # 统计每种行为
                        for det in detections:
                            class_name = det['class_name']
                            # 确保键值存在，防止因映射导致KeyError
                            if class_name not in behavior_counts:
                                behavior_counts[class_name] = 0
                                behavior_frames[class_name] = []
                                
                            behavior_counts[class_name] += 1
                            if frame_idx not in behavior_frames[class_name]:
                                behavior_frames[class_name].append(frame_idx)
                        
                        # 记录该帧的检测结果
                        all_detections.append({
                            'frame': frame_idx,
                            'timestamp': frame_idx / fps,
                            'detections': detections
                        })
                        
                        # 绘制检测框
                        frame = self.draw_detections(frame, detections)
                    
                    processed_frames += 1
                
                # 写入输出视频
                if out is not None:
                    out.write(frame)
                
                frame_idx += 1
                
                # 进度日志
                if frame_idx % 100 == 0:
                    logger.info(f"处理进度: {frame_idx}/{total_frames} 帧")
        
        finally:
            cap.release()
            if out is not None:
                out.release()
        
        logger.info(f"视频处理完成: 共处理 {processed_frames} 帧")
        
        return {
            'success': True,
            'video_info': {
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': total_frames,
                'processed_frames': processed_frames
            },
            'statistics': {
                'behavior_counts': behavior_counts,
                'behavior_frames': behavior_frames,
                'total_detections': sum(behavior_counts.values())
            },
            'detections': all_detections
        }


# 测试代码
if __name__ == "__main__":
    # 测试模型加载
    try:
        engine = DetectionEngine()
        print("✓ 检测引擎初始化成功")
    except Exception as e:
        print(f"✗ 检测引擎初始化失败: {e}")
