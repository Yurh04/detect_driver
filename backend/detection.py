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
        self.helper_model = None  # 辅助定位模型（用于找人）
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
            
            # 检查是否需要辅助模型：如果主模型里没有 person 类，且运行在监控视角视频，我们需要辅助定位
            # 即使不是监控视频，有辅助找人也能极大提升小目标的二阶段识别率
            if 'person' not in self.class_names:
                try:
                    helper_path = "models/yolov8n.pt"
                    if not os.path.exists(helper_path):
                        helper_path = "yolov8n.pt" # 尝试根目录
                    
                    logger.info(f"主模型缺少定位类，正在加载辅助定位模型: {helper_path}")
                    self.helper_model = YOLO(helper_path)
                    logger.info("辅助定位模型加载成功")
                except Exception as e:
                    logger.warning(f"辅助定位模型加载失败（但不影响主逻辑）: {e}")
                    
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
            
            # === 定义类别特定阈值 (压制误报，增加召回) ===
            # Smoke: 误报多，样本少 -> 设高阈值 (0.65)
            # Drink: 样本多，较重要 -> 设低阈值 (0.12)
            # Phone: 保持中等 (0.25)
            # Driver: 定位用 (0.25)
            class_thresholds = {
                'Smoke': 0.80,
                'Drink': 0.10,
                'Phone': 0.25,
                'Driver': 0.25
            }
            
            # 过滤第一阶段检测结果
            detections = [
                d for d in detections 
                if d['confidence'] >= class_thresholds.get(d['class_name'], conf_threshold)
            ]
            
            # === 两阶段检测优化 (深度重构) ===
            # 目标：无论主模型是否有person类，只要有定位能力，就进行裁剪放大检测
            
            # 1. 寻找潜在的驾驶员框 (定位源)
            driver_boxes = []
            
            # 优先从本轮已有的 detections 中找 Driver (如果是通用模型，detections 里已经有 Driver 了)
            driver_boxes = [d['bbox'] for d in detections if d['class_name'] == 'Driver']
            
            # 如果主模型没有定位到人，且我们有辅助模型，则调用辅助模型专门找人
            if not driver_boxes and self.helper_model:
                try:
                    # 降低阈值以提高定位成功率
                    helper_results = self.helper_model(image, conf=0.15, verbose=False)[0]
                    if helper_results.boxes is not None:
                        h_boxes = helper_results.boxes.xyxy.cpu().numpy()
                        h_classes = helper_results.boxes.cls.cpu().numpy().astype(int)
                        h_names = self.helper_model.names
                        
                        for hb, hc in zip(h_boxes, h_classes):
                            if h_names[hc] == 'person':
                                driver_boxes.append([float(x) for x in hb.tolist()])
                                # 如果是辅助模型找的人，我们也顺便存入detections供展示（可选）
                                detections.append({
                                    'bbox': [float(x) for x in hb.tolist()],
                                    'confidence': 0.8, # 辅助模型定位置信度
                                    'class_id': 99,
                                    'class_name': 'Driver'
                                })
                except Exception as e:
                    logger.warning(f"辅助定位推理失败: {e}")
            
            # 2. 对每个定位到的位置进行“特写”检测
            if driver_boxes:
                img_h, img_w = image.shape[:2]
                for d_box in driver_boxes:
                    x1, y1, x2, y2 = [int(v) for v in d_box]
                    
                    # 裁剪区域建议：扩大范围以包含更多可能的行为区域
                    margin_x = int((x2 - x1) * 0.35)  # 增加横向范围
                    margin_y = int((y2 - y1) * 0.3)   # 增加纵向范围
                    crop_x1 = max(0, x1 - margin_x)
                    crop_y1 = max(0, y1 - margin_y)
                    crop_x2 = min(img_w, x2 + margin_x)
                    crop_y2 = min(img_h, y2 + margin_y + int((y2 - y1) * 0.5))  # 向下扩展更多
                    
                    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1: continue
                    crop_img = image[crop_y1:crop_y2, crop_x1:crop_x2]
                    
                    # 使用【主模型】对"特写"进行识别，适当调高阈值以减少误报
                    try:
                        # 0.2 是一个较平衡的阈值，过低会导致背景噪点被识别为行为
                        detect_conf = 0.20 if 'person' not in self.class_names else 0.15
                        sub_results = self.model(crop_img, conf=detect_conf, verbose=False)[0]
                        if sub_results.boxes is not None:
                            s_boxes = sub_results.boxes.xyxy.cpu().numpy()
                            s_confs = sub_results.boxes.conf.cpu().numpy()
                            s_cls_ids = sub_results.boxes.cls.cpu().numpy().astype(int)
                            
                            for sb, sc, sid in zip(s_boxes, s_confs, s_cls_ids):
                                s_name = self.class_names[sid]
                                s_final_name = None
                                s_final_id = -1
                                
                                # 将局部坐标映射回全局坐标
                                g_box = [
                                    float(sb[0] + crop_x1),
                                    float(sb[1] + crop_y1),
                                    float(sb[2] + crop_x1),
                                    float(sb[3] + crop_y1)
                                ]
                                
                                # 处理自定义模型的类别映射
                                if 'person' not in self.class_names:
                                    # 如果是专用模型，直接透传它认准的结果
                                    # ！！！关键：在此处应用类别特定阈值过滤！！！
                                    target_threshold = class_thresholds.get(s_name, 0.20)
                                    if sc >= target_threshold:
                                        detections.append({
                                            'bbox': g_box,
                                            'confidence': float(sc),
                                            'class_id': int(sid),
                                            'class_name': s_name
                                        })
                                else:
                                    # 通用模型的映射逻辑
                                    if s_name == 'cell phone' and sc >= class_thresholds.get('Phone', 0.25):
                                        detections.append({
                                            'bbox': g_box, 'confidence': float(sc),
                                            'class_id': 1, 'class_name': 'Phone'
                                        })
                                    elif s_name in ['bottle', 'cup', 'wine glass'] and sc >= class_thresholds.get('Drink', 0.12):
                                        detections.append({
                                            'bbox': g_box, 'confidence': float(sc),
                                            'class_id': 2, 'class_name': 'Drink'
                                        })
                                    elif s_name == 'cigarette' and sc >= class_thresholds.get('Smoke', 0.65):
                                        detections.append({
                                            'bbox': g_box, 'confidence': float(sc),
                                            'class_id': 0, 'class_name': 'Smoke'
                                        })
                    except Exception as e:
                        logger.warning(f"特写检测失败: {e}")
            
            # === 3. 非极大值抑制 (NMS) 去重 ===
            # 目标：解决“一个行为被识别成好几种物体”以及框重叠的问题
            if len(detections) > 1:
                # 按置信度从高到低排序
                detections.sort(key=lambda x: x['confidence'], reverse=True)
                
                keep_detections = []
                while detections:
                    best_det = detections.pop(0)
                    keep_detections.append(best_det)
                    
                    # 过滤掉与当前最强框重叠严重的框
                    remaining = []
                    for det in detections:
                        iou = self._calculate_iou(best_det['bbox'], det['bbox'])
                        # 如果 IOU 较大 (重叠 > 45%)，认为是同一个目标，直接丢弃
                        if iou < 0.45:
                            remaining.append(det)
                    detections = remaining
                detections = keep_detections
            
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
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """计算两个边界框的交并比 (IOU)"""
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])
        
        inter_width = max(0, x2_min - x1_max)
        inter_height = max(0, y2_min - y1_max)
        inter_area = inter_width * inter_height
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
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
