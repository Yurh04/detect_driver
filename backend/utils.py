"""
工具函数模块
提供视频处理、文件管理等辅助功能
"""

import os
import base64
import cv2
import numpy as np
from typing import Dict, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def ensure_dir(directory: str):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"创建目录: {directory}")


def save_uploaded_file(file_content: bytes, filename: str, upload_dir: str = "uploads") -> str:
    """
    保存上传的文件
    
    Args:
        file_content: 文件内容
        filename: 原始文件名
        upload_dir: 上传目录
        
    Returns:
        保存的文件路径
    """
    ensure_dir(upload_dir)
    
    # 生成唯一文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = os.path.splitext(filename)[1]
    unique_filename = f"{timestamp}{ext}"
    
    filepath = os.path.join(upload_dir, unique_filename)
    
    with open(filepath, 'wb') as f:
        f.write(file_content)
    
    logger.info(f"文件已保存: {filepath}")
    return filepath


def image_to_base64(image: np.ndarray) -> str:
    """
    将numpy图像转换为base64字符串
    
    Args:
        image: OpenCV图像(BGR格式)
        
    Returns:
        base64编码的字符串
    """
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"


def base64_to_image(base64_str: str) -> np.ndarray:
    """
    将base64字符串转换为numpy图像
    
    Args:
        base64_str: base64编码的图像字符串
        
    Returns:
        OpenCV图像(BGR格式)
    """
    # 移除data:image/jpeg;base64,前缀(如果存在)
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return image


def format_statistics(behavior_counts: Dict[str, int], total_detections: int) -> Dict:
    """
    格式化统计数据
    
    Args:
        behavior_counts: 各行为计数
        total_detections: 总检测数
        
    Returns:
        格式化的统计信息
    """
    percentages = {}
    for behavior, count in behavior_counts.items():
        if total_detections > 0:
            percentages[behavior] = round((count / total_detections) * 100, 2)
        else:
            percentages[behavior] = 0.0
    
    return {
        'counts': behavior_counts,
        'percentages': percentages,
        'total': total_detections
    }


def calculate_duration(behavior_frames: Dict[str, List[int]], fps: int) -> Dict[str, float]:
    """
    计算每种行为出现的总时长(秒)
    
    Args:
        behavior_frames: 各行为出现的帧列表
        fps: 视频帧率
        
    Returns:
        各行为的时长字典
    """
    durations = {}
    for behavior, frames in behavior_frames.items():
        if frames:
            # 估算时长(帧数 / 帧率)
            duration = len(frames) / fps
            durations[behavior] = round(duration, 2)
        else:
            durations[behavior] = 0.0
    
    return durations


def get_video_info(video_path: str) -> Dict:
    """
    获取视频基本信息
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        视频信息字典
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': int(cap.get(cv2.CAP_PROP_FPS)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / int(cap.get(cv2.CAP_PROP_FPS))
    }
    
    cap.release()
    return info


def cleanup_old_files(directory: str, max_age_hours: int = 24):
    """
    清理旧文件
    
    Args:
        directory: 目录路径
        max_age_hours: 最大保留时间(小时)
    """
    if not os.path.exists(directory):
        return
    
    current_time = datetime.now()
    deleted_count = 0
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        if os.path.isfile(filepath):
            file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            age_hours = (current_time - file_time).total_seconds() / 3600
            
            if age_hours > max_age_hours:
                try:
                    os.remove(filepath)
                    deleted_count += 1
                    logger.info(f"删除旧文件: {filepath}")
                except Exception as e:
                    logger.error(f"删除文件失败: {filepath}, 错误: {e}")
    
    if deleted_count > 0:
        logger.info(f"清理完成: 删除了 {deleted_count} 个文件")
