"""
下载YOLOv8n预训练模型作为测试模型
"""

from ultralytics import YOLO
import os

print("正在下载YOLOv8n预训练模型...")

# 下载YOLOv8n模型（如果不存在会自动下载）
model = YOLO('yolov8n.pt')

# 获取下载的模型路径
# Ultralytics会将模型下载到用户目录，我们需要复制到models目录
import shutil

# 确保models目录存在
os.makedirs('models', exist_ok=True)

# 模型会被下载到当前目录或缓存目录
source_path = 'yolov8n.pt'
target_path = 'models/best.pt'

if os.path.exists(source_path):
    shutil.copy(source_path, target_path)
    print(f"✓ 模型已复制到 {target_path}")
else:
    # 从缓存目录查找
    import torch
    from pathlib import Path
    cache_dir = Path.home() / '.cache' / 'ultralytics'
    yolo_model = cache_dir / 'yolov8n.pt'
    
    if yolo_model.exists():
        shutil.copy(str(yolo_model), target_path)
        print(f"✓ 模型已从缓存复制到 {target_path}")
    else:
        print("✗ 未找到模型文件")
        print("请检查下载是否成功")

print("\n模型信息:")
print(f"- 类别数: {model.model.yaml['nc']}")
print(f"- 类别名: {model.names}")
print("\n注意: 这是COCO数据集的预训练模型(80类)")
print("用于测试系统功能，检测效果可能与您的驾驶员行为数据集不同")
