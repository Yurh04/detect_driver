"""
下载精度更高的YOLOv8s模型
"""
from ultralytics import YOLO
import os
import shutil

print("正在下载YOLOv8s (Small) 模型，精度比Nano更好...")

# 下载YOLOv8s模型
model = YOLO('yolov8s.pt')

# 确保models目录存在
os.makedirs('models', exist_ok=True)

source_path = 'yolov8s.pt'
target_path = 'models/best.pt'

# 备份旧模型
if os.path.exists(target_path):
    print("备份旧模型到 models/best_nano.pt")
    shutil.move(target_path, 'models/best_nano.pt')

# 复制新模型
if os.path.exists(source_path):
    shutil.copy(source_path, target_path)
    print(f"✓ 模型已更新为 YOLOv8s: {target_path}")
else:
    print("❌ 下载失败")
