# 驾驶员行为实时检测系统

<p align="center">
  <img src="https://img.shields.io/badge/YOLOv8-Detection-blue" alt="YOLOv8">
  <img src="https://img.shields.io/badge/FastAPI-Backend-green" alt="FastAPI">
  <img src="https://img.shields.io/badge/Python-3.8+-orange" alt="Python">
</p>

基于YOLOv8n的驾驶员行为实时检测系统，支持视频上传检测和实时摄像头检测两种模式，可以检测三种危险驾驶行为：

- 🚬 **抽烟 (Smoke)**
- 📱 **使用手机 (Phone)**  
- 🥤 **喝水 (Drink)**

## 功能特性

✨ **视频上传检测**: 上传视频文件，自动标注并生成统计报告  
📸 **实时摄像头检测**: 使用摄像头进行实时行为检测  
📊 **统计分析**: 详细的行为统计和可视化报告  
🎨 **现代化UI**: 精美的暗色主题界面，流畅的动画效果  
⚡ **高性能**: 基于YOLOv8n模型，快速准确的检测

## 系统架构

```
┌─────────────────┐
│   前端界面       │  HTML + CSS + JavaScript
│  (视频/摄像头)   │
└────────┬────────┘
         │ HTTP API
         ↓
┌─────────────────┐
│  FastAPI后端    │  Python + FastAPI
│  (视频处理)     │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  YOLOv8n模型    │  Ultralytics
│  (行为检测)     │
└─────────────────┘
```

## 环境要求

- **Python**: 3.8 或更高版本
- **浏览器**: Chrome, Firefox, Safari (推荐Chrome)
- **摄像头**: 用于实时检测功能
- **GPU**: 可选，用于加速推理

## 快速开始

### 1. 安装后端依赖

```bash
cd backend
pip install -r requirements.txt
```

### 2. 放置模型文件

将训练好的YOLOv8模型权重文件 `best.pt` 放在 `backend/models/` 目录下：

```
backend/models/best.pt
```

### 3. 启动后端服务

```bash
cd backend
python app.py
```

后端服务将在 `http://localhost:8000` 启动

### 4. 打开前端界面

使用浏览器打开 `frontend/index.html` 文件，或使用本地服务器：

```bash
# 使用Python启动简单HTTP服务器
cd frontend
python -m http.server 8080
```

然后访问 `http://localhost:8080`

## 使用说明

### 视频上传检测

1. 切换到"视频上传检测"标签页
2. 点击"选择文件"或拖拽视频文件到上传区域
3. 支持的格式：MP4, AVI, MOV, MKV
4. 点击"开始检测"按钮
5. 等待处理完成，查看检测结果和统计报告

### 实时摄像头检测

1. 切换到"实时摄像头检测"标签页
2. 点击"启动摄像头"按钮
3. 允许浏览器访问摄像头
4. 系统将自动进行实时检测并显示结果
5. 点击"停止检测"结束检测

## API文档

### 健康检查

```http
GET /api/health
```

返回API服务状态

### 上传视频检测

```http
POST /api/upload-video
Content-Type: multipart/form-data

file: <video_file>
confidence: 0.25 (可选)
```

返回检测结果和统计信息

### 单帧检测

```http
POST /api/detect-frame
Content-Type: application/json

{
  "image": "data:image/jpeg;base64,...",
  "confidence": 0.25
}
```

返回检测框和标注图像

### 下载处理后的视频

```http
GET /api/download/{filename}
```

下载标注后的视频文件

## 项目结构

```
detect_driver/
├── backend/                    # 后端代码
│   ├── app.py                 # FastAPI主应用
│   ├── detection.py           # YOLOv8检测引擎
│   ├── utils.py               # 工具函数
│   ├── requirements.txt       # Python依赖
│   ├── models/
│   │   └── best.pt           # 模型权重文件
│   ├── uploads/              # 上传的视频 (自动创建)
│   └── outputs/              # 输出的视频 (自动创建)
├── frontend/                  # 前端代码
│   ├── index.html            # 主页面
│   ├── style.css             # 样式表
│   └── app.js                # JavaScript逻辑
└── README.md                 # 本文档
```

## 配置说明

### 后端配置

在 `backend/app.py` 中可以修改：

- `API端口`: 默认 8000
- `模型路径`: 默认 `models/best.pt`
- `置信度阈值`: 默认 0.25

### 前端配置

在 `frontend/app.js` 中可以修改：

- `API_BASE_URL`: 后端API地址
- `检测间隔`: 实时检测的帧率

## 性能优化

### 提高检测速度

- 使用GPU加速（需要CUDA环境）
- 调整 `process_every_n_frames` 参数（跳帧处理）
- 降低输入视频分辨率

### 提高检测精度

- 调整置信度阈值 `conf_threshold`
- 使用更大的模型（YOLOv8s/m/l）
- 增加训练数据

## 常见问题

### Q: 模型加载失败怎么办？

A: 请确保：
1. `best.pt` 文件存在于 `backend/models/` 目录
2. 模型文件与YOLOv8n兼容
3. 已安装 `ultralytics` 包

### Q: 摄像头无法访问？

A: 请检查：
1. 浏览器是否允许摄像头权限
2. 使用HTTPS或localhost访问
3. 摄像头是否被其他程序占用

### Q: 视频处理速度很慢？

A: 可以尝试：
1. 调整 `process_every_n_frames` 参数增加跳帧
2. 降低视频分辨率
3. 使用GPU加速

### Q: 检测结果不准确？

A: 建议：
1. 调整置信度阈值
2. 确保光照条件良好
3. 检查模型训练质量

## 技术栈

**后端**:
- FastAPI - 现代高性能Web框架
- Ultralytics - YOLOv8实现
- OpenCV - 视频处理
- NumPy - 数值计算

**前端**:
- HTML5 - 页面结构
- CSS3 - 现代化样式
- JavaScript - 交互逻辑
- Canvas API - 图像绘制
- MediaDevices API - 摄像头访问

## 许可证

本项目仅供学习和研究使用。

## 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - 目标检测模型
- [FastAPI](https://fastapi.tiangolo.com/) - Web框架

---

**开发团队**: 模式识别课程大作业  
**联系方式**: 如有问题请联系项目组成员
