# 快速启动指南

## ✅ 环境已配置完成

虚拟环境已创建并安装好所有依赖！

## 📋 下一步操作

### 1. 获取模型文件

**重要：** 从同学那里获取训练好的YOLOv8模型权重文件 `best.pt`

将文件放置到：
```
backend/models/best.pt
```

### 2. 启动后端服务

**方式1 - 使用启动脚本（推荐）：**
```bash
cd backend
./start.sh
```

**方式2 - 手动启动：**
```bash
cd backend
source venv/bin/activate
python app.py
```

后端服务将在 `http://localhost:8000` 启动

### 3. 打开前端界面

**方式1 - 直接打开HTML文件：**
```bash
open frontend/index.html
```

**方式2 - 使用HTTP服务器（推荐）：**
```bash
cd frontend
python3 -m http.server 8080
# 然后在浏览器访问 http://localhost:8080
```

## 🎯 功能测试

### 测试视频上传检测
1. 切换到"视频上传检测"标签页
2. 选择或拖拽视频文件
3. 点击"开始检测"
4. 查看检测结果和统计报告

### 测试实时摄像头检测
1. 切换到"实时摄像头检测"标签页
2. 点击"启动摄像头"
3. 允许浏览器访问摄像头
4. 观察实时检测效果

## 📚 更多信息

- 完整文档：查看 `README.md`
- API文档：访问 `http://localhost:8000/docs`
- 项目结构和详细说明：查看 walkthrough.md

## ⚠️ 常见问题

**Q: 模型加载失败？**
- 确保 `best.pt` 文件在 `backend/models/` 目录
- 检查文件权限

**Q: 摄像头无法访问？**
- 使用Chrome浏览器
- 确保通过 localhost 或 HTTPS 访问
- 检查浏览器摄像头权限设置

**Q: 端口被占用？**
- 修改 `backend/app.py` 中的端口号
- 或者关闭占用8000端口的其他程序

## 🚀 开始使用

现在你可以：
1. 放置模型文件 `best.pt` 到 `backend/models/`
2. 运行 `cd backend && ./start.sh` 启动后端
3. 打开前端界面开始检测！

祝使用愉快！🎉
