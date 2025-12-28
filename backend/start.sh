#!/bin/bash
# 启动脚本 - 激活虚拟环境并启动FastAPI服务

echo "🚀 启动驾驶员行为检测系统后端服务..."
echo ""

# 激活虚拟环境
source venv/bin/activate

# 检查模型文件是否存在
if [ ! -f "models/best.pt" ]; then
    echo "⚠️  警告: 模型文件 models/best.pt 不存在"
    echo "请将训练好的模型文件放置到 models/ 目录下"
    echo ""
    read -p "是否继续启动服务？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "✓ 虚拟环境已激活"
echo "✓ 正在启动FastAPI服务..."
echo ""
echo "服务地址: http://localhost:8000"
echo "API文档: http://localhost:8000/docs"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

# 启动服务
python app.py
