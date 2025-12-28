#!/bin/bash
# 启动脚本 - 启动前端HTTP服务

PORT=3000

echo "🚀 启动驾驶员行为检测系统前端界面..."
echo ""
echo "后端服务默认运行在端口: 8000"
echo "前端服务将运行在端口: $PORT"
echo ""

# 检查端口是否占用
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  端口 $PORT 被占用，正在尝试清理..."
    lsof -ti:$PORT | xargs kill -9
    sleep 1
    echo "✓ 已清理占用端口的进程"
fi

echo "正在打开浏览器..."
# 仅在macOS上执行open，如果是Linux可以用xdg-open (这里假设用户是mac)
open "http://localhost:$PORT" 2>/dev/null || true

echo "前端运行中..."
echo "按 Ctrl+C 停止服务"
echo ""

# 使用Python内置HTTP服务器启动，绑定到IPv4
python3 -m http.server $PORT --bind 127.0.0.1
