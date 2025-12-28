"""
驾驶员行为检测系统 - FastAPI后端
提供视频上传检测和实时帧检测API
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import os
import logging
from typing import Optional

from detection import DetectionEngine
from utils import (
    save_uploaded_file, 
    image_to_base64, 
    base64_to_image,
    format_statistics,
    calculate_duration,
    ensure_dir,
    cleanup_old_files
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="驾驶员行为检测API",
    description="基于YOLOv8的驾驶员行为实时检测系统",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化检测引擎
try:
    detection_engine = DetectionEngine(model_path="models/best.pt")
    logger.info("检测引擎初始化成功")
except Exception as e:
    logger.error(f"检测引擎初始化失败: {e}")
    detection_engine = None

# 创建必要的目录
ensure_dir("uploads")
ensure_dir("outputs")
ensure_dir("models")


# Pydantic模型
class DetectFrameRequest(BaseModel):
    """单帧检测请求"""
    image: str  # base64编码的图像
    confidence: Optional[float] = 0.25


class DetectionResponse(BaseModel):
    """检测响应"""
    success: bool
    message: Optional[str] = None
    detections: list = []
    annotated_image: Optional[str] = None
    statistics: Optional[dict] = None


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "驾驶员行为检测API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/health",
            "upload_video": "/api/upload-video",
            "detect_frame": "/api/detect-frame"
        }
    }


@app.get("/api/health")
async def health_check():
    """健康检查"""
    model_status = "ready" if detection_engine else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "message": "API运行正常"
    }


@app.post("/api/upload-video")
async def upload_video(
    file: UploadFile = File(...),
    confidence: float = 0.25
):
    """
    上传视频进行检测
    
    Args:
        file: 上传的视频文件
        confidence: 置信度阈值
        
    Returns:
        检测结果和统计信息
    """
    if detection_engine is None:
        raise HTTPException(status_code=503, detail="检测引擎未初始化")
    
    # 检查文件类型
    allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"不支持的文件格式。支持的格式: {', '.join(allowed_extensions)}"
        )
    
    try:
        # 保存上传的文件
        content = await file.read()
        video_path = save_uploaded_file(content, file.filename, "uploads")
        
        # 生成输出视频路径
        output_filename = f"annotated_{os.path.basename(video_path)}"
        output_path = os.path.join("outputs", output_filename)
        
        logger.info(f"开始处理视频: {video_path}")
        
        # 如果是通用模型(COCO)，强制降低阈值以提高检出率
        if 'person' in detection_engine.class_names:
            confidence = 0.15
            
        # 处理视频
        result = detection_engine.process_video(
            video_path=video_path,
            output_path=output_path,
            conf_threshold=confidence,
            process_every_n_frames=2  # 每2帧处理一次以提高速度
        )
        
        if result['success']:
            # 计算详细统计
            video_info = result['video_info']
            stats = result['statistics']
            
            # 计算时长
            durations = calculate_duration(
                stats['behavior_frames'],
                video_info['fps']
            )
            
            # 格式化统计数据
            formatted_stats = format_statistics(
                stats['behavior_counts'],
                stats['total_detections']
            )
            formatted_stats['durations'] = durations
            formatted_stats['video_duration'] = round(
                video_info['total_frames'] / video_info['fps'], 2
            )
            
            # 清理旧文件
            cleanup_old_files("uploads", max_age_hours=2)
            cleanup_old_files("outputs", max_age_hours=2)
            
            return JSONResponse(content={
                "success": True,
                "message": "视频处理完成",
                "video_info": video_info,
                "statistics": formatted_stats,
                "output_video": output_filename,
                "detections": result['detections']
            })
        else:
            raise HTTPException(status_code=500, detail="视频处理失败")
    
    except Exception as e:
        logger.error(f"视频处理错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.post("/api/detect-frame")
async def detect_frame(request: DetectFrameRequest):
    """
    检测单帧图像(用于实时摄像头检测)
    
    Args:
        request: 包含base64图像的请求
        
    Returns:
        检测结果
    """
    if detection_engine is None:
        raise HTTPException(status_code=503, detail="检测引擎未初始化")
    
    try:
        # 将base64转换为图像
        image = base64_to_image(request.image)
    
        # 如果是通用模型，降低阈值
        conf = request.confidence
        if 'person' in detection_engine.class_names:
            conf = 0.15
            
        # 检测
        result = detection_engine.detect_image(image, conf)
        
        if result['success']:
            # 绘制检测框
            annotated_image = detection_engine.draw_detections(
                image, 
                result['detections']
            )
            
            # 转换为base64
            annotated_base64 = image_to_base64(annotated_image)
            
            # 统计信息
            behavior_counts = {}
            for det in result['detections']:
                name = det['class_name']
                behavior_counts[name] = behavior_counts.get(name, 0) + 1
            
            return JSONResponse(content={
                "success": True,
                "detections": result['detections'],
                "annotated_image": annotated_base64,
                "statistics": format_statistics(
                    behavior_counts, 
                    result['num_detections']
                )
            })
        else:
            return JSONResponse(content={
                "success": False,
                "error": result.get('error', 'Unknown error'),
                "detections": []
            })
    
    except Exception as e:
        logger.error(f"帧检测错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")


@app.get("/api/download/{filename}")
async def download_video(filename: str):
    """
    下载处理后的视频
    
    Args:
        filename: 文件名
        
    Returns:
        视频文件
    """
    filepath = os.path.join("outputs", filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="文件不存在")
    
    return FileResponse(
        filepath,
        media_type="video/mp4",
        filename=filename
    )


# 启动命令
if __name__ == "__main__":
    import uvicorn
    
    logger.info("启动驾驶员行为检测API服务器...")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
