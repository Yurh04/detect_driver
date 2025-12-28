/**
 * 驾驶员行为检测系统 - 前端JavaScript
 * 实现视频上传和实时摄像头检测功能
 */

// ==================== 配置 ====================
const API_BASE_URL = 'http://localhost:8000/api';

// ==================== 全局变量 ====================
let cameraStream = null;
let detectionInterval = null;
let realtimeStats = { Smoke: 0, Phone: 0, Drink: 0 };

// ==================== DOM元素 ====================
const elements = {
    // Navigation
    navItems: document.querySelectorAll('.nav-item'),
    viewContents: document.querySelectorAll('.view-content'),

    // Video Upload
    uploadArea: document.getElementById('upload-area'),
    videoFile: document.getElementById('video-file'),
    selectFileBtn: document.getElementById('select-file-btn'),
    fileInfo: document.getElementById('file-info'),
    filename: document.getElementById('filename'),
    uploadBtn: document.getElementById('upload-btn'),
    progressBar: document.getElementById('progress-bar'),
    progressFill: document.getElementById('progress-fill'),
    progressText: document.getElementById('progress-text'),
    uploadResults: document.getElementById('upload-results'),
    resultVideo: document.getElementById('result-video'),
    uploadStatistics: document.getElementById('upload-statistics'),

    // Realtime Monitor
    startCameraBtn: document.getElementById('start-camera'),
    stopCameraBtn: document.getElementById('stop-camera'),
    cameraVideo: document.getElementById('camera-video'),
    detectionCanvas: document.getElementById('detection-canvas'),
    statSmoke: document.getElementById('stat-smoke'),
    statPhone: document.getElementById('stat-phone'),
    statDrink: document.getElementById('stat-drink')
};

// ==================== 初始化 ====================
document.addEventListener('DOMContentLoaded', () => {
    initTabSwitching();
    initVideoUpload();
    initRealtimeDetection();
});

// ==================== Tab切换功能 ====================
function initTabSwitching() {
    elements.navItems.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.dataset.tab;

            // Update Menu
            elements.navItems.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Update Content
            elements.viewContents.forEach(content => {
                content.classList.remove('active');
            });
            document.getElementById(`${tabId}-tab`).classList.add('active');

            // Stop camera if leaving realtime
            if (tabId !== 'realtime' && cameraStream) {
                stopCamera();
            }
        });
    });
}

// ==================== 视频上传功能 ====================
function initVideoUpload() {
    // 点击选择文件
    elements.selectFileBtn.addEventListener('click', () => {
        elements.videoFile.click();
    });

    // 文件选择
    elements.videoFile.addEventListener('change', handleFileSelect);

    // 拖拽上传
    elements.uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.uploadArea.classList.add('dragover');
    });

    elements.uploadArea.addEventListener('dragleave', () => {
        elements.uploadArea.classList.remove('dragover');
    });

    elements.uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.uploadArea.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            elements.videoFile.files = files;
            handleFileSelect();
        }
    });

    // 开始检测
    elements.uploadBtn.addEventListener('click', uploadAndDetect);
}

function handleFileSelect() {
    const file = elements.videoFile.files[0];
    if (file) {
        elements.filename.textContent = file.name;
        elements.fileInfo.style.display = 'block';
        elements.uploadResults.style.display = 'none';
    }
}

async function uploadAndDetect() {
    const file = elements.videoFile.files[0];
    if (!file) {
        alert('请先选择视频文件');
        return;
    }

    // 显示进度条
    elements.progressBar.style.display = 'block';
    elements.uploadBtn.disabled = true;
    elements.progressFill.style.width = '0%';
    elements.progressText.textContent = '正在上传...';

    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('confidence', '0.25');

        // 模拟上传进度
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += 5;
            if (progress <= 90) {
                elements.progressFill.style.width = progress + '%';
            }
        }, 200);

        const response = await fetch(`${API_BASE_URL}/upload-video`, {
            method: 'POST',
            body: formData
        });

        clearInterval(progressInterval);

        if (!response.ok) {
            throw new Error('上传失败');
        }

        const result = await response.json();

        elements.progressFill.style.width = '100%';
        elements.progressText.textContent = '处理完成!';

        // 延迟显示结果
        setTimeout(() => {
            displayUploadResults(result);
            elements.progressBar.style.display = 'none';
            elements.uploadBtn.disabled = false;
        }, 500);

    } catch (error) {
        console.error('错误:', error);
        alert('视频处理失败: ' + error.message);
        elements.progressBar.style.display = 'none';
        elements.uploadBtn.disabled = false;
    }
}

function displayUploadResults(result) {
    // 显示结果视频
    if (result.output_video) {
        const videoUrl = `${API_BASE_URL.replace('/api', '')}/api/download/${result.output_video}`;
        elements.resultVideo.src = videoUrl;
        // 尝试自动播放
        elements.resultVideo.play().catch(e => console.log('自动播放需要用户交互'));
    }

    // 显示统计信息
    const stats = result.statistics;
    let statsHtml = '';

    // 动态生成各类别的统计
    for (const [name, count] of Object.entries(stats.counts)) {
        // 如果是通用模型，可能有几十个类别，只显示检测到的
        const percentage = stats.percentages[name] || 0;
        if (count > 0 || ['Smoke', 'Phone', 'Drink'].includes(name)) {
            statsHtml += `
                <div class="stat-item">
                    <span class="stat-label">🏷️ ${name}:</span>
                    <span class="stat-value">${count} 次 (${percentage}%)</span>
                </div>
            `;
        }
    }

    // 添加基础信息
    statsHtml += `
        <div class="stat-item">
            <span class="stat-label">⏱️ 视频时长:</span>
            <span class="stat-value">${stats.video_duration} 秒</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">📊 总检测数:</span>
            <span class="stat-value">${stats.total}</span>
        </div>
    `;

    elements.uploadStatistics.innerHTML = statsHtml;
    elements.uploadResults.style.display = 'block';
    elements.uploadResults.scrollIntoView({ behavior: 'smooth' });
}

// ==================== 实时检测功能 ====================
function initRealtimeDetection() {
    elements.startCameraBtn.addEventListener('click', startCamera);
    elements.stopCameraBtn.addEventListener('click', stopCamera);
}

async function startCamera() {
    try {
        // 请求摄像头权限
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 1280, height: 720 }
        });

        elements.cameraVideo.srcObject = cameraStream;

        // 设置canvas尺寸
        elements.cameraVideo.addEventListener('loadedmetadata', () => {
            elements.detectionCanvas.width = elements.cameraVideo.videoWidth;
            elements.detectionCanvas.height = elements.cameraVideo.videoHeight;
        });

        // 切换按钮
        elements.startCameraBtn.style.display = 'none';
        elements.stopCameraBtn.style.display = 'block';

        // 开始检测
        startDetection();

    } catch (error) {
        console.error('摄像头访问失败:', error);
        alert('无法访问摄像头，请检查权限设置');
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }

    if (detectionInterval) {
        clearInterval(detectionInterval);
        detectionInterval = null;
    }

    elements.cameraVideo.srcObject = null;
    elements.startCameraBtn.style.display = 'block';
    elements.stopCameraBtn.style.display = 'none';

    // 清空canvas
    const ctx = elements.detectionCanvas.getContext('2d');
    ctx.clearRect(0, 0, elements.detectionCanvas.width, elements.detectionCanvas.height);
}

function startDetection() {
    // 每0.5秒检测一次
    detectionInterval = setInterval(async () => {
        await detectCurrentFrame();
    }, 500);
}

async function detectCurrentFrame() {
    try {
        // 从video捕获当前帧
        const canvas = document.createElement('canvas');
        canvas.width = elements.cameraVideo.videoWidth;
        canvas.height = elements.cameraVideo.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(elements.cameraVideo, 0, 0);

        // 转换为base64
        const imageBase64 = canvas.toDataURL('image/jpeg', 0.8);

        // 发送到后端检测
        const response = await fetch(`${API_BASE_URL}/detect-frame`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: imageBase64,
                confidence: 0.25
            })
        });

        if (!response.ok) {
            throw new Error('检测请求失败');
        }

        const result = await response.json();

        if (result.success) {
            // 绘制检测框
            drawDetections(result.detections);

            // 更新统计
            updateRealtimeStats(result.detections);
        }

    } catch (error) {
        console.error('检测错误:', error);
    }
}

function drawDetections(detections) {
    const canvas = elements.detectionCanvas;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const colors = {
        0: '#f43f5e', // Smoke - Red
        1: '#10b981', // Phone - Green
        2: '#0ea5e9'  // Drink - Blue
    };

    detections.forEach(det => {
        const [x1, y1, x2, y2] = det.bbox;
        const color = colors[det.class_id] || '#cbd5e1';
        const w = x2 - x1;
        const h = y2 - y1;

        // 1. Precision Reticle (Corners)
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        const len = Math.min(w, h) * 0.15;

        // Draw 4 corners
        drawCorner(ctx, x1, y1, len, len, 0); // TL
        drawCorner(ctx, x2, y1, -len, len, 0); // TR
        drawCorner(ctx, x1, y2, len, -len, 0); // BL
        drawCorner(ctx, x2, y2, -len, -len, 0); // BR

        // 2. Subtle Fill
        ctx.fillStyle = hexToRgba(color, 0.05);
        ctx.fillRect(x1, y1, w, h);

        // 3. Technical Data Overlay
        const label = det.class_name.toUpperCase();
        const conf = (det.confidence * 100).toFixed(0) + '%';

        ctx.font = 'bold 12px "Space Grotesk", monospace';
        const labelWidth = ctx.measureText(label).width;

        // Lead-line and label
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x1 - 10, y1 - 10);
        ctx.lineTo(x1 - 15, y1 - 10);
        ctx.stroke();

        ctx.fillStyle = color;
        ctx.fillText(label, x1 - 15 - labelWidth, y1 - 14);

        ctx.fillStyle = '#f8fafc';
        ctx.font = 'normal 10px "Space Grotesk", monospace';
        ctx.fillText(conf, x1 - 15 - ctx.measureText(conf).width, y1 - 2);
    });
}

function drawCorner(ctx, x, y, w, h) {
    ctx.beginPath();
    ctx.moveTo(x, y + h);
    ctx.lineTo(x, y);
    ctx.lineTo(x + w, y);
    ctx.stroke();
}

function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function updateRealtimeStats(detections) {
    // We only update if something is detected to avoid flickering to 0 too fast
    // or keep the current session counts
    detections.forEach(det => {
        realtimeStats[det.class_name]++;
    });

    // Update display with formatting
    elements.statSmoke.textContent = realtimeStats.Smoke.toLocaleString();
    elements.statPhone.textContent = realtimeStats.Phone.toLocaleString();
    elements.statDrink.textContent = realtimeStats.Drink.toLocaleString();
}

// ==================== 工具函数 ====================
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}
