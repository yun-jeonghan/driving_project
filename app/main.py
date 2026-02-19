import os
import io
import cv2
import logging
import numpy as np
import sys
import traceback
from datetime import datetime
from pathlib import Path
from PIL import Image

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, StreamingResponse
from app.core.analyzer import DrivingAnalyzer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('admin_monitor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("RDRDS_Admin")

app = FastAPI(
    title="RDRDS API", 
    description="실시간 주행 위험 탐지 시스템 (Visualization 포함)",
    debug=True
)

# [에러 핸들러] 상세 Traceback 반환
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_msg = traceback.format_exc()
    logger.error(f"🚨 서버 에러 발생:\n{error_msg}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": str(exc), "traceback": error_msg}
    )

# 모델 로드
BASE_DIR = Path(__file__).resolve().parents[1]
yolo_p = BASE_DIR / "models" / "yolo26n.pt"
vggt_p = BASE_DIR / "models" / "model_tracker_fixed_e20.pt"

try:
    analyzer = DrivingAnalyzer(yolo_p, vggt_p)
    logger.info("🚀 분석 엔진 및 시각화 모듈 로드 완료")
except Exception as e:
    logger.error(f"❌ 엔진 로드 실패:\n{traceback.format_exc()}")
    analyzer = None

def log_high_risk_event(results: list):
    high_risk_objs = [obj for obj in results if obj['risk'] >= 80.0]
    if high_risk_objs:
        logger.warning(f"⚠️ HIGH RISK: {high_risk_objs}")

@app.get("/")
def read_root():
    return {"status": "online", "engine": "Active" if analyzer else "Inactive"}

# [엔드포인트 1] 데이터 분석 전용
@app.post("/analyze")
async def predict_risk(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not analyzer:
        raise HTTPException(status_code=503, detail="Engine Inactive")
    try:
        content = await file.read()
        img_pil = Image.open(io.BytesIO(content)).convert("RGB")
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        timestamp = datetime.now().timestamp()
        results = analyzer.analyze_frame(frame, timestamp)
        
        is_warning = any(obj['risk'] >= 80.0 for obj in results)
        if is_warning:
            background_tasks.add_task(log_high_risk_event, results)

        return JSONResponse(content={
            "status": "success", "is_warning": is_warning,
            "data": {"detections": results, "object_count": len(results)}
        })
    except Exception as e:
        raise e

# [엔드포인트 2] 시각화 결과 반환 (여기가 404 원인이었음)
@app.post("/analyze/visualize")
async def analyze_and_visualize(file: UploadFile = File(...)):
    if not analyzer:
        raise HTTPException(status_code=503, detail="Engine Inactive")
    try:
        content = await file.read()
        img_pil = Image.open(io.BytesIO(content)).convert("RGB")
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # 분석 및 시각화 수행
        results = analyzer.analyze_frame(frame, datetime.now().timestamp())
        vis_frame = analyzer.draw_results(frame, results)
        
        # PNG 스트리밍 응답
        _, im_png = cv2.imencode(".png", vis_frame)
        return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")
    except Exception as e:
        raise e

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)