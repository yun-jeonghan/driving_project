import os
import io
import logging
import uuid
import cv2
import numpy as np
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from PIL import Image
from pathlib import Path
from app.core.analyzer import DrivingAnalyzer

# [SRS 6.1] 시스템 관리자 로그 설정
logging.basicConfig(filename='admin_monitor.log', level=logging.INFO)
logger = logging.getLogger("RDRDS_Admin")

app = FastAPI(title="RDRDS API", description="[SRS 1.1] 실시간 주행 위험 탐지 시스템")

# 모델 초기화
BASE_DIR = Path(__file__).resolve().parents[1]
yolo_p = BASE_DIR / "models" / "yolo26n.pt"
vggt_p = BASE_DIR / "models" / "model_tracker_fixed_e20.pt"

try:
    analyzer = DrivingAnalyzer(yolo_p, vggt_p)
    logger.info("Analysis Engine successfully loaded on T4 GPU.")
except Exception as e:
    logger.error(f"Engine Load Failed: {e}")
    analyzer = None

@app.post("/analyze")
async def predict_risk(file: UploadFile = File(...)):
    """[SRS 3.3] 단일 프레임 분석 엔드포인트"""
    if not analyzer:
        raise HTTPException(status_code=503, detail="Engine Inactive")

    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        timestamp = datetime.now().timestamp()
        results = analyzer.analyze_frame(frame, timestamp)
        
        # [SRS 5.2] 위험 발생 시 트리거 (Trigger)
        is_warning = any(obj['risk'] >= 80.0 for obj in results)
        if is_warning:
            logger.warning(f"HIGH RISK EVENT: {results}")

        return JSONResponse(content={
            "status": "success",
            "is_warning": is_warning,
            "max_risk": max([obj['risk'] for obj in results]) if results else 0,
            "detections": results
        })
    except Exception as e:
        logger.error(f"Inference Error: {e}")
        return {"status": "error", "message": "Handle Inference Failure"}

@app.post("/analyze/visualize")
async def analyze_visualize(file: UploadFile = File(...)):
    """[SRS 3.1] 시각화 결과 직접 반환"""
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    results = analyzer.analyze_frame(frame, datetime.now().timestamp())
    vis_frame = analyzer.draw_results(frame, results)
    
    _, im_png = cv2.imencode(".png", vis_frame)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)