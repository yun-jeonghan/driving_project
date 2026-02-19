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

# [SRS 6.1] 시스템 관리자용 고위험 상황 로깅 설정
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
    description="[SRS 1.1] FastAPI 기반 실시간 주행 위험 탐지 시스템 (Visualization 통합본)",
    debug=True
)

# [강력한 에러 핸들러] 모든 서버 에러를 가로채서 상세히 출력
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_msg = traceback.format_exc()
    logger.error(f"🚨 서버 치명적 에러 발생:\n{error_msg}")
    sys.stdout.flush() 
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": str(exc),
            "traceback": error_msg
        }
    )

# 1. 모델 경로 설정 및 분석기 초기화
BASE_DIR = Path(__file__).resolve().parents[1]
yolo_p = BASE_DIR / "models" / "yolo26n.pt"
vggt_p = BASE_DIR / "models" / "model_tracker_fixed_e20.pt"

# 서버 시작 시 모델 로드
try:
    analyzer = DrivingAnalyzer(yolo_p, vggt_p)
    logger.info("🚀 분석 엔진 및 시각화 모듈 로드 완료 (T4 GPU 활성)")
except Exception as e:
    logger.error(f"❌ 엔진 로드 실패:\n{traceback.format_exc()}")
    analyzer = None

# [SRS 6.1] 고위험 상황 발생 시 백그라운드 로깅 함수
def log_high_risk_event(results: list):
    high_risk_objs = [obj for obj in results if obj['risk'] >= 80.0]
    if high_risk_objs:
        logger.warning(f"⚠️ HIGH RISK DETECTED: {high_risk_objs}")

@app.get("/")
def read_root():
    """시스템 상태 확인 엔드포인트"""
    return {
        "status": "online", 
        "engine": "Active" if analyzer else "Inactive",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/analyze")
async def predict_risk(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    [SRS 3.3] 실시간 프레임 분석 엔드포인트
    - 비동기 파일 읽기를 통해 I/O 지연을 최소화합니다.
    """
    if not analyzer:
        raise HTTPException(status_code=503, detail="분석 엔진이 준비되지 않았습니다.")

    try:
        request_content = await file.read()
        img_pil = Image.open(io.BytesIO(request_content)).convert("RGB")
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # 분석 수행
        timestamp = datetime.now().timestamp()
        results = analyzer.analyze_frame(frame, timestamp)

        # [SRS 4.4] 위험 감지 시 트리거 판단
        is_warning = any(obj['risk'] >= 80.0 for obj in results)
        
        # [SRS 6.1] 고위험 로그 백그라운드 처리
        if is_warning:
            background_tasks.add_task(log_high_risk_event, results)

        return JSONResponse(content={
            "status": "success",
            "is_warning": is_warning,
            "timestamp": timestamp,
            "data": {
                "detections": results,
                "object_count": len(results)
            }
        })
    except Exception as e:
        raise e

@app.post("/analyze/visualize")
async def analyze_and_visualize(file: UploadFile = File(...)):
    """
    [SRS 3.1] 시각화 결과 반환 엔드포인트
    - 분석된 객체와 위험도가 그려진 이미지를 직접 반환합니다.
    """
    if not analyzer:
        raise HTTPException(status_code=503, detail="Engine Inactive")

    try:
        data = await file.read()
        img_pil = Image.open(io.BytesIO(data)).convert("RGB")
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # 분석 수행
        results = analyzer.analyze_frame(frame, datetime.now().timestamp())
        
        # [SRS 3.1] 시각적 경고 레이어 합성
        vis_frame = analyzer.draw_results(frame, results)
        
        # 결과를 PNG 포맷으로 인코딩하여 스트리밍 반환
        _, im_png = cv2.imencode(".png", vis_frame)
        return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")
    except Exception as e:
        raise e

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)