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

app = FastAPI(title="RDRDS API", debug=True)

# ---------------------------------------------------------
# [에러 핸들러]
# ---------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_msg = traceback.format_exc()
    logger.error(f"🚨 서버 에러 발생:\n{error_msg}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": str(exc), "traceback": error_msg}
    )

# 1. 모델 경로 설정 및 분석기 초기화
BASE_DIR = Path(__file__).resolve().parents[1]
yolo_p = BASE_DIR / "models" / "yolo26n.pt"
vggt_p = BASE_DIR / "models" / "model_tracker_fixed_e20.pt"

try:
    analyzer = DrivingAnalyzer(yolo_p, vggt_p)
    
    # 🔥 [핵심 수정 사항] FastVGGT의 chunk_size 에러 방지
    # 모델 내부의 chunk_size가 0이 되지 않도록 강제로 설정합니다.
    if analyzer and hasattr(analyzer, 'vggt'):
        # DrivingAnalyzer 내부에 vggt 모델 인스턴스가 있다면 접근
        try:
            # 보통 vggt.model.chunk_size 또는 vggt.chunk_size에 위치합니다.
            # 이 값을 1024 정도로 설정하면 range(0, num, 1024)가 되어 에러가 해결됩니다.
            if hasattr(analyzer.vggt, 'model'):
                analyzer.vggt.model.chunk_size = 1024
            else:
                analyzer.vggt.chunk_size = 1024
            logger.info("🛠 FastVGGT chunk_size를 1024로 강제 설정 완료")
        except Exception as patch_e:
            logger.warning(f"⚠️ chunk_size 패치 실패 (무시 가능): {patch_e}")

    logger.info("🚀 분석 엔진 로드 완료 (T4 GPU 활성)")
except Exception as e:
    logger.error(f"❌ 엔진 로드 실패:\n{traceback.format_exc()}")
    analyzer = None

# [SRS 6.1] 백그라운드 로깅
def log_high_risk_event(results: list):
    high_risk_objs = [obj for obj in results if obj['risk'] >= 80.0]
    if high_risk_objs:
        logger.warning(f"⚠️ HIGH RISK DETECTED: {high_risk_objs}")

@app.post("/analyze")
async def predict_risk(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not analyzer:
        raise HTTPException(status_code=503, detail="분석 엔진이 준비되지 않았습니다.")

    try:
        request_content = await file.read()
        img_pil = Image.open(io.BytesIO(request_content)).convert("RGB")
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        timestamp = datetime.now().timestamp()
        results = analyzer.analyze_frame(frame, timestamp)

        is_warning = any(obj['risk'] >= 80.0 for obj in results)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)