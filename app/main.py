import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from app.core.analyzer import DrivingAnalyzer
from pathlib import Path
import uvicorn
import numpy as np
import io
import cv2
from PIL import Image
from fastapi.responses import StreamingResponse
from typing import List
import uuid

app = FastAPI(title="Driving Risk Analysis API", description="FastVGGT + YOLO26n 기반 주행 위험 분석")

# 1. 모델 경로 설정 및 분석기 초기화
BASE_DIR = Path(__file__).resolve().parents[1]
yolo_p = BASE_DIR / "models" / "yolo26n.pt"
vggt_p = BASE_DIR / "models" / "model_tracker_fixed_e20.pt"

# 서버 시작 시 모델을 메모리에 올립니다.
try:
    analyzer = DrivingAnalyzer(yolo_p, vggt_p)
    print("🚀 분석 엔진 로드 완료!")
except Exception as e:
    print(f"❌ 엔진 로드 실패: {e}")
    analyzer = None

@app.get("/")
def read_root():
    return {"message": "Driving Analyzer API is running", "engine": "Active" if analyzer else "Inactive"}

@app.post("/analyze")
async def analyze_frame(file: UploadFile = File(...)):
    if not analyzer:
        raise HTTPException(status_code=503, detail="분석 엔진이 준비되지 않았습니다.")

    try:
        # 2. 업로드된 이미지 처리
        request_object_content = await file.read()
        img_pil = Image.open(io.BytesIO(request_object_content)).convert("RGB")
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # 3. 분석 수행
        results = analyzer.analyze(frame)

        # 4. 전체 프레임 위험도 요약
        max_risk = max([obj['risk'] for obj in results]) if results else 0
        system_alert = "NORMAL"
        if any(obj['alert'] == "DANGER" for obj in results): system_alert = "DANGER"
        elif any(obj['alert'] == "WARNING" for obj in results): system_alert = "WARNING"

        return {
            "status": "success",
            "system_alert": system_alert,
            "max_risk_score": max_risk,
            "detections": results,
            "object_count": len(results)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/analyze/visualize")
async def analyze_and_visualize(file: UploadFile = File(...)):
    """이미지를 분석하고 시각화된 결과 이미지를 직접 반환"""
    data = await file.read()
    img_pil = Image.open(io.BytesIO(data)).convert("RGB")
    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    results = analyzer.analyze(frame)
    # 시각화 수행
    vis_frame = analyzer.draw_results(frame, results)
    
    # 결과를 다시 이미지 포맷으로 변환하여 반환
    _, im_png = cv2.imencode(".png", vis_frame)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")

@app.post("/analyze/batch")
async def analyze_batch(
    files: List[UploadFile] = File(...), 
    interval: float = Form(0.1)  # 기본값 0.1초 (10 FPS 가정)
):
    """
    여러 장의 이미지를 순차적 프레임으로 인식하여 처리
    interval: 각 이미지 사이의 시간 간격 (초 단위)
    """
    if not analyzer:
        raise HTTPException(status_code=503, detail="분석 엔진 로드 실패")

    batch_results = []
    current_time = 0.0

    # 배치 처리 시 이전 추적 기록을 초기화하고 싶다면 아래 주석 해제
    # analyzer.history = {} 

    for file in files:
        # 1. 파일 읽기 및 변환
        data = await file.read()
        try:
            img_pil = Image.open(io.BytesIO(data)).convert("RGB")
            frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            continue # 손상된 이미지는 건너뜀

        # 2. analyze_video_frame 호출 (추적 및 속도 계산 포함)
        res = analyzer.analyze_video_frame(frame, current_time)
        
        batch_results.append({
            "filename": file.filename,
            "timestamp": round(current_time, 3),
            "detections": res
        })
        
        # 3. 다음 프레임을 위한 시간 업데이트
        current_time += interval
        
    return {
        "status": "success", 
        "interval_used": interval,
        "results": batch_results
    }

@app.post("/analyze/batch/visualize")
async def analyze_batch_visualize(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...), 
    interval: float = Form(0.1)
):
    if not analyzer:
        raise HTTPException(status_code=503, detail="분석 엔진 로드 실패")
    
    # 임시 파일 경로
    temp_video_path = f"temp_output_{uuid.uuid4()}.mp4"
    video_writer = None
    current_time = 0.0
    analyzer.history = {}

    try:
        for file in files:
            data = await file.read()
            img_pil = Image.open(io.BytesIO(data)).convert("RGB")
            frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            # [해결 핵심] 첫 프레임의 크기를 기준으로 고정
            if video_writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = 1.0 / max(interval, 0.01)
                video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
                target_size = (width, height)

            # 모든 프레임을 동일 크기로 리사이즈하여 에러 방지
            frame_resized = cv2.resize(frame, target_size)
            
            # 분석 및 시각화
            results = analyzer.analyze_video_frame(frame_resized, current_time)
            vis_frame = analyzer.draw_results(frame_resized, results)
            
            # 비디오에 프레임 추가
            video_writer.write(vis_frame)
            current_time += interval

    except Exception as e:
        if video_writer: video_writer.release()
        if os.path.exists(temp_video_path): os.remove(temp_video_path)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if video_writer:
            video_writer.release()

    background_tasks.add_task(os.remove, temp_video_path)
    return FileResponse(temp_video_path, media_type="video/mp4", filename="driving_analysis.mp4")


@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    # 1. 업로드된 파일 임시 저장
    temp_path = "temp_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(temp_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_results = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 성능을 위해 3프레임당 1번만 분석 (초당 10번꼴)
        if frame_count % 3 == 0:
            timestamp = frame_count / fps
            res = analyzer.analyze_video_frame(frame, timestamp)
            total_results.append({"frame": frame_count, "timestamp": timestamp, "detections": res})
        
        frame_count += 1
    
    cap.release()
    os.remove(temp_path) # 임시 파일 삭제

    return {"status": "success", "video_analysis": total_results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)