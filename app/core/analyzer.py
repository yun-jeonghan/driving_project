import sys
import os
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any

# [SRS 4.1.2] 시계열 분석을 위한 프레임 버퍼 클래스 (FIFO Queue)
class VideoBuffer:
    def __init__(self, size: int = 5):
        self.size = size
        self.buffer = []

    def push(self, frame: np.ndarray):
        """새 프레임을 넣고, 사이즈를 초과하면 가장 오래된 프레임 삭제"""
        self.buffer.append(frame)
        if len(self.buffer) > self.size:
            self.buffer.pop(0)

    def is_ready(self) -> bool:
        """[REQ-FUNC-201] 버퍼가 가득 찼는지 확인"""
        return len(self.buffer) == self.size

    def get_all(self) -> List[np.ndarray]:
        return self.buffer

# [SRS 4.1.3] YOLOv8 객체 탐지 및 거리 추정 핸들러
class YOLOHandler:
    def __init__(self, path: Path):
        from ultralytics import YOLO
        self.model = YOLO(str(path))

    def track(self, frame: np.ndarray):
        """[REQ-FUNC-101] 실시간 객체 추적 수행 (ID 유지)"""
        return self.model.track(frame, persist=True, verbose=False)[0]

# [SRS 4.2.3] FastVGGT 깊이 및 맥락 분석 핸들러
class FastVGGTHandler:
    def __init__(self, path: Path, device: str):
        self.device = device
        # [SRS 2.4] FastVGGT 레포지토리 경로 추가
        REPO_PATH = Path(__file__).resolve().parents[2] / "FastVGGT_repo"
        if str(REPO_PATH) not in sys.path:
            sys.path.insert(0, str(REPO_PATH))
        
        try:
            from vggt.models.vggt import VGGT
            self.model = VGGT()
            checkpoint = torch.load(str(path), map_location=self.device)
            state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device).eval()
            print("✅ FastVGGT 모델 로드 완료")
        except Exception as e:
            print(f"❌ FastVGGT 로드 실패: {e}")
            self.model = None

    @torch.no_grad()
    def get_depth_map(self, frame: np.ndarray) -> np.ndarray:
        """[SRS 3.3] 전처리 후 깊이 맵(Depth Map) 생성"""
        if self.model is None:
            return np.ones((frame.shape[0], frame.shape[1])) * 50.0

        # 전처리: 518x392 해상도로 리사이징 (모델 최적화 규격)
        img_input = cv2.resize(frame, (518, 392))
        img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).float().to(self.device) / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            output = self.model(img_tensor)
        
        # 출력 텐서 파싱
        if isinstance(output, dict):
            depth = output.get('depth', list(output.values())[0])
        elif isinstance(output, (list, tuple)):
            depth = output[0]
        else:
            depth = output
            
        return depth.squeeze().float().cpu().numpy() if torch.is_tensor(depth) else depth

# [SRS 4.1] 통합 주행 분석기 (Orchestrator)
class DrivingAnalyzer:
    def __init__(self, yolo_path: Path, vggt_path: Path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo = YOLOHandler(yolo_path)
        self.vggt = FastVGGTHandler(vggt_path, self.device)
        self.buffer = VideoBuffer(size=5) # [REQ-FUNC-201] 5프레임 슬라이딩 윈도우
        self.history = {} # 객체별 추적 데이터 저장: {id: {'dist': d, 'time': t}}

    def analyze_frame(self, frame: np.ndarray, timestamp: float) -> List[Dict[str, Any]]:
        """[SRS 4.1] 실시간 프레임 분석 및 위험 점수 산출"""
        self.buffer.push(frame) # 버퍼 업데이트
        
        # 1. 객체 탐지 및 추적
        yolo_res = self.yolo.track(frame)
        
        # 2. 깊이 맵 생성 및 리사이징
        depth_map = self.vggt.get_depth_map(frame)
        depth_map_resized = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
        
        frame_report = []
        if yolo_res.boxes is not None:
            boxes = yolo_res.boxes.xyxy.cpu().numpy()
            clss = yolo_res.boxes.cls.cpu().numpy().astype(int)
            # YOLO ID가 없을 경우 -1 할당
            ids = yolo_res.boxes.id.cpu().numpy().astype(int) if yolo_res.boxes.id is not None else [-1] * len(boxes)

            for box, obj_id, cls in zip(boxes, ids, clss):
                x1, y1, x2, y2 = map(int, box)
                
                # ROI(객체 영역) 평균 거리 계산
                roi_depth = depth_map_resized[y1:y2, x1:x2]
                curr_dist = np.mean(roi_depth) if roi_depth.size > 0 else 50.0
                
                # [REQ-FUNC-202] 상대 속도 계산: v = (이전 거리 - 현재 거리) / 시간 변화량
                velocity = 0.0
                if obj_id != -1 and obj_id in self.history:
                    prev = self.history[obj_id]
                    delta_t = timestamp - prev['time']
                    if delta_t > 0:
                        velocity = (prev['dist'] - curr_dist) / delta_t
                
                # [SRS 4.2.2] 위험 점수 산출 (Risk Score Formula)
                # 점수 = (기준 거리 10m 가중치) + (상대 속도 가중치)
                risk_score = min(100.0, (10.0 / (curr_dist + 1e-6)) * 10 + (max(0, velocity) * 15))
                
                # [SRS 5.2] 위험 단계 설정 (트리거 기준 80점)
                alert = "DANGER" if risk_score >= 80.0 else "WARNING" if risk_score >= 50.0 else "NORMAL"

                # 히스토리 업데이트 (ID 추적이 가능할 때만)
                if obj_id != -1:
                    self.history[obj_id] = {'dist': curr_dist, 'time': timestamp}
                
                frame_report.append({
                    "id": int(obj_id),
                    "label": self.yolo.model.names[cls],
                    "dist_m": round(float(curr_dist), 2),
                    "velocity_mps": round(float(velocity), 2),
                    "risk": round(risk_score, 1),
                    "alert": alert,
                    "bbox": [x1, y1, x2, y2]
                })
        return frame_report

    def draw_results(self, frame: np.ndarray, results: List[Dict]) -> np.ndarray:
        """[SRS 3.1] 시각적 경고 레이어 오버레이"""
        annotated = frame.copy()
        for res in results:
            x1, y1, x2, y2 = res['bbox']
            # 위험 등급별 색상 (BGR)
            color = (0, 0, 255) if res['alert'] == "DANGER" else (0, 165, 255) if res['alert'] == "WARNING" else (0, 255, 0)
            
            # 바운딩 박스
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # 정보 라벨링
            label_text = f"ID:{res['id']} {res['dist_m']}m"
            risk_text = f"Risk: {res['risk']}"
            cv2.putText(annotated, label_text, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(annotated, risk_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return annotated