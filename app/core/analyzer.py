import sys
import os
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any

# [SRS 4.1.2] 시계열 분석을 위한 프레임 버퍼 클래스
class VideoBuffer:
    def __init__(self, size: int = 5):
        self.size = size
        self.buffer = []

    def push(self, frame: np.ndarray):
        self.buffer.append(frame)
        if len(self.buffer) > self.size:
            self.buffer.pop(0)

    def is_full(self) -> bool:
        return len(self.buffer) == self.size

    def get_batch(self) -> List[np.ndarray]:
        return self.buffer

# [SRS 4.1.3] YOLO 객체 탐지 및 거리 추정 핸들러
class YOLOHandler:
    def __init__(self, path: Path):
        from ultralytics import YOLO
        self.model = YOLO(str(path))

    def track(self, frame: np.ndarray):
        # [REQ-FUNC-101] 실시간 추적을 위해 persist=True 활성화
        return self.model.track(frame, persist=True, verbose=False)[0]

# [SRS 4.2.3] FastVGGT 깊이 및 맥락 분석 핸들러
class FastVGGTHandler:
    def __init__(self, path: Path, device: str):
        self.device = device
        # [SRS 2.4] FastVGGT 레포지토리 경로 패치
        REPO_PATH = Path(__file__).resolve().parents[2] / "FastVGGT_repo"
        if str(REPO_PATH) not in sys.path:
            sys.path.insert(0, str(REPO_PATH))
        
        from vggt.models.vggt import VGGT
        self.model = VGGT()
        checkpoint = torch.load(str(path), map_location=self.device)
        state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def get_depth_map(self, frame: np.ndarray) -> np.ndarray:
        # [SRS 3.3] 전처리: 518x392 해상도로 리사이징 및 텐서 변환
        img_input = cv2.resize(frame, (518, 392))
        img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).float().to(self.device) / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            output = self.model(img_tensor)
        
        if isinstance(output, dict):
            depth = output.get('depth', list(output.values())[0])
        elif isinstance(output, (list, tuple)):
            depth = output[0]
        else:
            depth = output
            
        return depth.squeeze().float().cpu().numpy() if torch.is_tensor(depth) else depth

# [SRS 4.1] 통합 분석 엔진 (Orchestrator)
class DrivingAnalyzer:
    def __init__(self, yolo_path: Path, vggt_path: Path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo = YOLOHandler(yolo_path)
        self.vggt = FastVGGTHandler(vggt_path, self.device)
        self.buffer = VideoBuffer(size=5) # [REQ-FUNC-201]
        self.history = {} # {id: {'dist': d, 'time': t, 'center': c}}

    def analyze_frame(self, frame: np.ndarray, timestamp: float) -> List[Dict[str, Any]]:
        self.buffer.push(frame)
        yolo_res = self.yolo.track(frame)
        depth_map = self.vggt.get_depth_map(frame)
        depth_map_resized = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
        
        frame_report = []
        if yolo_res.boxes is not None:
            boxes = yolo_res.boxes.xyxy.cpu().numpy()
            clss = yolo_res.boxes.cls.cpu().numpy().astype(int)
            ids = yolo_res.boxes.id.cpu().numpy().astype(int) if yolo_res.boxes.id is not None else [-1] * len(boxes)

            for box, obj_id, cls in zip(boxes, ids, clss):
                x1, y1, x2, y2 = map(int, box)
                roi_depth = depth_map_resized[y1:y2, x1:x2]
                curr_dist = np.mean(roi_depth) if roi_depth.size > 0 else 50.0
                
                # [REQ-FUNC-202] 속도 계산: v = delta_d / delta_t
                velocity = 0.0
                if obj_id != -1 and obj_id in self.history:
                    prev = self.history[obj_id]
                    delta_t = timestamp - prev['time']
                    if delta_t > 0:
                        velocity = (prev['dist'] - curr_dist) / delta_t
                
                # [SRS 4.2.2] 위험 점수 산출 공식
                # Risk = (10 / distance) * 5 + (velocity * 10)
                risk_score = min(100.0, (10.0 / (curr_dist + 1e-6)) * 5 + (max(0, velocity) * 10))
                alert = "DANGER" if risk_score >= 80.0 else "WARNING" if risk_score >= 50.0 else "NORMAL"

                self.history[obj_id] = {'dist': curr_dist, 'time': timestamp, 'label': self.yolo.model.names[cls]}
                
                frame_report.append({
                    "id": int(obj_id), "label": self.yolo.model.names[cls],
                    "dist_m": round(float(curr_dist), 2), "velocity_mps": round(float(velocity), 2),
                    "risk": round(risk_score, 1), "alert": alert, "bbox": [x1, y1, x2, y2]
                })
        return frame_report

    def draw_results(self, frame: np.ndarray, results: List[Dict]) -> np.ndarray:
        # [SRS 3.1] 시각적 경고 인터페이스 구현
        annotated = frame.copy()
        for res in results:
            x1, y1, x2, y2 = res['bbox']
            color = (0, 0, 255) if res['alert'] == "DANGER" else (0, 165, 255) if res['alert'] == "WARNING" else (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{res['id']} {res['dist_m']}m {res['risk']}pts"
            cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return annotated