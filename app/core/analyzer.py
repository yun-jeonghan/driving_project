import sys
import os
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any

class VideoBuffer:
    def __init__(self, size: int = 5):
        self.size = size
        self.buffer = []
    def push(self, frame: np.ndarray):
        self.buffer.append(frame)
        if len(self.buffer) > self.size: self.buffer.pop(0)
    def is_ready(self) -> bool: return len(self.buffer) == self.size
    def get_all(self) -> List[np.ndarray]: return self.buffer

class YOLOHandler:
    def __init__(self, path: Path):
        from ultralytics import YOLO
        self.model = YOLO(str(path))
    def track(self, frame: np.ndarray):
        return self.model.track(frame, persist=True, verbose=False)[0]

class FastVGGTHandler:
    def __init__(self, path: Path, device: str):
        self.device = device
        REPO_PATH = Path(__file__).resolve().parents[2] / "FastVGGT_repo"
        if str(REPO_PATH) not in sys.path: sys.path.insert(0, str(REPO_PATH))
        
        # [Fix] chunk_size 0 방어 코드
        merge_file = REPO_PATH / "merging" / "merge.py"
        if merge_file.exists():
            with open(merge_file, "r") as f: content = f.read()
            if "range(0, num_src, chunk_size)" in content:
                new_content = content.replace("range(0, num_src, chunk_size)", "range(0, num_src, (chunk_size if chunk_size > 0 else 1024))")
                with open(merge_file, "w") as f: f.write(new_content)

        from vggt.models.vggt import VGGT
        self.model = VGGT()
        checkpoint = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(checkpoint.get('model', checkpoint), strict=False)
        self.model.to(self.device).eval()
        for m in self.model.modules():
            if hasattr(m, 'chunk_size'): m.chunk_size = 1024

    @torch.no_grad()
    def get_depth_map(self, frame: np.ndarray) -> np.ndarray:
        img_input = cv2.resize(frame, (518, 392))
        img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).float().to(self.device) / 255.0
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            output = self.model(img_tensor.unsqueeze(0))
        depth = output.get('depth', list(output.values())[0]) if isinstance(output, dict) else output
        return depth.squeeze().float().cpu().numpy()

class DrivingAnalyzer:
    def __init__(self, yolo_path: Path, vggt_path: Path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo = YOLOHandler(yolo_path)
        self.vggt = FastVGGTHandler(vggt_path, self.device)
        self.history = {}

    def analyze_frame(self, frame: np.ndarray, timestamp: float) -> List[Dict[str, Any]]:
        yolo_res = self.yolo.track(frame)
        depth_map = self.vggt.get_depth_map(frame)
        depth_map_res = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
        
        frame_report = []
        if hasattr(yolo_res, 'boxes') and yolo_res.boxes is not None:
            boxes = yolo_res.boxes.xyxy.cpu().numpy()
            clss = yolo_res.boxes.cls.cpu().numpy().astype(int)
            ids = yolo_res.boxes.id.cpu().numpy().astype(int) if yolo_res.boxes.id is not None else [-1]*len(boxes)

            for box, obj_id, cls in zip(boxes, ids, clss):
                x1, y1, x2, y2 = map(int, box)
                roi = depth_map_res[max(0,y1):y2, max(0,x1):x2]
                curr_dist = np.mean(roi) if roi.size > 0 else 50.0
                
                velocity = 0.0
                if obj_id != -1 and obj_id in self.history:
                    prev = self.history[obj_id]
                    dt = timestamp - prev['time']
                    if dt > 0: velocity = (prev['dist'] - curr_dist) / dt
                
                risk = min(100.0, (10.0 / (curr_dist + 1e-6)) * 10 + (max(0, velocity) * 15))
                if obj_id != -1: self.history[obj_id] = {'dist': curr_dist, 'time': timestamp}
                
                frame_report.append({
                    "id": int(obj_id), "label": self.yolo.model.names[cls],
                    "dist_m": round(float(curr_dist), 2), "velocity_mps": round(float(velocity), 2),
                    "risk": round(risk, 1), "bbox": [x1, y1, x2, y2]
                })
        return frame_report

    def draw_results(self, frame: np.ndarray, results: List[Dict]) -> np.ndarray:
        annotated = frame.copy()
        for res in results:
            x1, y1, x2, y2 = res['bbox']
            # 위험도에 따른 색상 변경
            color = (0,0,255) if res['risk'] >= 80 else (0,165,255) if res['risk'] >= 50 else (0,255,0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # 레이블에 거리와 속도 모두 표시
            label = f"ID:{res['id']} {res['dist_m']}m {res['velocity_mps']}m/s"
            cv2.putText(annotated, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(annotated, f"Risk: {res['risk']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return annotated