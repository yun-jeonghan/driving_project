import sys
import os
import torch
import numpy as np
import cv2
from pathlib import Path

# 1. 경로 설정
CORE_DIR = Path(__file__).resolve().parent
BASE_DIR = CORE_DIR.parents[1] 
REPO_PATH = BASE_DIR / "FastVGGT_repo"
MODELS_DIR = BASE_DIR / "models"

if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))

# 2. [핵심] 런타임 몽키 패치 (원본 파일 수정 X)
try:
    import merging.merge as vggt_merge
    original_func = vggt_merge.fast_similarity_chunks

    def patched_fast_similarity_chunks(a, b, chunk_size, *args, **kwargs):
        # chunk_size가 0이거나 너무 작으면 강제로 1024로 할당
        safe_chunk_size = chunk_size if (chunk_size and chunk_size > 0) else 1024
        return original_func(a, b, safe_chunk_size, *args, **kwargs)

    # 메모리상에서 함수 교체
    vggt_merge.fast_similarity_chunks = patched_fast_similarity_chunks
    print("✅ FastVGGT 런타임 패치 적용 완료 (원본 보존)")
except Exception as e:
    print(f"⚠️ 패치 적용 실패 (무시 가능): {e}")

try:
    from vggt.models.vggt import VGGT
except ImportError:
    print("❌ 레포지토리 경로를 찾을 수 없습니다.")
    sys.exit(1)

class DrivingAnalyzer:
    def __init__(self, yolo_path, vggt_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        from ultralytics import YOLO
        
        self.detector = YOLO(str(yolo_path))
        self.vggt = VGGT()
        
        checkpoint = torch.load(str(vggt_path), map_location=self.device)
        state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        self.vggt.load_state_dict(state_dict, strict=False)
        self.vggt.to(self.device).eval()

        self.history = {}  # {object_id: {'dist': d, 'time': t}} 형태 저장
        print("✅ 모델 로드 및 장치 할당 완료")

    @torch.no_grad()
    def _get_depth_map(self, frame):
        # 1. 전처리 (Dust3r/FastVGGT 최적화 해상도)
        target_w, target_h = 518, 392
        img_input = cv2.resize(frame, (target_w, target_h))
        img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).float().to(self.device) / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        
        # 2. 추론
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            output = self.vggt(img_tensor)
        
        # 3. [에러 해결 지점] 결과 파싱 (or 연산자 제거)
        if isinstance(output, dict):
            # 'depth' 키가 있으면 가져오고, 없으면 첫 번째 밸류를 가져옵니다.
            depth = output.get('depth')
            if depth is None:
                depth = list(output.values())[0]
        elif isinstance(output, (list, tuple)):
            # 리스트로 반환될 경우 첫 번째 아이템 선택
            depth = output[0]
        else:
            depth = output
            
        # 4. 후처리: 텐서라면 numpy로 변환
        if torch.is_tensor(depth):
            return depth.squeeze().float().cpu().numpy()
        return depth # 이미 numpy인 경우

    def analyze(self, frame):
        yolo_res = self.detector(frame, verbose=False)[0]
        depth_map = self._get_depth_map(frame)
        depth_map_resized = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))

        results = []
        for box in yolo_res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = self.detector.names[int(box.cls[0])]
            
            roi_depth = depth_map_resized[y1:y2, x1:x2]
            avg_depth = np.mean(roi_depth) if roi_depth.size > 0 else 50.0
            
            # 위험도 계산: 기준 거리 10m
            risk = round(10.0 / (avg_depth + 1e-6), 4)
            
            # 경보 등급
            alert = "NORMAL"
            if risk >= 1.5: alert = "DANGER"
            elif risk >= 0.8: alert = "WARNING"
            
            results.append({
                "label": label, "dist_m": round(float(avg_depth), 2),
                "risk": risk, "alert": alert, "bbox": [x1, y1, x2, y2]
            })
        return results

    def draw_results(self, frame, results):
        """분석 결과(BBox, ID, 거리, 속도, 리스크)를 이미지에 그립니다."""
        annotated_frame = frame.copy()
        
        for res in results:
            # 좌표 및 기본 정보 추출
            x1, y1, x2, y2 = res['bbox']
            obj_id = res.get('id', '?')
            label = res['label']
            dist = res['dist_m']
            velocity = res.get('velocity_mps', 0.0) # 속도 정보가 없으면 0 처리
            risk = res['risk']
            alert = res['alert']
            
            # 경보 레벨에 따른 색상 설정 (BGR)
            color = (0, 255, 0) # Green (Normal)
            thickness = 2
            if alert == "DANGER": 
                color = (0, 0, 255) # Red
                thickness = 3
            elif alert == "WARNING": 
                color = (0, 165, 255) # Orange
            
            # 1. 바운딩 박스 그리기
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # 2. 텍스트 정보 구성 (ID, 라벨, 거리, 속도)
            # 예: [ID:1] car | 12.5m | -3.2m/s (멀어짐)
            vel_str = f"{velocity:+.1f}m/s" if velocity != 0 else "Stable"
            label_text = f"[ID:{obj_id}] {label} | {dist}m | {vel_str}"
            
            # 3. 리스크 점수 텍스트
            risk_text = f"Risk: {risk} ({alert})"
            
            # 텍스트 배경 박스 그리기 (가독성 확보)
            (tw1, th1), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            (tw2, th2), _ = cv2.getTextSize(risk_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - th1 - th2 - 15), (x1 + max(tw1, tw2), y1), color, -1)
            
            # 텍스트 쓰기 (흰색 글씨)
            cv2.putText(annotated_frame, label_text, (x1, y1 - th1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(annotated_frame, risk_text, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame
    
    def calculate_velocity(self, obj_id, current_dist, current_time):
        """이전 기록과 비교하여 상대 속도(m/s) 계산"""
        if obj_id not in self.history:
            self.history[obj_id] = {'dist': current_dist, 'time': current_time}
            return 0.0
        
        prev = self.history[obj_id]
        delta_d = prev['dist'] - current_dist  # 양수면 가까워지는 중
        delta_t = current_time - prev['time']
        
        velocity = delta_d / delta_t if delta_t > 0 else 0
        
        # 기록 업데이트
        self.history[obj_id] = {'dist': current_dist, 'time': current_time}
        return velocity
    
    def analyze_video_frame(self, frame, timestamp):
        # 1. 일단 탐지(Detection)를 먼저 수행합니다. (트래킹에만 의존 X)
        results = self.detector.track(frame, persist=True, verbose=False)[0]
        
        depth_map = self._get_depth_map(frame)
        depth_map_resized = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
        frame_report = []

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)
            # YOLO가 준 ID가 있으면 쓰고, 없으면 -1로 둡니다.
            ids = results.boxes.id.cpu().numpy().astype(int) if results.boxes.id is not None else [-1] * len(boxes)

            for i, (box, obj_id, cls) in enumerate(zip(boxes, ids, clss)):
                x1, y1, x2, y2 = map(int, box)
                label = self.detector.names[cls]
                center_now = ((x1 + x2) / 2, (y1 + y2) / 2)

                # [핵심] 1초 간격 대응 강제 매칭 로직
                # YOLO가 ID를 못 줬거나(-1), 새로 부여했더라도 우리가 히스토리와 대조합니다.
                best_match_id = obj_id
                min_dist = 200 # 1초 동안 이동 가능한 픽셀 거리 (화면 크기에 따라 조절)

                for old_id, old_data in self.history.items():
                    if old_data['label'] == label:
                        # 유클리드 거리 계산
                        dist = np.linalg.norm(np.array(center_now) - np.array(old_data['center']))
                        if dist < min_dist:
                            min_dist = dist
                            best_match_id = old_id
                
                # 최종 결정된 ID (새로 나타난 놈이면 새로운 고유 ID 부여)
                if best_match_id == -1:
                    # 히스토리에 없는 완전히 새로운 객체라면 현재 루프에서 가장 큰 ID + 1 부여
                    new_id = max(self.history.keys()) + 1 if self.history else 1
                    obj_id = new_id
                else:
                    obj_id = best_match_id

                # 거리 및 속도 계산
                roi_depth = depth_map_resized[y1:y2, x1:x2]
                curr_dist = np.mean(roi_depth) if roi_depth.size > 0 else 50.0

                velocity = 0.0
                if obj_id in self.history:
                    prev_data = self.history[obj_id]
                    delta_d = prev_data['dist'] - curr_dist
                    delta_t = timestamp - prev_data['time']
                    if delta_t > 0:
                        velocity = delta_d / delta_t
                
                # 히스토리 업데이트 (다음 프레임을 위해 저장)
                self.history[obj_id] = {
                    'dist': curr_dist, 
                    'time': timestamp, 
                    'center': center_now,
                    'label': label
                }

                # 리스크 점수 (속도가 음수면 멀어지는 것이므로 0 처리)
                risk = round((10.0 / (curr_dist + 1e-6)) + (max(0, velocity) * 0.7), 4)
                alert = "DANGER" if risk >= 20.0 else "WARNING" if risk >= 10.0 else "NORMAL"

                frame_report.append({
                    "id": int(obj_id),
                    "label": label,
                    "dist_m": round(float(curr_dist), 2),
                    "velocity_mps": round(float(velocity), 2),
                    "risk": risk,
                    "alert": alert,
                    "bbox": [x1, y1, x2, y2]
                })

        return frame_report

if __name__ == "__main__":
    yolo_file = MODELS_DIR / "yolo26n.pt"
    vggt_file = MODELS_DIR / "model_tracker_fixed_e20.pt"

    try:
        analyzer = DrivingAnalyzer(yolo_file, vggt_file)
        
        # 테스트: 인터넷에서 실제 도로/차량 이미지를 가져옵니다.
        import requests
        from PIL import Image
        from io import BytesIO
        
        url = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg"
        response = requests.get(url)
        test_img = np.array(Image.open(BytesIO(response.content)))
        test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR) # OpenCV 형식으로 변환

        print("📸 실제 이미지 분석 시작...")
        res = analyzer.analyze(test_img)
        
        if not res:
            print("🤔 객체가 탐지되지 않았습니다. 모델 경로를 확인하세요.")
        else:
            print("🚀 [분석 성공] 결과:")
            for obj in res:
                print(f" - {obj['label']}: 거리 {obj['dist_m']}m | 위험도 {obj['risk']} | 등급 [{obj['alert']}]")
                
    except Exception as e:
        import traceback
        print(f"❌ 최종 에러 상세:\n{traceback.format_exc()}")