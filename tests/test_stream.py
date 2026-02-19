import requests
import time
import os
from pathlib import Path

# [SRS 3.3] 서버 주소
API_URL = "http://localhost:8000/analyze"
# [SRS 5.1] 목표 10 FPS (0.1초 간격)
INTERVAL = 0.1 

def run_scenario_test(base_dir: str):
    base_path = Path(base_dir)
    # 1. 'video'로 시작하는 하위 폴더들 찾기
    video_folders = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("video")])
    
    if not video_folders:
        print("❌ 테스트할 비디오 폴더를 찾을 수 없습니다.")
        return

    for folder in video_folders:
        print(f"\n🎬 시나리오 시작: {folder.name}")
        print("-" * 50)
        
        # 2. 폴더 내 이미지들을 이름순으로 정렬 (시간 순서 보장)
        # 팁: 파일명이 1, 2, 3... 이라면 정렬 로직이 중요합니다.
        images = sorted(folder.glob("*.jpg"), key=lambda x: x.name)
        
        for i, img_p in enumerate(images):
            start_time = time.time()
            
            with open(img_p, "rb") as f:
                # [SRS 3.4] 비동기 스트림 시뮬레이션
                response = requests.post(API_URL, files={"file": f})
            
            latency = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                res = response.json()
                # [SRS 5.1] 150ms 이내 응답 확인
                status = "✅" if latency <= 150 else "⚠️ SLOW"
                print(f"[{folder.name}] Frame {i+1:<3} | {latency:6.1f}ms | {status} | Buffer: {res['data']['object_count']} objs")
            else:
                print(f"❌ Error at {img_p.name}: {response.status_code}")

            # 10 FPS 유지를 위해 대기
            time.sleep(max(0, INTERVAL - (time.time() - start_time)))
        
        print(f"🏁 {folder.name} 시나리오 종료")

if __name__ == "__main__":
    # 'tests' 폴더 내의 비디오 시나리오 실행
    run_scenario_test("tests")