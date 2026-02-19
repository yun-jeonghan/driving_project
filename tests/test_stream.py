import requests
import time
import cv2
import numpy as np
import concurrent.futures
import os
from pathlib import Path

API_URL = "http://localhost:8000/analyze/visualize"
FPS = 10.0

def process_scenario(folder_path: Path, client_id: int):
    # 가상 클라이언트 구분을 위해 ID 부여
    client_name = f"Client_{client_id}_{folder_path.name}"
    print(f"🚀 [{client_name}] 스트림 시작")
    
    images = sorted(folder_path.glob("*.jpg"), key=lambda x: x.name)
    if not images: 
        print(f"⚠️ [{client_name}] 이미지가 없습니다.")
        return

    # 비디오 저장 설정 (Client ID별로 별도 저장)
    os.makedirs("runs", exist_ok=True)
    sample_img = cv2.imread(str(images[0]))
    h, w, _ = sample_img.shape
    output_path = f"runs/{client_name}_result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, FPS, (w, h))

    for i, img_p in enumerate(images):
        start_time = time.time()
        try:
            with open(img_p, "rb") as f:
                resp = requests.post(API_URL, files={"file": f})
            
            if resp.status_code == 200:
                nparr = np.frombuffer(resp.content, np.uint8)
                res_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                out.write(res_frame)
                
                latency = (time.time() - start_time) * 1000
                # 10프레임마다 로그 출력 (너무 많으면 보기 힘듦)
                if (i + 1) % 10 == 0:
                    print(f"[{client_name}] Frame {i+1:<3} | Latency: {latency:6.1f}ms")
            else:
                print(f"❌ [{client_name}] Error: {resp.status_code}")
        except Exception as e:
            print(f"🔌 [{client_name}] Connection Fail: {e}")
        
        # 10 FPS 유지를 위한 대기
        time.sleep(max(0, (1/FPS) - (time.time() - start_time)))

    out.release()
    print(f"🏁 [{client_name}] 완료 -> {output_path}")

def run_concurrent_test(base_dir: str, num_clients: int = 3):
    base_path = Path(base_dir)
    video_folders = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("video")])

    if not video_folders:
        print("❌ 테스트할 소스 폴더(video*)가 없습니다.")
        return

    print(f"🔥 총 {num_clients}개의 가상 클라이언트를 가동합니다. (소스: {video_folders[0].name})")
    
    # ThreadPoolExecutor를 사용하여 실제 병렬 실행
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_clients) as executor:
        # 동일한 폴더를 사용하더라도 client_id를 다르게 주어 병렬 실행
        futures = [executor.submit(process_scenario, video_folders[0], i+1) for i in range(num_clients)]
        concurrent.futures.wait(futures)

if __name__ == "__main__":
    # 여기서 num_clients를 3으로 설정하면 3명이 동시에 쏘는 효과가 납니다.
    run_concurrent_test("tests", num_clients=3)