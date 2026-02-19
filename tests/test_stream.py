import requests
import time
import cv2
import numpy as np
import concurrent.futures
from pathlib import Path

API_URL = "http://localhost:8000/analyze/visualize"
FPS = 10.0

def process_scenario(folder_path: Path):
    scenario_name = folder_path.name
    print(f"🚀 [Client-{scenario_name}] 시작")
    
    images = sorted(folder_path.glob("*.jpg"), key=lambda x: x.name)
    if not images: return

    # 첫 프레임으로 비디오 사이즈 결정
    sample_img = cv2.imread(str(images[0]))
    h, w, _ = sample_img.shape
    
    # 비디오 라이터 설정
    output_path = f"runs/{scenario_name}_result.mp4"
    os.makedirs("runs", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, FPS, (w, h))

    for i, img_p in enumerate(images):
        start_time = time.time()
        try:
            with open(img_p, "rb") as f:
                resp = requests.post(API_URL, files={"file": f})
            
            if resp.status_code == 200:
                # 바이너리 이미지를 numpy 배열로 변환
                nparr = np.frombuffer(resp.content, np.uint8)
                res_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                out.write(res_frame)
                
                latency = (time.time() - start_time) * 1000
                print(f"[{scenario_name}] Frame {i+1:<3} | Latency: {latency:6.1f}ms")
            else:
                print(f"❌ [{scenario_name}] Error: {resp.status_code}")
        except Exception as e:
            print(f"🔌 [{scenario_name}] Connection Fail: {e}")
        
        # 10 FPS 보정
        time.sleep(max(0, (1/FPS) - (time.time() - start_time)))

    out.release()
    print(f"🏁 [Client-{scenario_name}] 완료 -> {output_path}")

def run_concurrent_test(base_dir: str, num_clients: int = 3):
    """
    여러 시나리오 폴더를 동시에 실행하여 비동기 대응 능력을 확인합니다.
    """
    base_path = Path(base_dir)
    # 테스트할 폴더들 (video1, video2 등)
    video_folders = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("video")])[:num_clients]

    print(f"🔥 {len(video_folders)}개의 클라이언트가 동시에 요청을 보냅니다...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(video_folders)) as executor:
        executor.map(process_scenario, video_folders)

if __name__ == "__main__":
    import os
    # 시나리오가 1개뿐이라면 동일 폴더를 여러 번 호출하도록 수정 가능
    run_concurrent_test("tests", num_clients=2)