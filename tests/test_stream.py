import requests
import time
import os
import json
from pathlib import Path

# 서버 설정
API_URL = "http://localhost:8000/analyze"
INTERVAL = 0.1  # 10 FPS

def run_scenario_test(base_dir: str):
    base_path = Path(base_dir)
    # 'video'로 시작하는 하위 폴더 찾기
    video_folders = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("video")])
    
    if not video_folders:
        print(f"❌ 테스트할 비디오 폴더를 찾을 수 없습니다. (경로: {base_path.absolute()})")
        return

    for folder in video_folders:
        print(f"\n🎬 시나리오 시작: {folder.name}")
        print("-" * 60)
        
        # 이미지 파일 정렬
        images = sorted(folder.glob("*.jpg"), key=lambda x: x.name)
        if not images:
            print(f"⚠️ {folder.name}에 JPG 이미지가 없습니다.")
            continue
        
        for i, img_p in enumerate(images):
            start_time = time.time()
            
            try:
                with open(img_p, "rb") as f:
                    response = requests.post(API_URL, files={"file": f})
                
                latency = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    res = response.json()
                    status = "✅" if latency <= 150 else "⚠️ SLOW"
                    obj_count = res.get('data', {}).get('object_count', 0)
                    print(f"[{folder.name}] Frame {i+1:<3} | {latency:6.1f}ms | {status} | Objs: {obj_count}")
                else:
                    print(f"\n🔥 [서버 에러 {response.status_code}] 발생!")
                    try:
                        err_json = response.json()
                        print(f"에러 메시지: {err_json.get('message')}")
                        print("-" * 30)
                        print("상세 Traceback:")
                        print(err_json.get('traceback'))
                        print("-" * 30)
                    except:
                        print(f"응답 전문: {response.text}")
                    
                    print("\n🛑 에러 분석을 위해 테스트를 중단합니다.")
                    return # 원인 파악을 위해 첫 에러에서 멈춤

            except Exception as e:
                print(f"🔌 통신 실패: {e}")
                return

            # FPS 유지
            sleep_time = max(0, INTERVAL - (time.time() - start_time))
            time.sleep(sleep_time)
        
        print(f"🏁 {folder.name} 시나리오 종료")

if __name__ == "__main__":
    # 프로젝트 루트의 tests 폴더 실행
    run_scenario_test("tests")