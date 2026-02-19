import requests
import time
import os
import json
from pathlib import Path

# [설정] 일반 분석과 시각화 주소 구분
API_URL_DATA = "http://localhost:8000/analyze"
API_URL_VIS = "http://localhost:8000/analyze/visualize"
INTERVAL = 0.1 

def run_scenario_test(base_dir: str):
    base_path = Path(base_dir)
    video_folders = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("video")])
    
    if not video_folders:
        print(f"❌ '{base_dir}' 내에 테스트 폴더가 없습니다.")
        return

    # 결과 저장을 위한 폴더 생성
    output_root = Path("runs/visualize")
    output_root.mkdir(parents=True, exist_ok=True)

    for folder in video_folders:
        print(f"\n🎬 시나리오 시각화 테스트: {folder.name}")
        print("-" * 60)
        
        # 결과 저장 하위 폴더
        save_dir = output_root / folder.name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        images = sorted(folder.glob("*.jpg"), key=lambda x: x.name)
        
        for i, img_p in enumerate(images):
            start_time = time.time()
            
            try:
                # 1. 시각화 엔드포인트 호출 (StreamingResponse로 이미지를 받아옴)
                with open(img_p, "rb") as f:
                    response = requests.post(API_URL_VIS, files={"file": f})
                
                latency = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    # 2. 결과 이미지 저장
                    save_path = save_dir / f"res_{img_p.name}"
                    with open(save_path, "wb") as out_f:
                        out_f.write(response.content)
                    
                    status = "✅" if latency <= 150 else "⚠️ SLOW"
                    print(f"[{folder.name}] Frame {i+1:<3} | {latency:6.1f}ms | {status} | Saved: {save_path.name}")
                else:
                    print(f"❌ 에러 발생 ({img_p.name}): {response.status_code}")
                    print(response.text)

            except Exception as e:
                print(f"🔌 통신 실패: {e}")
                return

            # 속도 조절
            time.sleep(max(0, INTERVAL - (time.time() - start_time)))
        
        print(f"🏁 {folder.name} 시나리오 종료 (저장 완료: {save_dir})")

if __name__ == "__main__":
    run_scenario_test("tests")