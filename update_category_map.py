import json
import os
import glob

# =========================================================
# [설정] 파일 경로 및 폴더 이름
# =========================================================
METADATA_FILE = 'wine_metadata.jsonl'       # 메타데이터 파일
OUTPUT_FILE = 'winery_category_map.json'    # 결과 저장 파일 (덮어쓰기)
REVIEW_DIR = 'cleaned'                      # 리뷰 데이터가 있는 폴더명 (cleaned 또는 review)

def get_review_count(wine_id):
    """
    해당 와인 ID의 리뷰 파일이 있으면 라인 수(리뷰 수)를 세어 반환합니다.
    없으면 0을 반환합니다.
    """
    # 1. cleaned 폴더의 jsonl 파일 확인 (한 줄 = 리뷰 하나)
    file_path = os.path.join(REVIEW_DIR, f"wine_{wine_id}_clean.jsonl")
    
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 라인 수가 곧 리뷰 개수
                return sum(1 for _ in f)
        except:
            return 0
            
    # (선택사항) 만약 cleaned가 없고 review 폴더를 확인해야 한다면 로직 추가 가능
    return 0

def main():
    # 1. 메타데이터 로딩 및 와이너리별 그룹화
    print(f"📖 {METADATA_FILE} 읽는 중...")
    winery_groups = {}
    
    if not os.path.exists(METADATA_FILE):
        print(f"❌ 오류: {METADATA_FILE} 파일이 없습니다.")
        return

    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                wine = json.loads(line)
                # 와이너리 이름 정규화 (소문자, 앞뒤 공백 제거)
                raw_winery = wine.get('winery')
                if not raw_winery: continue
                
                winery_key = raw_winery.strip().lower()
                
                if winery_key not in winery_groups:
                    winery_groups[winery_key] = []
                winery_groups[winery_key].append(wine)
            except json.JSONDecodeError:
                continue

    print(f"✅ 총 {len(winery_groups)}개의 와이너리 발견.")

    # 2. 와이너리별 대표 와인 선정 (리뷰 수 기준)
    category_map = {}
    print(f"🔍 와이너리별 대표 와인 스캔 중 (폴더: {REVIEW_DIR})...")

    for i, (winery, wines) in enumerate(winery_groups.items()):
        best_wine = None
        max_reviews = -1
        
        # 해당 와이너리의 모든 와인을 순회하며 리뷰 수 체크
        for wine in wines:
            w_id = wine.get('id')
            count = get_review_count(w_id)
            
            # 리뷰가 더 많거나, 리뷰 수는 같아도 아직 선택된 와인이 없으면 갱신
            if count > max_reviews:
                max_reviews = count
                best_wine = wine
            elif max_reviews == -1 and best_wine is None:
                # 리뷰 파일이 아예 없는 경우라도 일단 첫 번째 와인을 선택
                best_wine = wine

        # 3. 카테고리 정보 생성: [Country, Region1, Region2, ...]
        if best_wine:
            country = best_wine.get('country', 'Unknown')
            regions = best_wine.get('region', [])
            
            # 리스트 합치기
            category_info = [country] + regions
            category_map[winery] = category_info

        # 진행 상황 표시 (100개마다)
        if (i + 1) % 100 == 0:
            print(f"   ...{i + 1}개 와이너리 처리 완료")

    # 4. 결과 저장
    print(f"💾 {OUTPUT_FILE} 에 저장 중...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(category_map, f, indent=4, ensure_ascii=False)
    
    print("✨ 완료되었습니다!")

if __name__ == "__main__":
    main()