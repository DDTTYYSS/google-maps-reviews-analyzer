import os
import csv
from dotenv import load_dotenv
import requests
from datetime import datetime
import argparse

# 1. 載入 .env
load_dotenv()

# 2. 從環境變數讀取金鑰
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
if not API_KEY:
    raise RuntimeError("請先在 .env 設定 GOOGLE_MAPS_API_KEY")

# 設定命令列參數
parser = argparse.ArgumentParser(description='獲取 Google 評論')
parser.add_argument('--query', type=str, help='若使用關鍵字搜尋，輸入地點名稱')
parser.add_argument('--place_id', type=str, default="ChIJty1ap7erQjQRWmO-mTv1zkQ", 
                    help='直接指定 Place ID (預設：台北君悅酒店)')
parser.add_argument('--related', action='store_true', help='同時爬取相關地點')
parser.add_argument('--max_related', type=int, default=3, help='相關地點最大數量 (預設3)')
parser.add_argument('--output_dir', type=str, default="data_store", help='CSV 檔案輸出目錄路徑 (預設: data_store)')
args = parser.parse_args()

# 確保輸出目錄存在
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
    print(f"已建立輸出目錄: {args.output_dir}")

# 更多語言選項
languages = [
    "zh-TW", "en"
]

def search_place(query):
    """使用關鍵字搜尋地點"""
    search_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": query,
        "key": API_KEY
    }
    
    resp = requests.get(search_url, params=params)
    data = resp.json()
    
    if data.get("status") == "OK" and data.get("results"):
        print(f"搜尋 '{query}' 的結果:")
        for i, place in enumerate(data["results"][:5], 1):
            print(f"{i}. {place['name']} (PlaceID: {place['place_id']})")
        
        return data["results"][0]["place_id"]  # 返回第一個結果的 Place ID
    
    print(f"找不到與 '{query}' 相關的地點")
    return None

def find_related_places(place_id):
    """尋找相關地點"""
    # 先獲取地點詳情以取得地理位置
    details_url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "fields": "name,geometry",
        "key": API_KEY
    }
    
    resp = requests.get(details_url, params=params)
    data = resp.json()
    
    if data.get("status") != "OK":
        print("無法獲取地點詳情，無法找到相關地點")
        return []
    
    # 獲取地理位置
    location = data["result"]["geometry"]["location"]
    
    # 使用 nearby search 尋找附近相似地點
    nearby_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{location['lat']},{location['lng']}",
        "type": "lodging",  # 假設是住宿類型
        "radius": 1000,     # 1公里範圍
        "key": API_KEY
    }
    
    resp = requests.get(nearby_url, params=params)
    data = resp.json()
    
    if data.get("status") == "OK" and data.get("results"):
        related_places = []
        
        for place in data["results"][:args.max_related + 1]:  # 多獲取一個，因為可能包含原始地點
            # 排除原始地點
            if place["place_id"] != place_id:
                related_places.append({
                    "name": place["name"],
                    "place_id": place["place_id"]
                })
                
                if len(related_places) >= args.max_related:
                    break
                    
        return related_places
    
    return []

def get_place_reviews(place_id, place_name=""):
    """獲取特定地點的評論"""
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    all_reviews = []
    rating = 0
    
    # 嘗試不同設定以獲取更多評論
    review_configs = [
        {"sort": None},                      # 預設排序
        {"sort": "newest"},                 # 最新評論
        {"sort": "most_relevant"},          # 最相關評論
        {"sort": "score"},                  # 按分數排序
        {"no_translations": False}            # 不帶翻譯 (可能獲取不同評論)
    ]
    
    for lang in languages:
        reviews_count = 0  # 追蹤此語言的評論數
        unique_reviews_before = len(all_reviews)
        
        for config in review_configs:
            params = {
                "place_id": place_id,
                "fields": "name,rating,reviews",
                "language": lang,
                "key": API_KEY
            }
            
            # 添加額外參數
            if config.get("sort"):
                params["reviews_sort"] = config["sort"]
            if config.get("no_translations"):
                params["reviews_no_translations"] = "true"
            
            resp = requests.get(url, params=params)
            data = resp.json()
            
            if data.get("status") == "OK":
                info = data["result"]
                if not place_name:  # 只在第一次獲取名稱和評分
                    place_name = info["name"]
                    rating = info.get("rating", 0)
                
                # 將該語言的評論加入全部評論
                reviews = info.get("reviews", [])
                
                # 如果沒有評論，嘗試下一個設定
                if len(reviews) == 0:
                    continue
                    
                reviews_count += len(reviews)
                
                for review in reviews:
                    # 用評論ID (或作者+時間) 來避免重複評論
                    review_id = f"{review['author_name']}_{review['time']}"
                    # 檢查這個評論是否已經存在
                    if not any(r.get('_id') == review_id for r in all_reviews):
                        review['_id'] = review_id  # 為了後續去重處理
                        review['_language'] = lang  # 標記該評論來自哪種語言請求
                        review['_place_name'] = place_name  # 添加地點名稱
                        review['_place_id'] = place_id  # 添加地點ID
                        all_reviews.append(review)
            else:
                config_desc = f"sort={config.get('sort')}" if config.get('sort') else "無排序"
                if config.get('no_translations'):
                    config_desc += ", 無翻譯"
                
                if "error_message" in data:
                    print(f"  API 呼叫失敗 ({lang}, {config_desc})，狀態：{data.get('status')}, 錯誤：{data['error_message'][:50]}...")
                else:
                    print(f"  API 呼叫失敗 ({lang}, {config_desc})，狀態：{data.get('status')}")
        
        unique_reviews_after = len(all_reviews)
        unique_reviews_added = unique_reviews_after - unique_reviews_before
        print(f"  語言 {lang}: 找到 {reviews_count} 則評論, 新增 {unique_reviews_added} 則不重複評論")
    
    return all_reviews, place_name, rating

# 主程序
all_place_reviews = []
place_infos = []

# 處理用戶輸入
if args.query:
    print(f"搜尋地點: {args.query}")
    found_place_id = search_place(args.query)
    if found_place_id:
        target_place_id = found_place_id
    else:
        print("無法找到地點，使用預設 Place ID")
        target_place_id = args.place_id
else:
    target_place_id = args.place_id

# 首先獲取主要地點的評論
print(f"\n獲取主要地點 (ID: {target_place_id}) 的評論:")
reviews, place_name, rating = get_place_reviews(target_place_id)
all_place_reviews.extend(reviews)
place_infos.append({"place_id": target_place_id, "name": place_name, "rating": rating})

# 如果要獲取相關地點的評論
if args.related:
    print(f"\n尋找相關地點...")
    related_places = find_related_places(target_place_id)
    
    if related_places:
        print(f"找到 {len(related_places)} 個相關地點:")
        for i, place in enumerate(related_places, 1):
            print(f"{i}. {place['name']} (ID: {place['place_id']})")
            
            print(f"\n獲取相關地點 {place['name']} 的評論:")
            rel_reviews, rel_name, rel_rating = get_place_reviews(place['place_id'], place['name'])
            all_place_reviews.extend(rel_reviews)
            place_infos.append({"place_id": place['place_id'], "name": rel_name, "rating": rel_rating})
    else:
        print("找不到相關地點")

# 顯示結果
if all_place_reviews:
    print("\n評論統計:")
    for place in place_infos:
        place_reviews = [r for r in all_place_reviews if r.get('_place_id') == place['place_id']]
        print(f"- {place['name']}（評分：{place['rating']}）: {len(place_reviews)} 則評論")
    
    print(f"\n總共收集到 {len(all_place_reviews)} 則不重複評論\n")
    
    # 輸出部分評論範例
    sample_size = min(5, len(all_place_reviews))
    print(f"評論範例 ({sample_size} 則):")
    for r in all_place_reviews[:sample_size]:
        lang_marker = f"[{r['_language']}]" if r['_language'] != languages[0] else ""
        place_marker = f" - {r['_place_name']}" if len(place_infos) > 1 else ""
        print(f"- {r['author_name']}{lang_marker}{place_marker}：{r['rating']}★（{r['relative_time_description']}）")
        # 只顯示評論的前100個字符
        review_text = r['text']
        if len(review_text) > 100:
            review_text = review_text[:100] + "..."
        print(f"  「{review_text}」\n")
    
    # 儲存到 CSV 檔案
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if len(place_infos) == 1:
        filename = f"{place_infos[0]['name'].replace(' ', '_')}_{timestamp}.csv"
    else:
        filename = f"Multiple_Places_{timestamp}.csv"
    
    # 將檔案存在指定目錄下
    filepath = os.path.join(args.output_dir, filename)
    
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['place_name', 'place_id', 'place_rating', 'author_name', 'rating', 'language', 
                      'review_text', 'time', 'relative_time', 'author_url', 'profile_photo_url']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for review in all_place_reviews:
            writer.writerow({
                'place_name': review['_place_name'],
                'place_id': review['_place_id'],
                'place_rating': next((p['rating'] for p in place_infos if p['place_id'] == review['_place_id']), 0),
                'author_name': review['author_name'],
                'rating': review['rating'],
                'language': review['_language'],
                'review_text': review['text'],
                'time': review['time'],
                'relative_time': review['relative_time_description'],
                'author_url': review.get('author_url', ''),
                'profile_photo_url': review.get('profile_photo_url', '')
            })
    
    print(f"已將評論儲存至: {filepath}")
else:
    print("沒有找到任何評論")

print("\n使用範例:")
print("1. 基本用法 (使用預設的 Place ID): python more_reviews.py")
print("2. 指定 Place ID: python more_reviews.py --place_id ChIJty1ap7erQjQRWmO-mTv1zkQ")
print("3. 關鍵字搜尋地點: python more_reviews.py --query \"台北君悅酒店\"")
print("4. 同時爬取相關地點評論: python more_reviews.py --related")
print("5. 指定相關地點最大數量: python more_reviews.py --related --max_related 5")
print("6. 指定輸出目錄: python more_reviews.py --output_dir \"my_data_folder\"") 


## 去爬資料設定不同的起始點
## 去串ChatGPT API(將我的結論去丟給AI 去做分析，判斷正向負向，同一則評論可以問不同語言模型，如果有conflict就再看看要怎麼辦，研究一下conflict resolution)
## 想要得知關於重大事件（要去找那個重大事件之前的時間點，疫情前半年之類的或是疫情高峰，可以去做分析，另一個維度按照時間來區分的分析）
## 可以按季節（春夏秋冬）（颱風季），簡單來說分類每個時間來講各自趨勢或者是說會有什麼東西
## 地震、海嘯、颱風
## 今年可能像是關稅、民族意識看有沒有跟旅遊區是有什麼關聯性（但主要是用來分析那個時間區段有沒有差別）
## 可以根據不同的時段來去爬文，進行分析，看是正向負向
## 下一步就是去讓這個服務上雲端，自動更新以及提供分析報告，像是一個時間段


## 需要先把爬蟲根據時間地點季節爬下來分類，然後餵給語言模型判斷，有第三個模組分析結果，然後給出相關的預測 