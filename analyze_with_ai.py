import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import json
from datetime import datetime
import time
from tqdm import tqdm
import re

# 載入環境變數
load_dotenv()

# 初始化 OpenAI 客戶端
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("請在 .env 檔案中設定 OPENAI_API_KEY")

def create_analysis_directory():
    """創建分析結果目錄"""
    results_dir = "analysis_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir

def clean_text(text):
    """清理文本，移除特殊字符並處理空值"""
    if pd.isna(text) or text == "":
        return None
    # 移除特殊字符但保留中文、英文、日文等文字
    text = re.sub(r'[^\w\s\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', ' ', str(text))
    # 移除多餘空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else None

def get_chatgpt_sentiment(text, model="gpt-3.5-turbo", max_retries=3):
    """使用 ChatGPT 分析評論情感，包含重試機制"""
    if not text:
        return {
            'sentiment': 'unknown',
            'confidence': 0,
            'reason': 'Empty review text'
        }

    prompt = f"""Analyze the sentiment of the following review and respond with a JSON object containing:
1. sentiment: either "positive", "negative", or "neutral"
2. confidence: a number between 0 and 1
3. reason: a brief explanation

Review text:
{text}

Respond ONLY with the JSON object, no other text or formatting. Example response format:
{{"sentiment": "positive", "confidence": 0.9, "reason": "The review expresses satisfaction"}}"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis expert. Always respond with a valid JSON object containing sentiment, confidence, and reason fields. Do not include any other text or formatting."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={ "type": "json_object" }
            )
            
            # 解析 JSON 回應
            result = json.loads(response.choices[0].message.content)
            
            # 驗證結果格式
            required_fields = ['sentiment', 'confidence', 'reason']
            if not all(field in result for field in required_fields):
                raise ValueError("Missing required fields in response")
            
            # 驗證情感值
            if result['sentiment'].lower() not in ['positive', 'negative', 'neutral']:
                raise ValueError("Invalid sentiment value")
            
            # 驗證信心值
            confidence = float(result['confidence'])
            if not 0 <= confidence <= 1:
                raise ValueError("Invalid confidence value")
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON 解析錯誤 (嘗試 {attempt + 1}/{max_retries}): {str(e)}")
            print(f"原始回應: {response.choices[0].message.content}")
            if attempt == max_retries - 1:
                return {
                    'sentiment': 'unknown',
                    'confidence': 0,
                    'reason': f'JSON parsing error: {str(e)}'
                }
            time.sleep(2 ** attempt)  # 指數退避
            
        except Exception as e:
            print(f"API 呼叫錯誤 (嘗試 {attempt + 1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                return {
                    'sentiment': 'unknown',
                    'confidence': 0,
                    'reason': f'API error: {str(e)}'
                }
            time.sleep(2 ** attempt)  # 指數退避

def resolve_conflict(analyses):
    """解決不同模型分析結果的衝突"""
    # 計算每個情感類別的總信心度
    sentiment_scores = {
        'positive': 0,
        'negative': 0,
        'neutral': 0
    }
    
    for analysis in analyses:
        if analysis and 'sentiment' in analysis and 'confidence' in analysis:
            sentiment = analysis['sentiment'].lower()
            confidence = float(analysis['confidence'])
            sentiment_scores[sentiment] += confidence
    
    # 找出最高分的情感類別
    max_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])
    
    # 如果最高分的情感類別分數明顯高於其他類別（差距大於0.3），則採用該結果
    other_scores = [score for sentiment, score in sentiment_scores.items() if sentiment != max_sentiment[0]]
    if not other_scores or (max_sentiment[1] - max(other_scores)) > 0.3:
        return {
            'sentiment': max_sentiment[0],
            'confidence': max_sentiment[1] / len(analyses),
            'resolution_method': 'confidence_threshold',
            'all_analyses': analyses
        }
    
    # 如果分數接近，則採用最保守的判斷（neutral > positive > negative）
    if sentiment_scores['neutral'] > 0:
        return {
            'sentiment': 'neutral',
            'confidence': sentiment_scores['neutral'] / len(analyses),
            'resolution_method': 'conservative',
            'all_analyses': analyses
        }
    elif sentiment_scores['positive'] > 0:
        return {
            'sentiment': 'positive',
            'confidence': sentiment_scores['positive'] / len(analyses),
            'resolution_method': 'conservative',
            'all_analyses': analyses
        }
    else:
        return {
            'sentiment': 'negative',
            'confidence': sentiment_scores['negative'] / len(analyses),
            'resolution_method': 'conservative',
            'all_analyses': analyses
        }

def analyze_reviews_with_ai(input_file, output_file=None):
    """使用 AI 模型分析評論"""
    try:
        # 讀取 CSV 檔案
        df = pd.read_csv(input_file)
        
        # 驗證必要的欄位
        required_columns = ['review_text']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV 檔案缺少必要的欄位: {', '.join(missing_columns)}")
        
        # 清理文本
        df['review_text'] = df['review_text'].apply(clean_text)
        
        # 準備結果列表
        results = []
        
        # 使用模型分析每則評論
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="分析評論"):
            review_text = row['review_text']
            
            # 使用模型分析
            analysis = get_chatgpt_sentiment(review_text)
            
            # 如果分析失敗，使用預設值
            if not analysis or analysis['sentiment'] == 'unknown':
                final_analysis = {
                    'sentiment': 'unknown',
                    'confidence': 0,
                    'resolution_method': 'error',
                    'all_analyses': []
                }
            else:
                final_analysis = {
                    'sentiment': analysis['sentiment'],
                    'confidence': analysis['confidence'],
                    'resolution_method': 'direct',
                    'all_analyses': [analysis]
                }
            
            # 將原始資料和 AI 分析結果合併
            result = row.to_dict()
            result.update({
                'ai_sentiment': final_analysis['sentiment'],
                'ai_confidence': final_analysis['confidence'],
                'resolution_method': final_analysis['resolution_method'],
                'all_analyses': json.dumps(final_analysis['all_analyses'], ensure_ascii=False)
            })
            
            results.append(result)
            
            # 避免 API 限制
            time.sleep(1)
        
        # 轉換為 DataFrame
        result_df = pd.DataFrame(results)
        
        # 創建分析結果目錄
        results_dir = create_analysis_directory()
        
        # 生成時間戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 創建結果子目錄
        result_subdir = os.path.join(results_dir, f"analysis_{timestamp}")
        os.makedirs(result_subdir)
        
        # 儲存結果
        if output_file is None:
            output_file = os.path.join(result_subdir, "analysis_results.csv")
        
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        # 創建摘要文件
        summary = {
            "timestamp": timestamp,
            "total_reviews": len(result_df),
            "successful_analyses": len(result_df[result_df['ai_sentiment'] != 'unknown']),
            "failed_analyses": len(result_df[result_df['ai_sentiment'] == 'unknown']),
            "sentiment_distribution": result_df['ai_sentiment'].value_counts().to_dict(),
            "resolution_methods": result_df['resolution_method'].value_counts().to_dict(),
            "average_confidence": result_df['ai_confidence'].mean(),
            "place_statistics": {}
        }
        
        # 按地點統計
        if 'place_name' in result_df.columns:
            for place in result_df['place_name'].unique():
                place_df = result_df[result_df['place_name'] == place]
                summary["place_statistics"][place] = {
                    "total_reviews": len(place_df),
                    "successful_analyses": len(place_df[place_df['ai_sentiment'] != 'unknown']),
                    "failed_analyses": len(place_df[place_df['ai_sentiment'] == 'unknown']),
                    "sentiment_distribution": place_df['ai_sentiment'].value_counts().to_dict(),
                    "average_confidence": place_df['ai_confidence'].mean()
                }
        
        # 保存摘要
        with open(os.path.join(result_subdir, "summary.json"), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # 輸出分析摘要
        print("\n分析摘要:")
        print(f"總評論數: {len(result_df)}")
        print(f"成功分析: {summary['successful_analyses']}")
        print(f"分析失敗: {summary['failed_analyses']}")
        print("\n情感分布:")
        sentiment_counts = result_df['ai_sentiment'].value_counts()
        for sentiment, count in sentiment_counts.items():
            print(f"{sentiment}: {count} 則 ({count/len(result_df)*100:.1f}%)")
        
        print("\n解決方法分布:")
        resolution_counts = result_df['resolution_method'].value_counts()
        for method, count in resolution_counts.items():
            print(f"{method}: {count} 則 ({count/len(result_df)*100:.1f}%)")
        
        print(f"\n結果已儲存至: {result_subdir}")
        print("\n結果包含:")
        print("1. analysis_results.csv - 原始分析數據")
        print("2. summary.json - 分析摘要")
        
    except Exception as e:
        print(f"分析過程中發生錯誤: {str(e)}")
        raise

def main():
    # 設定目錄路徑
    input_dir = "data_store/Multi_places"
    output_dir = "data_store/ai_analysis"
    
    # 確保輸出目錄存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 取得所有 CSV 檔案
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("找不到可分析的 CSV 檔案")
        return
    
    # 找出最新的檔案（根據檔名中的時間戳）
    latest_file = max(csv_files, key=lambda x: x.split('_')[-1].split('.')[0])
    input_file = os.path.join(input_dir, latest_file)
    
    # 生成輸出檔案名稱（使用相同的時間戳）
    timestamp = latest_file.split('_')[-1].split('.')[0]
    output_file = os.path.join(output_dir, f"ai_analysis_{timestamp}.csv")
    
    print(f"分析檔案: {input_file}")
    print(f"結果將儲存至: {output_file}")
    
    analyze_reviews_with_ai(input_file, output_file)

if __name__ == "__main__":
    main()