import pinecone
from pinecone import Pinecone
import time
import numpy as np

# Pinecone 設定
PINECONE_API_KEY = "pcsk_6k1GxA_GGMdrBj2MasWquFDMCnkbz6U4zxRSqpznZue37y2XgANWdBSsagY1zDxKUKdGyP"
pinecone_env = "us-east-1"
index_name = "test"

def get_all_vectors_from_pinecone():
    """
    從 Pinecone 向量資料庫中撷取所有資料
    使用多種查詢方法來獲取向量
    """
    try:
        # 初始化 Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # 連接到指定的 index
        index = pc.Index(index_name)
        
        # 獲取 index 統計資訊
        stats = index.describe_index_stats()
        print(f"Index 統計資訊: {stats}")
        
        total_vectors = stats.get('total_vector_count', 0)
        print(f"Index 中總共有 {total_vectors} 個向量")
        
        if total_vectors == 0:
            print("Index 中沒有向量")
            return []
        
        all_vectors = []
        
        # 方法1: 嘗試使用 list() 方法
        print("嘗試使用 list() 方法...")
        try:
            list_response = index.list(limit=1000)
            if 'vectors' in list_response and list_response['vectors']:
                vector_ids = [v['id'] for v in list_response['vectors']]
                print(f"通過 list() 方法找到 {len(vector_ids)} 個向量 ID")
                
                # 分批獲取向量資料
                batch_size = 100
                for i in range(0, len(vector_ids), batch_size):
                    batch_ids = vector_ids[i:i + batch_size]
                    fetch_response = index.fetch(ids=batch_ids)
                    
                    if 'vectors' in fetch_response:
                        for vector_id, vector_data in fetch_response['vectors'].items():
                            vector_info = {
                                'id': vector_id,
                                'values': vector_data.get('values', []),
                                'metadata': vector_data.get('metadata', {}),
                                'sparse_values': vector_data.get('sparse_values', {})
                            }
                            all_vectors.append(vector_info)
                    
                    print(f"已處理 {min(i + batch_size, len(vector_ids))}/{len(vector_ids)} 個向量")
                    time.sleep(0.1)
                
                return all_vectors
            else:
                print("list() 方法沒有返回向量")
        except Exception as e:
            print(f"list() 方法失敗: {e}")
        
        # 方法2: 使用多種查詢策略來發現向量
        print("使用多種查詢方法來發現向量...")
        
        discovered_vectors = {}
        
        # 策略1: 隨機查詢
        print("執行隨機查詢...")
        num_random_queries = 50  # 增加隨機查詢次數
        
        for i in range(num_random_queries):
            # 生成隨機查詢向量
            random_vector = np.random.random(1024).tolist()
            
            try:
                query_response = index.query(
                    vector=random_vector,
                    top_k=10,
                    include_metadata=True,
                    include_values=True
                )
                
                if 'matches' in query_response:
                    for match in query_response['matches']:
                        vector_id = match['id']
                        if vector_id not in discovered_vectors:
                            discovered_vectors[vector_id] = {
                                'id': vector_id,
                                'values': match.get('values', []),
                                'metadata': match.get('metadata', {}),
                                'score': match.get('score', 0.0)
                            }
                
                if i % 10 == 0:  # 每10次顯示一次進度
                    print(f"隨機查詢 {i+1}/{num_random_queries} 完成，已發現 {len(discovered_vectors)} 個獨特向量")
                time.sleep(0.05)
                
            except Exception as e:
                print(f"隨機查詢 {i+1} 失敗: {e}")
                continue
        
        # 策略2: 使用極端值向量查詢
        print("執行極端值向量查詢...")
        extreme_vectors = [
            [1.0] * 1024,  # 全部最大值
            [0.0] * 1024,  # 全部最小值
            [-1.0] * 1024,  # 全部負值
            [0.5] * 1024,  # 全部中間值
        ]
        
        for i, extreme_vector in enumerate(extreme_vectors):
            try:
                query_response = index.query(
                    vector=extreme_vector,
                    top_k=20,  # 增加 top_k
                    include_metadata=True,
                    include_values=True
                )
                
                if 'matches' in query_response:
                    for match in query_response['matches']:
                        vector_id = match['id']
                        if vector_id not in discovered_vectors:
                            discovered_vectors[vector_id] = {
                                'id': vector_id,
                                'values': match.get('values', []),
                                'metadata': match.get('metadata', {}),
                                'score': match.get('score', 0.0)
                            }
                
                print(f"極端值查詢 {i+1}/{len(extreme_vectors)} 完成，已發現 {len(discovered_vectors)} 個獨特向量")
                time.sleep(0.1)
                
            except Exception as e:
                print(f"極端值查詢 {i+1} 失敗: {e}")
                continue
        
        # 策略3: 使用已發現向量進行鄰近查詢
        if discovered_vectors:
            print("使用已發現向量進行鄰近查詢...")
            existing_vectors = list(discovered_vectors.values())
            
            for i, vector in enumerate(existing_vectors[:5]):  # 使用前5個向量
                if vector['values']:
                    try:
                        # 對向量加入小幅度擾動
                        perturbed_vector = [v + np.random.normal(0, 0.1) for v in vector['values']]
                        
                        query_response = index.query(
                            vector=perturbed_vector,
                            top_k=15,
                            include_metadata=True,
                            include_values=True
                        )
                        
                        if 'matches' in query_response:
                            for match in query_response['matches']:
                                vector_id = match['id']
                                if vector_id not in discovered_vectors:
                                    discovered_vectors[vector_id] = {
                                        'id': vector_id,
                                        'values': match.get('values', []),
                                        'metadata': match.get('metadata', {}),
                                        'score': match.get('score', 0.0)
                                    }
                        
                        print(f"鄰近查詢 {i+1}/5 完成，已發現 {len(discovered_vectors)} 個獨特向量")
                        time.sleep(0.1)
                        
                    except Exception as e:
                        print(f"鄰近查詢 {i+1} 失敗: {e}")
                        continue
        
        all_vectors = list(discovered_vectors.values())
        print(f"通過所有查詢方法總共發現 {len(all_vectors)} 個向量")
        
        # 如果還是沒找到所有向量，給出建議
        if len(all_vectors) < total_vectors:
            print(f"\n注意: 只找到 {len(all_vectors)}/{total_vectors} 個向量")
            print("這可能是因為:")
            print("1. 某些向量在向量空間中位置較為孤立")
            print("2. Pinecone 的 list() API 在某些情況下不穩定")
            print("3. 需要更多的查詢策略")
            print("\n建議:")
            print("1. 重新運行腳本，有時會找到更多向量")
            print("2. 如果需要完整資料，考慮重新上傳向量")
        
        return all_vectors
        
    except Exception as e:
        print(f"錯誤: {e}")
        return []

def save_vectors_to_file(vectors, filename="pinecone_vectors.txt"):
    """
    將向量資料儲存到檔案
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"總共找到 {len(vectors)} 個向量\n")
            f.write("=" * 80 + "\n\n")
            
            for i, vector in enumerate(vectors):
                f.write(f"=== 向量 {i+1} ===\n")
                f.write(f"ID: {vector['id']}\n")
                f.write(f"維度: {len(vector['values'])}\n")
                
                # 處理 metadata
                if vector['metadata']:
                    f.write(f"Metadata:\n")
                    for key, value in vector['metadata'].items():
                        # 限制長度避免檔案過大
                        if len(str(value)) > 200:
                            f.write(f"  {key}: {str(value)[:200]}...\n")
                        else:
                            f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"Metadata: 無\n")
                
                # 顯示向量值的統計資訊
                if vector['values']:
                    values = vector['values']
                    f.write(f"向量統計:\n")
                    f.write(f"  最小值: {min(values):.6f}\n")
                    f.write(f"  最大值: {max(values):.6f}\n")
                    f.write(f"  平均值: {sum(values)/len(values):.6f}\n")
                    f.write(f"  前10個值: {values[:10]}\n")
                
                if 'score' in vector:
                    f.write(f"相似度分數: {vector['score']:.6f}\n")
                
                f.write("-" * 50 + "\n\n")
        
        print(f"向量資料已儲存到 {filename}")
        
    except Exception as e:
        print(f"儲存檔案時發生錯誤: {e}")

def print_vector_summary(vectors):
    """
    印出向量資料摘要
    """
    if not vectors:
        print("沒有找到任何向量資料")
        return
    
    print(f"\n=== 向量資料摘要 ===")
    print(f"總向量數量: {len(vectors)}")
    
    if vectors:
        print(f"向量維度: {len(vectors[0]['values'])}")
        
        # 統計 metadata 的鍵
        all_metadata_keys = set()
        text_samples = []
        
        for vector in vectors:
            if vector['metadata']:
                all_metadata_keys.update(vector['metadata'].keys())
                # 收集一些文本樣本
                if 'text' in vector['metadata']:
                    text = vector['metadata']['text']
                    if len(text) > 50:
                        text_samples.append(text[:100] + "...")
                    else:
                        text_samples.append(text)
        
        print(f"Metadata 欄位: {list(all_metadata_keys)}")
        
        if text_samples:
            print(f"\n前3個文本樣本:")
            for i, text in enumerate(text_samples[:3]):
                print(f"  {i+1}. {text}")
    
    # 顯示前幾個向量的詳細資訊
    print(f"\n=== 前3個向量詳細資訊 ===")
    for i, vector in enumerate(vectors[:3]):
        print(f"\n向量 {i+1}:")
        print(f"  ID: {vector['id']}")
        
        if vector['metadata']:
            print(f"  Metadata 鍵: {list(vector['metadata'].keys())}")
            if 'source_file' in vector['metadata']:
                print(f"  來源檔案: {vector['metadata']['source_file']}")
            if 'text' in vector['metadata']:
                text = vector['metadata']['text']
                if len(text) > 100:
                    print(f"  文本: {text[:100]}...")
                else:
                    print(f"  文本: {text}")
        
        if vector['values']:
            print(f"  向量值 (前5個): {vector['values'][:5]}")
        
        if 'score' in vector:
            print(f"  相似度分數: {vector['score']:.6f}")

if __name__ == "__main__":
    print("開始從 Pinecone 撷取向量資料...")
    
    # 獲取所有向量
    all_vectors = get_all_vectors_from_pinecone()
    
    if all_vectors:
        # 印出摘要
        print_vector_summary(all_vectors)
        
        # 儲存到檔案
        save_vectors_to_file(all_vectors)
        
        print(f"\n成功撷取 {len(all_vectors)} 個向量！")
    else:
        print("沒有撷取到任何向量資料")
        print("這可能是因為:")
        print("1. Index 中的向量無法通過 list() 方法訪問")
        print("2. 向量沒有正確的 metadata")
        print("3. API 權限問題")