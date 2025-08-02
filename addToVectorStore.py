import pinecone
from pinecone import Pinecone
import numpy as np
import time
import uuid
import json
import csv
import pandas as pd
import os
import re
from sentence_transformers import SentenceTransformer
from pathlib import Path
import PyPDF2
import fitz  # PyMuPDF
import hashlib

# Pinecone 設定
PINECONE_API_KEY = "pcsk_6k1GxA_GGMdrBj2MasWquFDMCnkbz6U4zxRSqpznZue37y2XgANWdBSsagY1zDxKUKdGyP"
pinecone_env = "us-east-1"
index_name = "test"

def initialize_pinecone():
    """
    初始化 Pinecone 連接
    """
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(index_name)
        print(f"成功連接到 Pinecone index: {index_name}")
        return index
    except Exception as e:
        print(f"初始化 Pinecone 時發生錯誤: {e}")
        return None

def get_recommended_chunk_size(content, split_by='char'):
    """
    根據內容類型推薦 chunk 大小
    """
    content_length = len(content)
    
    if split_by == 'char':
        if content_length < 5000:
            return 500, 50  # chunk_size, overlap_size
        elif content_length < 50000:
            return 1000, 100
        else:
            return 1500, 150
    
    elif split_by == 'line':
        lines = content.split('\n')
        total_lines = len(lines)
        
        if total_lines < 100:
            return 10, 2
        elif total_lines < 1000:
            return 20, 3
        else:
            return 30, 5
    
    elif split_by == 'sentence':
        sentences = re.split(r'[.!?。！？]+', content)
        total_sentences = len(sentences)
        
        if total_sentences < 50:
            return 5, 1
        elif total_sentences < 200:
            return 10, 2
        else:
            return 15, 3
    
    return 1000, 100  # 預設值

def calculate_optimal_overlap(chunk_size, content_type='general'):
    """
    計算最佳重疊大小
    """
    overlap_ratios = {
        'technical': 0.15,    # 技術文件需要更多重疊
        'narrative': 0.10,    # 故事性文本
        'general': 0.12,      # 一般文本
        'legal': 0.20,        # 法律文件需要更多上下文
        'academic': 0.15      # 學術文章
    }
    
    ratio = overlap_ratios.get(content_type, 0.12)
    optimal_overlap = int(chunk_size * ratio)
    
    # 設定最小和最大值
    min_overlap = max(50, chunk_size // 20)
    max_overlap = chunk_size // 3
    
    return max(min_overlap, min(optimal_overlap, max_overlap))

def smart_chunk_text(content, max_chunk_size=1000, overlap_size=100, split_by='char'):
    """
    智能文本分割，保持語義完整性
    """
    chunks = []
    
    if split_by == 'char':
        # 優先在句號處分割
        sentences = re.split(r'([.!?。！？]+)', content)
        current_chunk = ""
        
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                sentence = sentences[i] + sentences[i + 1]
            else:
                sentence = sentences[i]
            
            if len(current_chunk + sentence) <= max_chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # 保留重疊部分
                    if overlap_size > 0:
                        current_chunk = current_chunk[-overlap_size:] + sentence
                    else:
                        current_chunk = sentence
                else:
                    # 句子太長，強制分割
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
    
    elif split_by == 'paragraph':
        # 按段落分割
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk + para) <= max_chunk_size:
                current_chunk += para + '\n\n'
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    if overlap_size > 0:
                        # 保留上一個段落的一部分
                        last_para = current_chunk.split('\n\n')[-2] if '\n\n' in current_chunk else ""
                        current_chunk = last_para[-overlap_size:] + para
                    else:
                        current_chunk = para
                else:
                    current_chunk = para
        
        if current_chunk:
            chunks.append(current_chunk.strip())
    
    return chunks

def read_txt_file(file_path, chunk_size=None, overlap_size=None, split_by='char', content_type='general'):
    """
    讀取 TXT 檔案並智能分割成 chunks
    
    Args:
        file_path: 檔案路徑
        chunk_size: chunk 大小（None 時自動計算）
        overlap_size: 重疊大小（None 時自動計算）
        split_by: 分割方式 ('char', 'line', 'sentence', 'paragraph')
        content_type: 內容類型 ('general', 'technical', 'narrative', 'legal', 'academic')
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 自動計算參數
        if chunk_size is None or overlap_size is None:
            recommended_chunk, recommended_overlap = get_recommended_chunk_size(content, split_by)
            chunk_size = chunk_size or recommended_chunk
            overlap_size = overlap_size or calculate_optimal_overlap(chunk_size, content_type)
        
        print(f"使用 chunk_size: {chunk_size}, overlap_size: {overlap_size}")
        
        chunks = []
        
        if split_by == 'char':
            content = content.replace('\n', ' ').strip()
            for i in range(0, len(content), chunk_size - overlap_size):
                chunk = content[i:i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk.strip())
                    
        elif split_by == 'line':
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            for i in range(0, len(lines), chunk_size - overlap_size):
                chunk_lines = lines[i:i + chunk_size]
                if chunk_lines:
                    chunk = '\n'.join(chunk_lines)
                    chunks.append(chunk)
                    
        elif split_by == 'sentence':
            sentences = re.split(r'[.!?。！？]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            for i in range(0, len(sentences), chunk_size - overlap_size):
                chunk_sentences = sentences[i:i + chunk_size]
                if chunk_sentences:
                    chunk = '。'.join(chunk_sentences) + '。'
                    chunks.append(chunk)
        
        elif split_by == 'paragraph':
            chunks = smart_chunk_text(content, chunk_size, overlap_size, 'paragraph')
        
        # 過濾太短的 chunks
        min_chunk_length = max(50, chunk_size // 10)
        chunks = [chunk for chunk in chunks if len(chunk) >= min_chunk_length]
        
        print(f"生成了 {len(chunks)} 個 chunks")
        return chunks
        
    except Exception as e:
        print(f"讀取 TXT 檔案時發生錯誤: {e}")
        return []

def read_pdf_file(file_path, chunk_size=None, overlap_size=None, split_by='char', content_type='general', method='pymupdf'):
    """
    讀取 PDF 檔案並分割成 chunks
    改進版本：增強中文支持和錯誤處理
    """
    try:
        content = ""
        
        if method == 'pypdf2':
            # 使用 PyPDF2 讀取，增強編碼處理
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    
                    # 嘗試修復編碼問題
                    if page_text:
                        try:
                            # 嘗試重新編碼
                            page_text = page_text.encode('latin1').decode('utf-8', errors='ignore')
                        except:
                            # 如果失敗，保持原文本但清理特殊字符
                            page_text = ''.join(char for char in page_text if ord(char) < 65536)
                    
                    content += page_text + "\n"
        
        elif method == 'pymupdf':
            # 使用 PyMuPDF 讀取，這通常對中文支持更好
            doc = fitz.open(file_path)
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()
                
                # 清理文本
                if page_text:
                    # 移除過多的空白字符
                    page_text = ' '.join(page_text.split())
                    content += page_text + "\n"
            doc.close()
        
        # 加入調試信息
        print(f"原始提取的內容長度: {len(content)}")
        print(f"原始內容預覽: {content[:200]}...")
        
        # 額外的文本清理
        content = clean_extracted_text(content)
        
        print(f"清理後的內容長度: {len(content)}")
        print(f"清理後內容預覽: {content[:200]}...")
        
        if not content.strip():
            print("PDF 檔案中沒有可提取的文本，可能是掃描版PDF")
            print("建議：")
            print("1. 使用OCR工具（如 pytesseract）處理掃描版PDF")
            print("2. 檢查PDF是否包含可選擇的文本")
            return []
        
        # 檢查是否仍有大量亂碼
        if is_mostly_garbled(content):
            print("檢測到可能的亂碼，嘗試其他方法...")
            return try_alternative_pdf_methods(file_path, chunk_size, overlap_size, split_by, content_type)
        
        # 自動計算參數
        if chunk_size is None or overlap_size is None:
            recommended_chunk, recommended_overlap = get_recommended_chunk_size(content, split_by)
            chunk_size = chunk_size or recommended_chunk
            overlap_size = overlap_size or calculate_optimal_overlap(chunk_size, content_type)
        
        print(f"使用 chunk_size: {chunk_size}, overlap_size: {overlap_size}")
        
        chunks = []
        
        if split_by == 'char':
            content = content.replace('\n', ' ').strip()
            for i in range(0, len(content), chunk_size - overlap_size):
                chunk = content[i:i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk.strip())
                    
        elif split_by == 'line':
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            for i in range(0, len(lines), chunk_size - overlap_size):
                chunk_lines = lines[i:i + chunk_size]
                if chunk_lines:
                    chunk = '\n'.join(chunk_lines)
                    chunks.append(chunk)
                    
        elif split_by == 'sentence':
            sentences = re.split(r'[.!?。！？]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            for i in range(0, len(sentences), chunk_size - overlap_size):
                chunk_sentences = sentences[i:i + chunk_size]
                if chunk_sentences:
                    chunk = '。'.join(chunk_sentences) + '。'
                    chunks.append(chunk)
        
        elif split_by == 'paragraph':
            chunks = smart_chunk_text(content, chunk_size, overlap_size, 'paragraph')
        
        # 調整過濾條件，避免過於嚴格
        min_chunk_length = max(20, chunk_size // 20)  # 降低最小長度要求
        chunks = [chunk for chunk in chunks if len(chunk) >= min_chunk_length]
        
        print(f"過濾前 chunks 數量: {len(chunks) if chunks else 0}")
        print(f"生成了 {len(chunks)} 個 chunks")
        
        # 如果沒有 chunks，嘗試簡單分割
        if not chunks and content.strip():
            print("嘗試簡單分割...")
            simple_chunks = [content[i:i+1000] for i in range(0, len(content), 800)]
            chunks = [chunk.strip() for chunk in simple_chunks if chunk.strip()]
            print(f"簡單分割生成了 {len(chunks)} 個 chunks")
        
        return chunks
        
    except Exception as e:
        print(f"讀取 PDF 檔案時發生錯誤: {e}")
        return []
    
def clean_extracted_text(text):
    """
    清理提取的文本
    """
    if not text:
        return ""
    
    # 移除控制字符
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    
    # 替換常見的亂碼模式
    replacements = {
        'ï¿½': '',  # 常見的UTF-8解碼錯誤標記
        '�': '',     # Unicode替換字符
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # 清理多餘的空白
    text = '\n'.join(line.strip() for line in text.split('\n'))
    text = '\n'.join(line for line in text.split('\n') if line)
    
    return text

def is_mostly_garbled(text, threshold=0.3):
    """
    檢查文本是否主要由亂碼組成
    """
    if not text:
        return True
    
    # 計算非ASCII字符的比例
    non_ascii_count = sum(1 for char in text if ord(char) > 127)
    total_chars = len(text)
    
    if total_chars == 0:
        return True
    
    # 如果非ASCII字符比例過高且不是常見的中文字符
    non_ascii_ratio = non_ascii_count / total_chars
    
    # 檢查是否包含常見的中文字符範圍
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0
    
    # 如果中文字符比例很低但非ASCII字符很多，可能是亂碼
    if non_ascii_ratio > threshold and chinese_ratio < 0.1:
        return True
    
    return False

def try_alternative_pdf_methods(file_path, chunk_size, overlap_size, split_by, content_type):
    """
    嘗試其他PDF處理方法
    """
    print("嘗試其他PDF處理方法...")
    
    # 方法1: 嘗試不同的PyMuPDF選項
    try:
        import fitz
        doc = fitz.open(file_path)
        content = ""
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            # 嘗試不同的文本提取選項
            text_dict = page.get_text("dict")
            page_text = ""
            
            for block in text_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if "text" in span:
                                page_text += span["text"] + " "
            
            content += page_text + "\n"
        
        doc.close()
        
        content = clean_extracted_text(content)
        if content.strip() and not is_mostly_garbled(content):
            print("使用替代方法成功提取文本")
            
            # 自動計算參數
            if chunk_size is None or overlap_size is None:
                recommended_chunk, recommended_overlap = get_recommended_chunk_size(content, split_by)
                chunk_size = chunk_size or recommended_chunk
                overlap_size = overlap_size or calculate_optimal_overlap(chunk_size, content_type)
            
            print(f"使用 chunk_size: {chunk_size}, overlap_size: {overlap_size}")
            
            chunks = []
            
            if split_by == 'char':
                content = content.replace('\n', ' ').strip()
                for i in range(0, len(content), chunk_size - overlap_size):
                    chunk = content[i:i + chunk_size]
                    if chunk.strip():
                        chunks.append(chunk.strip())
                        
            elif split_by == 'line':
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                for i in range(0, len(lines), chunk_size - overlap_size):
                    chunk_lines = lines[i:i + chunk_size]
                    if chunk_lines:
                        chunk = '\n'.join(chunk_lines)
                        chunks.append(chunk)
                        
            elif split_by == 'sentence':
                sentences = re.split(r'[.!?。！？]+', content)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                for i in range(0, len(sentences), chunk_size - overlap_size):
                    chunk_sentences = sentences[i:i + chunk_size]
                    if chunk_sentences:
                        chunk = '。'.join(chunk_sentences) + '。'
                        chunks.append(chunk)
            
            elif split_by == 'paragraph':
                chunks = smart_chunk_text(content, chunk_size, overlap_size, 'paragraph')
            
            # 過濾太短的 chunks
            min_chunk_length = max(50, chunk_size // 10)
            chunks = [chunk for chunk in chunks if len(chunk) >= min_chunk_length]
            
            print(f"生成了 {len(chunks)} 個 chunks")
            return chunks
            
    except Exception as e:
        print(f"替代方法1失敗: {e}")
    
    # 方法2: 建議使用OCR
    print("建議解決方案:")
    print("1. 安裝 pytesseract 進行OCR文字識別")
    print("2. 使用 pdf2image 將PDF轉換為圖片後OCR")
    print("3. 檢查PDF是否為掃描版本")
    
    return []

def import_file_to_pinecone(file_path, text_column=None, chunk_size=None, overlap_size=None, split_by='char', content_type='general'):
    """
    從檔案匯入資料到 Pinecone
    """
    # 初始化 Pinecone
    index = initialize_pinecone()
    if not index:
        return False
    
    # 檢查檔案是否存在
    if not os.path.exists(file_path):
        print(f"檔案不存在: {file_path}")
        return False
    
    file_name = Path(file_path).stem
    file_ext = Path(file_path).suffix.lower()
    
    print(f"正在處理檔案: {file_path}")
    
    # 根據檔案類型讀取資料
    if file_ext == '.txt':
        file_data = read_txt_file(file_path, chunk_size, overlap_size, split_by, content_type)
        if file_data:  # 檢查是否為空
            file_data = [{'text': text, 'metadata': {'chunk_index': i}, 'row_index': i} 
                        for i, text in enumerate(file_data)]
        else:
            file_data = []
    elif file_ext == '.pdf':
        file_data = read_pdf_file(file_path, chunk_size, overlap_size, split_by, content_type)
        if file_data:  # 檢查是否為空
            file_data = [{'text': text, 'metadata': {'chunk_index': i}, 'row_index': i} 
                        for i, text in enumerate(file_data)]
        else:
            file_data = []
    elif file_ext == '.csv':
        file_data = read_csv_file(file_path, text_column)
        if file_data is None:  # 如果返回 None，設為空列表
            file_data = []
    elif file_ext == '.json':
        file_data = read_json_file(file_path)
        if file_data is None:  # 如果返回 None，設為空列表
            file_data = []
    else:
        print(f"不支援的檔案格式: {file_ext}")
        return False
    
    if not file_data:
        print("檔案中沒有有效的資料")
        return False
    
    print(f"從檔案中讀取到 {len(file_data)} 筆資料")
    
    # 處理資料並生成向量
    vectors = process_file_data(file_data, file_name)
    
    if not vectors:
        print("沒有生成任何向量")
        return False
    
    # 上傳到 Pinecone
    success_count = upload_vectors_to_pinecone(index, vectors)
    
    return success_count > 0

def read_csv_file(file_path, text_column=None):
    """
    讀取 CSV 檔案
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # 如果沒有指定文本欄位，自動選擇
        if text_column is None:
            text_columns = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    text_columns.append(col)
            
            if text_columns:
                text_column = text_columns[0]
                print(f"自動選擇文本欄位: {text_column}")
            else:
                print("找不到文本欄位")
                return []
        
        # 將每一行轉換為字典格式
        data = []
        for index, row in df.iterrows():
            text = str(row[text_column])
            metadata = {}
            
            # 將其他欄位作為 metadata
            for col in df.columns:
                if col != text_column:
                    metadata[col] = str(row[col])
            
            data.append({
                'text': text,
                'metadata': metadata,
                'row_index': index
            })
        
        return data
    except Exception as e:
        print(f"讀取 CSV 檔案時發生錯誤: {e}")
        return []

def read_json_file(file_path):
    """
    讀取 JSON 檔案
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # 如果是列表格式
        if isinstance(data, list):
            result = []
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    # 找到文本欄位
                    text_field = None
                    for key in ['text', 'content', 'description', 'message']:
                        if key in item:
                            text_field = key
                            break
                    
                    if text_field:
                        text = str(item[text_field])
                        metadata = {k: v for k, v in item.items() if k != text_field}
                        result.append({
                            'text': text,
                            'metadata': metadata,
                            'row_index': i
                        })
                elif isinstance(item, str):
                    result.append({
                        'text': item,
                        'metadata': {},
                        'row_index': i
                    })
            return result
        
        # 如果是字典格式
        elif isinstance(data, dict):
            return [{
                'text': str(data),
                'metadata': {},
                'row_index': 0
            }]
        
        return []
    except Exception as e:
        print(f"讀取 JSON 檔案時發生錯誤: {e}")
        return []

# 全局模型變數，避免重複加載
_embedding_model = None

def get_embedding_model():
    """
    獲取或初始化嵌入模型
    """
    global _embedding_model
    if _embedding_model is None:
        print("正在加載 Sentence Transformer 模型 (all-roberta-large-v1)...")
        _embedding_model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
        print("模型加載完成")
    return _embedding_model

def create_embeddings(text, dimension=1024):
    """
    使用 Sentence Transformers 創建嵌入向量
    all-roberta-large-v1 模型產生 1024 維向量
    """
    try:
        # 獲取模型
        model = get_embedding_model()
        
        # 生成嵌入向量
        embedding = model.encode(text, convert_to_tensor=False)
        
        # 確保向量是 list 格式
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        
        # 驗證維度
        if len(embedding) != dimension:
            print(f"警告: 期望維度 {dimension}，實際維度 {len(embedding)}")
            
            if len(embedding) > dimension:
                # 截斷向量
                embedding = embedding[:dimension]
                print(f"向量已截斷至 {dimension} 維")
            else:
                # 填充向量
                padding = [0.0] * (dimension - len(embedding))
                embedding.extend(padding)
                print(f"向量已填充至 {dimension} 維")
        
        return embedding
    
    except Exception as e:
        print(f"生成嵌入向量時發生錯誤: {e}")
        print(f"使用 {dimension} 維隨機向量作為備選...")
        return np.random.random(dimension).tolist()

def process_file_data(file_data, file_name):
    """
    處理檔案資料並轉換為向量格式
    """
    vectors = []
    
    # 將中文檔名轉換為 ASCII 安全的格式
    safe_file_name = generate_safe_file_id(file_name)
    
    for item in file_data:
        if isinstance(item, dict):
            text = item.get('text', '')
            metadata = item.get('metadata', {})
            row_index = item.get('row_index', 0)
        else:
            text = str(item)
            metadata = {}
            row_index = 0
        
        if not text.strip():
            continue
        
        # 生成 ASCII 安全的向量 ID
        vector_id = f"{safe_file_name}_{uuid.uuid4().hex[:8]}"
        
        # 生成嵌入向量
        vector_values = create_embeddings(text)
        
        # 準備 metadata
        final_metadata = {
            "text": text,
            "source_file": file_name,
            "length": len(text),
            "created_at": str(int(time.time())),
            "row_index": row_index
        }
        
        # 合併額外的 metadata
        final_metadata.update(metadata)
        
        vector = {
            "id": vector_id,
            "values": vector_values,
            "metadata": final_metadata
        }
        
        vectors.append(vector)
    
    return vectors

def generate_safe_file_id(file_name):
    """
    將檔名轉換為 ASCII 安全的 ID
    """
    # 方法1: 使用 hash
    file_hash = hashlib.md5(file_name.encode('utf-8')).hexdigest()[:12]
    return f"file_{file_hash}"

def upload_vectors_to_pinecone(index, vectors, batch_size=100):
    """
    將向量批次上傳到 Pinecone
    """
    try:
        total_vectors = len(vectors)
        print(f"開始上傳 {total_vectors} 個向量...")
        
        success_count = 0
        
        # 分批上傳
        for i in range(0, total_vectors, batch_size):
            batch = vectors[i:i + batch_size]
            
            try:
                # 上傳向量
                index.upsert(vectors=batch)
                success_count += len(batch)
                
                print(f"已上傳 {min(i + batch_size, total_vectors)}/{total_vectors} 個向量")
                
                # 避免 API 限制
                time.sleep(0.1)
                
            except Exception as e:
                print(f"上傳批次時發生錯誤: {e}")
                continue
        
        print(f"上傳完成！成功上傳 {success_count}/{total_vectors} 個向量")
        return success_count
        
    except Exception as e:
        print(f"上傳向量時發生錯誤: {e}")
        return 0

def get_index_stats():
    """
    獲取 index 統計資訊
    """
    index = initialize_pinecone()
    if not index:
        return None
    
    try:
        stats = index.describe_index_stats()
        print(f"Index 統計資訊:")
        print(f"  總向量數量: {stats.get('total_vector_count', 0)}")
        print(f"  維度: {stats.get('dimension', 'N/A')}")
        if 'namespaces' in stats:
            print(f"  命名空間: {list(stats['namespaces'].keys())}")
        return stats
    except Exception as e:
        print(f"獲取統計資訊時發生錯誤: {e}")
        return None

def main():
    """
    主程式
    """
    print("=== 檔案匯入 Pinecone 向量資料庫程式 ===")
    
    while True:
        print("\n請選擇操作:")
        print("1. 匯入單一檔案")
        print("2. 批次匯入多個檔案")
        print("3. 查看資料庫統計資訊")
        print("4. 退出")
        
        choice = input("請輸入選項 (1-4): ").strip()
        
        if choice == "1":
            # 匯入單一檔案
            file_path = input("請輸入檔案路徑: ").strip()
            
            # 根據檔案類型詢問相關參數
            chunk_size = None
            overlap_size = None
            split_by = 'char'
            content_type = 'general'
            
            if file_path.lower().endswith(('.txt', '.pdf')):
                print("\n=== 文本分割設定 ===")
                chunk_input = input("請輸入 chunk 大小 (留空自動計算): ").strip()
                if chunk_input:
                    chunk_size = int(chunk_input)
                
                overlap_input = input("請輸入 overlap 大小 (留空自動計算): ").strip()
                if overlap_input:
                    overlap_size = int(overlap_input)
                
                print("分割方式選項:")
                print("  char - 按字符分割")
                print("  line - 按行分割")
                print("  sentence - 按句子分割")
                print("  paragraph - 按段落分割")
                split_method = input("請選擇分割方式 (預設 char): ").strip()
                if split_method in ['char', 'line', 'sentence', 'paragraph']:
                    split_by = split_method
                
                print("內容類型選項:")
                print("  general - 一般文本")
                print("  technical - 技術文件")
                print("  academic - 學術文章")
                print("  legal - 法律文件")
                print("  narrative - 故事文本")
                content_type_input = input("請選擇內容類型 (預設 general): ").strip()
                if content_type_input in ['general', 'technical', 'academic', 'legal', 'narrative']:
                    content_type = content_type_input
            
            # 如果是 CSV 檔案，詢問文本欄位
            text_column = None
            if file_path.lower().endswith('.csv'):
                text_column = input("請輸入文本欄位名稱 (留空自動選擇): ").strip()
                if not text_column:
                    text_column = None
            
            success = import_file_to_pinecone(file_path, text_column, chunk_size, overlap_size, split_by, content_type)
            if success:
                print("檔案匯入成功！")
            else:
                print("檔案匯入失敗！")
                
        elif choice == "2":
            # 批次匯入
            folder_path = input("請輸入資料夾路徑: ").strip()
            
            if not os.path.exists(folder_path):
                print("資料夾不存在")
                continue
            
            # 支援的檔案格式
            supported_extensions = ['.txt', '.pdf', '.csv', '.json']
            files = []
            
            for ext in supported_extensions:
                files.extend(Path(folder_path).glob(f"*{ext}"))
            
            if not files:
                print("資料夾中沒有支援的檔案")
                continue
            
            print(f"找到 {len(files)} 個檔案:")
            for file in files:
                print(f"  - {file.name}")
            
            confirm = input("是否繼續匯入? (y/n): ").strip().lower()
            if confirm == 'y':
                success_count = 0
                for file_path in files:
                    print(f"\n處理檔案: {file_path.name}")
                    if import_file_to_pinecone(str(file_path)):
                        success_count += 1
                
                print(f"\n批次匯入完成！成功匯入 {success_count}/{len(files)} 個檔案")
                
        elif choice == "3":
            # 查看統計資訊
            get_index_stats()
            
        elif choice == "4":
            print("退出程式")
            break
            
        else:
            print("無效選項，請重新選擇")

if __name__ == "__main__":
    main()