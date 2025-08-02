from flask import Flask, request, render_template, jsonify
import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import logging
from werkzeug.utils import secure_filename
import uuid
import time
import hashlib
import re
import numpy as np
from pathlib import Path
import PyPDF2
import fitz  # PyMuPDF
import pandas as pd
import json

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 配置上傳設定
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'csv', 'json'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# 確保上傳資料夾存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# API 金鑰配置
PINECONE_API_KEY = "pcsk_6k1GxA_GGMdrBj2MasWquFDMCnkbz6U4zxRSqpznZue37y2XgANWdBSsagY1zDxKUKdGyP"
GEMINI_API_KEY = "AIzaSyBMmPbOhVBrUOyJ3jnYXdS5prXox25Hgbk"

# Pinecone 設定
pinecone_env = "us-east-1"
index_name = "test"

def allowed_file(filename):
    """檢查檔案格式是否允許"""
    logger.info(f"檢查檔案: {filename}")
    
    if not filename:
        logger.warning(f"檔案名稱為空: {filename}")
        return False
    
    if '.' not in filename:
        logger.warning(f"檔案名稱沒有副檔名: {filename}")
        return False
    
    try:
        ext = filename.rsplit('.', 1)[1].lower()
        logger.info(f"檔案副檔名: {ext}")
        logger.info(f"允許的格式: {ALLOWED_EXTENSIONS}")
        
        is_allowed = ext in ALLOWED_EXTENSIONS
        logger.info(f"檔案格式檢查結果: {is_allowed}")
        
        return is_allowed
    except IndexError as e:
        logger.error(f"解析檔案副檔名時發生錯誤: {e}")
        return False
    except Exception as e:
        logger.error(f"檔案格式檢查時發生未知錯誤: {e}")
        return False

# 從 addToVectorStore.py 導入的函數
def get_recommended_chunk_size(content, split_by='char'):
    """根據內容類型推薦 chunk 大小"""
    content_length = len(content)
    
    if split_by == 'char':
        if content_length < 5000:
            return 500, 50
        elif content_length < 50000:
            return 1000, 100
        else:
            return 1500, 150
    elif split_by == 'line':
        lines = [line for line in content.split('\n') if line.strip()]
        total_lines = len(lines)
        
        if total_lines < 10:
            return 3, 1
        elif total_lines < 100:
            return 10, 2
        elif total_lines < 1000:
            return 20, 3
        else:
            return 30, 5
    elif split_by == 'sentence':
        sentences = [s.strip() for s in re.split(r'[.!?。！？]+', content) if s.strip()]
        total_sentences = len(sentences)
        
        if total_sentences < 10:
            return 3, 1
        elif total_sentences < 50:
            return 5, 1
        elif total_sentences < 200:
            return 10, 2
        else:
            return 15, 3
    elif split_by == 'paragraph':
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        total_paragraphs = len(paragraphs)
        
        if total_paragraphs < 5:
            return 2, 1
        elif total_paragraphs < 20:
            return 5, 1
        else:
            return 10, 2
    
    return 1000, 100

def calculate_optimal_overlap(chunk_size, content_type='general'):
    """計算最佳重疊大小"""
    overlap_ratios = {
        'technical': 0.15,
        'narrative': 0.10,
        'general': 0.12,
        'legal': 0.20,
        'academic': 0.15
    }
    
    ratio = overlap_ratios.get(content_type, 0.12)
    optimal_overlap = int(chunk_size * ratio)
    
    # 確保 overlap_size 不會太大，避免無限循環
    min_overlap = max(1, chunk_size // 20)
    max_overlap = max(1, chunk_size // 3)
    
    return max(min_overlap, min(optimal_overlap, max_overlap))

def clean_extracted_text(text):
    """清理提取的文本"""
    if not text:
        return ""
    
    # 移除控制字符，但保留換行和制表符
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t\r')
    
    # 清理常見的亂碼字符
    replacements = {
        'ï¿½': '',
        '�': '',
        '\x00': '',
        '\ufffd': '',  # Unicode 替換字符
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # 規範化空白字符
    text = re.sub(r'\r\n', '\n', text)  # 統一換行符
    text = re.sub(r'\r', '\n', text)
    text = re.sub(r'\t', ' ', text)  # 制表符轉空格
    text = re.sub(r' +', ' ', text)  # 多個空格合併為一個
    
    # 清理行
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if line and len(line) > 1:  # 過濾掉太短的行
            cleaned_lines.append(line)
    
    # 重新組合
    text = '\n'.join(cleaned_lines)
    
    # 移除多餘的空行
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def is_mostly_garbled(text, threshold=0.3):
    """檢查文本是否主要由亂碼組成"""
    if not text:
        return True
    
    # 統計各種字符類型
    total_chars = len(text)
    if total_chars == 0:
        return True
    
    non_ascii_count = sum(1 for char in text if ord(char) > 127)
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    english_chars = sum(1 for char in text if char.isalpha() and ord(char) < 128)
    digit_chars = sum(1 for char in text if char.isdigit())
    punctuation_chars = sum(1 for char in text if char in '.,!?;:()[]{}"\'-')
    
    # 計算比例
    non_ascii_ratio = non_ascii_count / total_chars
    chinese_ratio = chinese_chars / total_chars
    english_ratio = english_chars / total_chars
    meaningful_ratio = (chinese_chars + english_chars + digit_chars + punctuation_chars) / total_chars
    
    logger.info(f"文本分析 - 總字符: {total_chars}, 中文: {chinese_ratio:.2%}, "
                f"英文: {english_ratio:.2%}, 有意義字符: {meaningful_ratio:.2%}")
    
    # 如果有意義字符比例太低，可能是亂碼
    if meaningful_ratio < 0.3:
        return True
    
    # 如果非 ASCII 字符很多但中文很少，可能是亂碼
    if non_ascii_ratio > threshold and chinese_ratio < 0.1:
        return True
    
    return False

def read_txt_file(file_path, chunk_size=None, overlap_size=None, split_by='char', content_type='general'):
    """讀取 TXT 檔案並智能分割成 chunks"""
    try:
        # 嘗試多種編碼
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'big5', 'latin-1']
        content = ""
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read()
                logger.info(f"成功使用 {encoding} 編碼讀取檔案")
                break
            except UnicodeDecodeError:
                continue
        
        if not content:
            logger.error("無法使用任何編碼讀取檔案")
            return []
        
        # 檢查內容是否為空
        if not content.strip():
            logger.warning("TXT 檔案內容為空")
            return []
        
        logger.info(f"檔案內容長度: {len(content)} 字符")
        
        if chunk_size is None or overlap_size is None:
            recommended_chunk, recommended_overlap = get_recommended_chunk_size(content, split_by)
            chunk_size = chunk_size or recommended_chunk
            overlap_size = overlap_size or calculate_optimal_overlap(chunk_size, content_type)
        
        logger.info(f"使用參數 - chunk_size: {chunk_size}, overlap_size: {overlap_size}, split_by: {split_by}")
        
        # 確保 overlap_size 不會太大，避免無限循環
        overlap_size = min(overlap_size, chunk_size - 1)
        step = max(1, chunk_size - overlap_size)
        
        chunks = []
        
        if split_by == 'char':
            content = content.replace('\n', ' ').strip()
            for i in range(0, len(content), step):
                chunk = content[i:i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk.strip())
                    
        elif split_by == 'line':
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            if not lines:
                logger.warning("檔案中沒有有效的行")
                # 嘗試簡單分割
                if content.strip():
                    simple_chunks = [content[i:i+1000] for i in range(0, len(content), 800)]
                    chunks = [chunk.strip() for chunk in simple_chunks if chunk.strip()]
                    logger.info(f"使用簡單分割，生成 {len(chunks)} 個 chunks")
                return chunks
            
            for i in range(0, len(lines), step):
                chunk_lines = lines[i:i + chunk_size]
                if chunk_lines:
                    chunk = '\n'.join(chunk_lines)
                    if chunk.strip():
                        chunks.append(chunk)
                    
        elif split_by == 'sentence':
            sentences = re.split(r'[.!?。！？]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                logger.warning("檔案中沒有有效的句子")
                # 嘗試簡單分割
                if content.strip():
                    simple_chunks = [content[i:i+1000] for i in range(0, len(content), 800)]
                    chunks = [chunk.strip() for chunk in simple_chunks if chunk.strip()]
                    logger.info(f"使用簡單分割，生成 {len(chunks)} 個 chunks")
                return chunks
            
            for i in range(0, len(sentences), step):
                chunk_sentences = sentences[i:i + chunk_size]
                if chunk_sentences:
                    chunk = '。'.join(chunk_sentences) + '。'
                    if chunk.strip():
                        chunks.append(chunk)
        
        elif split_by == 'paragraph':
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            if not paragraphs:
                logger.warning("檔案中沒有有效的段落")
                # 嘗試簡單分割
                if content.strip():
                    simple_chunks = [content[i:i+1000] for i in range(0, len(content), 800)]
                    chunks = [chunk.strip() for chunk in simple_chunks if chunk.strip()]
                    logger.info(f"使用簡單分割，生成 {len(chunks)} 個 chunks")
                return chunks
            
            for i in range(0, len(paragraphs), step):
                chunk_paragraphs = paragraphs[i:i + chunk_size]
                if chunk_paragraphs:
                    chunk = '\n\n'.join(chunk_paragraphs)
                    if chunk.strip():
                        chunks.append(chunk)
        
        # 過濾太短的 chunks，使用更寬鬆的標準
        min_chunk_length = max(10, chunk_size // 20)
        filtered_chunks = [chunk for chunk in chunks if len(chunk) >= min_chunk_length]
        
        logger.info(f"原始 chunks 數量: {len(chunks)}, 過濾後: {len(filtered_chunks)}")
        
        # 如果過濾後沒有 chunks，嘗試更寬鬆的過濾標準
        if not filtered_chunks and chunks:
            min_chunk_length = 5
            filtered_chunks = [chunk for chunk in chunks if len(chunk) >= min_chunk_length]
            logger.info(f"使用更寬鬆標準後的 chunks 數量: {len(filtered_chunks)}")
        
        # 最後的保險措施：如果還是沒有 chunks，強制分割
        if not filtered_chunks and content.strip():
            logger.warning("所有分割方法都失敗，使用強制分割")
            simple_chunks = [content[i:i+500] for i in range(0, len(content), 400)]
            filtered_chunks = [chunk.strip() for chunk in simple_chunks if len(chunk.strip()) >= 10]
            logger.info(f"強制分割生成 {len(filtered_chunks)} 個 chunks")
        
        return filtered_chunks
        
    except Exception as e:
        logger.error(f"讀取 TXT 檔案時發生錯誤: {e}")
        return []

def read_pdf_file(file_path, chunk_size=None, overlap_size=None, split_by='char', content_type='general', method='pymupdf'):
    """讀取 PDF 檔案並分割成 chunks"""
    try:
        content = ""
        
        logger.info(f"開始處理 PDF 檔案: {file_path}")
        
        # 檢查文件是否為有效的 PDF
        if not file_path.lower().endswith('.pdf'):
            logger.error(f"檔案不是 PDF 格式: {file_path}")
            return []
        
        # 檢查檔案大小
        try:
            file_size = os.path.getsize(file_path)
            logger.info(f"PDF 檔案大小: {file_size} bytes")
            if file_size == 0:
                logger.error("PDF 檔案為空")
                return []
        except OSError as e:
            logger.error(f"無法讀取檔案大小: {e}")
            return []
        
        if method == 'pymupdf':
            try:
                # 檢查 fitz 是否可用
                if not hasattr(fitz, 'open'):
                    logger.error("PyMuPDF (fitz) 模組未正確安裝")
                    method = 'pypdf2'
                else:
                    doc = fitz.open(file_path)
                    logger.info(f"成功打開 PDF，共 {doc.page_count} 頁")
                    
                    if doc.page_count == 0:
                        logger.warning("PDF 檔案沒有頁面")
                        doc.close()
                        return []
                    
                    for page_num in range(doc.page_count):
                        try:
                            page = doc[page_num]
                            page_text = page.get_text(sort=True)  # 添加 sort=True 改善文字順序
                            
                            if page_text and page_text.strip():
                                # 清理頁面文字
                                page_text = ' '.join(page_text.split())
                                content += page_text + "\n"
                                logger.info(f"頁面 {page_num + 1} 提取了 {len(page_text)} 個字符")
                            else:
                                logger.warning(f"頁面 {page_num + 1} 沒有文字內容")
                        except Exception as page_error:
                            logger.error(f"處理頁面 {page_num + 1} 時發生錯誤: {page_error}")
                            continue
                    
                    doc.close()
            except Exception as e:
                logger.error(f"使用 PyMuPDF 處理 PDF 失敗: {e}")
                method = 'pypdf2'  # 切換到備用方法
        
        # 如果 PyMuPDF 失敗，使用 PyPDF2
        if method == 'pypdf2' or not content.strip():
            try:
                logger.info("嘗試使用 PyPDF2 處理 PDF")
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    
                    # 檢查 PDF 是否加密
                    if pdf_reader.is_encrypted:
                        logger.error("PDF 檔案已加密，無法處理")
                        return []
                    
                    logger.info(f"PyPDF2 檢測到 {len(pdf_reader.pages)} 頁")
                    
                    for page_num in range(len(pdf_reader.pages)):
                        try:
                            page = pdf_reader.pages[page_num]
                            page_text = page.extract_text()
                            if page_text and page_text.strip():
                                content += page_text + "\n"
                                logger.info(f"PyPDF2 頁面 {page_num + 1} 提取了 {len(page_text)} 個字符")
                            else:
                                logger.warning(f"PyPDF2 頁面 {page_num + 1} 沒有文字內容")
                        except Exception as page_error:
                            logger.error(f"PyPDF2 處理頁面 {page_num + 1} 時發生錯誤: {page_error}")
                            continue
                            
            except Exception as e2:
                logger.error(f"PyPDF2 也失敗: {e2}")
                return []
        
        logger.info(f"原始提取內容長度: {len(content)}")
        
        if not content.strip():
            logger.warning("PDF 檔案中沒有可提取的文本內容")
            return []
        
        # 清理文本
        content = clean_extracted_text(content)
        logger.info(f"清理後內容長度: {len(content)}")
        
        if not content.strip():
            logger.warning("清理後 PDF 檔案沒有有效內容")
            return []
        
        # 檢查是否為亂碼
        if is_mostly_garbled(content):
            logger.warning("檢測到可能的亂碼，但仍會嘗試處理")
            # 可以在這裡添加額外的處理邏輯
        
        # 設置默認參數
        if chunk_size is None or overlap_size is None:
            recommended_chunk, recommended_overlap = get_recommended_chunk_size(content, split_by)
            chunk_size = chunk_size or recommended_chunk
            overlap_size = overlap_size or calculate_optimal_overlap(chunk_size, content_type)
        
        logger.info(f"使用參數 - chunk_size: {chunk_size}, overlap_size: {overlap_size}, split_by: {split_by}")
        
        # 確保參數有效
        overlap_size = min(overlap_size, chunk_size - 1)
        step = max(1, chunk_size - overlap_size)
        
        chunks = []
        
        # 根據分割方式處理
        if split_by == 'char':
            content = content.replace('\n', ' ').strip()
            for i in range(0, len(content), step):
                chunk = content[i:i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk.strip())
                    
        elif split_by == 'line':
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            if not lines:
                logger.warning("PDF 中沒有有效的行")
                if content.strip():
                    simple_chunks = [content[i:i+1000] for i in range(0, len(content), 800)]
                    chunks = [chunk.strip() for chunk in simple_chunks if chunk.strip()]
                    logger.info(f"使用簡單分割，生成 {len(chunks)} 個 chunks")
                return chunks
            
            for i in range(0, len(lines), step):
                chunk_lines = lines[i:i + chunk_size]
                if chunk_lines:
                    chunk = '\n'.join(chunk_lines)
                    if chunk.strip():
                        chunks.append(chunk)
                        
        elif split_by == 'sentence':
            sentences = re.split(r'[.!?。！？]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                logger.warning("PDF 中沒有有效的句子")
                if content.strip():
                    simple_chunks = [content[i:i+1000] for i in range(0, len(content), 800)]
                    chunks = [chunk.strip() for chunk in simple_chunks if chunk.strip()]
                    logger.info(f"使用簡單分割，生成 {len(chunks)} 個 chunks")
                return chunks
            
            for i in range(0, len(sentences), step):
                chunk_sentences = sentences[i:i + chunk_size]
                if chunk_sentences:
                    chunk = '。'.join(chunk_sentences) + '。'
                    if chunk.strip():
                        chunks.append(chunk)
        
        elif split_by == 'paragraph':
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            if not paragraphs:
                logger.warning("PDF 中沒有有效的段落")
                if content.strip():
                    simple_chunks = [content[i:i+1000] for i in range(0, len(content), 800)]
                    chunks = [chunk.strip() for chunk in simple_chunks if chunk.strip()]
                    logger.info(f"使用簡單分割，生成 {len(chunks)} 個 chunks")
                return chunks
            
            for i in range(0, len(paragraphs), step):
                chunk_paragraphs = paragraphs[i:i + chunk_size]
                if chunk_paragraphs:
                    chunk = '\n\n'.join(chunk_paragraphs)
                    if chunk.strip():
                        chunks.append(chunk)
        
        # 過濾過短的 chunks
        min_chunk_length = max(10, chunk_size // 20)
        filtered_chunks = [chunk for chunk in chunks if len(chunk) >= min_chunk_length]
        
        logger.info(f"PDF 原始 chunks 數量: {len(chunks)}, 過濾後: {len(filtered_chunks)}")
        
        # 如果過濾後沒有 chunks，放寬標準
        if not filtered_chunks and chunks:
            min_chunk_length = 5
            filtered_chunks = [chunk for chunk in chunks if len(chunk) >= min_chunk_length]
            logger.info(f"PDF 使用更寬鬆標準後的 chunks 數量: {len(filtered_chunks)}")
        
        # 最後的保險措施
        if not filtered_chunks and content.strip():
            logger.warning("所有分割方法都失敗，使用強制分割")
            simple_chunks = [content[i:i+500] for i in range(0, len(content), 400)]
            filtered_chunks = [chunk.strip() for chunk in simple_chunks if len(chunk.strip()) >= 10]
            logger.info(f"強制分割生成 {len(filtered_chunks)} 個 chunks")
        
        return filtered_chunks
        
    except Exception as e:
        logger.error(f"讀取 PDF 檔案時發生錯誤: {e}")
        import traceback
        logger.error(f"詳細錯誤: {traceback.format_exc()}")
        return []

def read_csv_file(file_path, text_column=None):
    """讀取 CSV 檔案"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        
        if text_column is None:
            text_columns = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    text_columns.append(col)
            
            if text_columns:
                text_column = text_columns[0]
            else:
                return []
        
        data = []
        for index, row in df.iterrows():
            text = str(row[text_column])
            metadata = {}
            
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
        logger.error(f"讀取 CSV 檔案時發生錯誤: {e}")
        return []

def read_json_file(file_path):
    """讀取 JSON 檔案"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        if isinstance(data, list):
            result = []
            for i, item in enumerate(data):
                if isinstance(item, dict):
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
        
        elif isinstance(data, dict):
            return [{
                'text': str(data),
                'metadata': {},
                'row_index': 0
            }]
        
        return []
    except Exception as e:
        logger.error(f"讀取 JSON 檔案時發生錯誤: {e}")
        return []

def generate_safe_file_id(file_name):
    """將檔名轉換為 ASCII 安全的 ID"""
    file_hash = hashlib.md5(file_name.encode('utf-8')).hexdigest()[:12]
    return f"file_{file_hash}"

class RAGService:
    def __init__(self):
        self.sentence_model = None
        self.pinecone_index = None
        self.gemini_model = None
        self.initialize_services()
    
    def initialize_services(self):
        try:
            logger.info("正在初始化 SentenceTransformer...")
            self.sentence_model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
            
            logger.info("正在初始化 Pinecone...")
            pc = Pinecone(api_key=PINECONE_API_KEY)
            
            existing_indexes = pc.list_indexes()
            index_exists = any(idx.name == index_name for idx in existing_indexes)
            
            if not index_exists:
                logger.warning(f"索引 '{index_name}' 不存在，請確保已創建該索引")
            
            self.pinecone_index = pc.Index(index_name)
            
            logger.info("正在初始化 Gemini...")
            genai.configure(api_key=GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel('models/gemini-2.5-flash')
            
            logger.info("所有服務初始化完成")
            
        except Exception as e:
            logger.error(f"服務初始化失敗: {str(e)}")
            raise e
    
    def vectorize_query(self, query):
        """將查詢向量化"""
        try:
            embedding = self.sentence_model.encode([query])
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"查詢向量化失敗: {str(e)}")
            raise e
    
    def create_embeddings(self, text, dimension=1024):
        """創建嵌入向量"""
        try:
            embedding = self.sentence_model.encode(text, convert_to_tensor=False)
            
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            if len(embedding) != dimension:
                if len(embedding) > dimension:
                    embedding = embedding[:dimension]
                else:
                    padding = [0.0] * (dimension - len(embedding))
                    embedding.extend(padding)
            
            return embedding
        
        except Exception as e:
            logger.error(f"生成嵌入向量時發生錯誤: {e}")
            return np.random.random(dimension).tolist()
    
    def process_file_data(self, file_data, file_name):
        """處理檔案資料並轉換為向量格式"""
        vectors = []
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
            
            vector_id = f"{safe_file_name}_{uuid.uuid4().hex[:8]}"
            vector_values = self.create_embeddings(text)
            
            final_metadata = {
                "text": text,
                "source_file": file_name,
                "length": len(text),
                "created_at": str(int(time.time())),
                "row_index": row_index
            }
            
            final_metadata.update(metadata)
            
            vector = {
                "id": vector_id,
                "values": vector_values,
                "metadata": final_metadata
            }
            
            vectors.append(vector)
        
        return vectors
    
    def upload_vectors_to_pinecone(self, vectors, batch_size=100):
        """將向量批次上傳到 Pinecone"""
        try:
            total_vectors = len(vectors)
            success_count = 0
            
            for i in range(0, total_vectors, batch_size):
                batch = vectors[i:i + batch_size]
                
                try:
                    self.pinecone_index.upsert(vectors=batch)
                    success_count += len(batch)
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"上傳批次時發生錯誤: {e}")
                    continue
            
            return success_count
            
        except Exception as e:
            logger.error(f"上傳向量時發生錯誤: {e}")
            return 0
    
    def import_file_to_pinecone(self, file_path, **kwargs):
        """從檔案匯入資料到 Pinecone"""
        try:
            if not os.path.exists(file_path):
                return False, "檔案不存在"
            
            file_name = Path(file_path).stem
            file_ext = Path(file_path).suffix.lower()
            
            file_data = []
            
            if file_ext == '.txt':
                chunks = read_txt_file(file_path, **kwargs)
                if chunks:
                    file_data = [{'text': text, 'metadata': {'chunk_index': i}, 'row_index': i} 
                               for i, text in enumerate(chunks)]
            elif file_ext == '.pdf':
                chunks = read_pdf_file(file_path, **kwargs)
                if chunks:
                    file_data = [{'text': text, 'metadata': {'chunk_index': i}, 'row_index': i} 
                               for i, text in enumerate(chunks)]
            elif file_ext == '.csv':
                file_data = read_csv_file(file_path, kwargs.get('text_column'))
            elif file_ext == '.json':
                file_data = read_json_file(file_path)
            else:
                return False, f"不支援的檔案格式: {file_ext}"
            
            if not file_data:
                return False, "檔案中沒有有效的資料"
            
            vectors = self.process_file_data(file_data, file_name)
            
            if not vectors:
                return False, "沒有生成任何向量"
            
            success_count = self.upload_vectors_to_pinecone(vectors)
            
            if success_count > 0:
                return True, f"成功上傳 {success_count} 個向量"
            else:
                return False, "向量上傳失敗"
                
        except Exception as e:
            logger.error(f"匯入檔案時發生錯誤: {e}")
            return False, f"匯入檔案時發生錯誤: {str(e)}"
    
    def retrieve_similar_chunks(self, query_vector, top_k=3):
        """從 Pinecone 檢索最相似的文字塊"""
        try:
            results = self.pinecone_index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )
            
            chunks = []
            for match in results.matches:
                if 'text' in match.metadata:
                    chunks.append({
                        'text': match.metadata['text'],
                        'score': match.score,
                        'id': match.id
                    })
            
            return chunks
        except Exception as e:
            logger.error(f"檢索相似塊失敗: {str(e)}")
            raise e
    
    def generate_context(self, chunks):
        """將檢索到的文字塊組合成上下文"""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"文檔片段 {i} (相似度: {chunk['score']:.4f}):\n{chunk['text']}")
        
        return "\n\n".join(context_parts)
    
    def create_prompt(self, query, context):
        """創建完整的提示詞"""
        prompt = f"""你是一個專業的知識庫助手，擅長分析文檔並提供準確、有用的回答。請仔細閱讀以下上下文資訊，並基於這些資訊來回答用戶的問題。

        上下文資訊：
        {context}

        用戶問題：{query}

        回答指引：
        1. 優先使用上下文中的資訊來回答問題
        2. 如果上下文包含相關資訊，請：
        - 直接引用或概括相關內容
        - 在回答中明確指出資訊來源（如："根據文檔片段X"）
        - 確保回答與上下文保持一致
        3. 如果上下文資訊不足或不相關，請：
        - 明確說明上下文中缺少相關資訊
        - 說明無法基於提供的資料回答問題
        - 避免編造或推測未在上下文中提及的內容
        4. 回答格式要求：
        - 使用繁體中文
        - 結構清晰、邏輯分明
        - 提供具體、可操作的資訊
        - 如有必要，可以提出進一步詢問的建議

        請根據以上指引提供準確且有幫助的回答："""
        
        return prompt
    
    def generate_response(self, prompt):
        """使用 Gemini 生成回應"""
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"生成回應失敗: {str(e)}")
            raise e
    
    def process_query(self, query):
        """處理完整的 RAG 流程"""
        try:
            query_vector = self.vectorize_query(query)
            similar_chunks = self.retrieve_similar_chunks(query_vector, top_k=3)
            
            if not similar_chunks:
                return "抱歉，我在知識庫中沒有找到與您問題相關的資訊。請您重新表述問題或提供更多詳細資訊。"
            
            context = self.generate_context(similar_chunks)
            prompt = self.create_prompt(query, context)
            response = self.generate_response(prompt)
            
            return {
                'answer': response,
                'retrieved_chunks': similar_chunks,
                'context': context
            }
            
        except Exception as e:
            logger.error(f"處理查詢時發生錯誤: {str(e)}")
            return f"處理查詢時發生錯誤: {str(e)}"

# 初始化 RAG 服務
rag_service = RAGService()

@app.route('/')
def index():
    """主頁面"""
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def handle_query():
    """處理查詢請求"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': '請提供查詢內容'}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({'error': '查詢內容不能為空'}), 400
        
        result = rag_service.process_query(query)
        
        if isinstance(result, str):
            return jsonify({'error': result}), 500
        
        return jsonify({
            'success': True,
            'query': query,
            'answer': result['answer'],
            'retrieved_chunks': result['retrieved_chunks'],
            'debug_info': {
                'context': result['context'],
                'num_chunks': len(result['retrieved_chunks'])
            }
        })
        
    except Exception as e:
        logger.error(f"處理請求時發生錯誤: {str(e)}")
        return jsonify({'error': f'處理請求時發生錯誤: {str(e)}'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """處理檔案上傳請求"""
    try:
        # 詳細檢查請求
        logger.info(f"收到上傳請求")
        logger.info(f"請求方法: {request.method}")
        logger.info(f"請求檔案: {request.files}")
        
        if 'file' not in request.files:
            logger.error("請求中沒有 'file' 欄位")
            return jsonify({'error': '沒有選擇檔案'}), 400
        
        file = request.files['file']
        
        # 詳細檢查檔案物件
        logger.info(f"檔案物件: {file}")
        logger.info(f"檔案名稱: '{file.filename}'")
        logger.info(f"檔案名稱類型: {type(file.filename)}")
        logger.info(f"檔案名稱長度: {len(file.filename) if file.filename else 0}")
        
        if not file.filename or file.filename == '':
            logger.error(f"檔案名稱為空或無效: '{file.filename}'")
            return jsonify({'error': '沒有選擇檔案或檔案名稱無效'}), 400
        
        # 檢查檔案大小
        try:
            file.seek(0, 2)  # 移到檔案末尾
            file_size = file.tell()
            file.seek(0)  # 重置檔案指標
            logger.info(f"檔案大小: {file_size} bytes")
        except Exception as size_error:
            logger.error(f"檢查檔案大小時發生錯誤: {size_error}")
            return jsonify({'error': '無法檢查檔案大小'}), 400
        
        if file_size > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({
                'error': f'檔案過大。最大允許大小: {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB'
            }), 400
        
        if file_size == 0:
            return jsonify({'error': '檔案為空'}), 400
        
        # 檔案格式檢查
        logger.info(f"開始檔案格式檢查...")
        if not allowed_file(file.filename):
            error_msg = f'不支援的檔案格式。檔名: "{file.filename}"，支援格式: {list(ALLOWED_EXTENSIONS)}'
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 400
        
        # 保存檔案
        try:
            filename = secure_filename(file.filename)
            logger.info(f"安全檔名: '{filename}'")
            
            if not filename:
                logger.error("secure_filename 返回空值")
                return jsonify({'error': '檔案名稱無效，無法處理'}), 400
            
            timestamp = int(time.time())
            safe_filename = f"{timestamp}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
            logger.info(f"儲存路徑: {file_path}")
            
            file.save(file_path)
            logger.info(f"檔案已保存到: {file_path}")
            
        except Exception as save_error:
            logger.error(f"保存檔案失敗: {save_error}")
            return jsonify({'error': f'保存檔案失敗: {str(save_error)}'}), 500
        
        # 驗證檔案是否成功保存
        if not os.path.exists(file_path):
            logger.error("檔案保存後不存在")
            return jsonify({'error': '檔案保存失敗'}), 500
        
        # 檢查實際檔案大小
        actual_size = os.path.getsize(file_path)
        logger.info(f"實際保存的檔案大小: {actual_size} bytes")
        
        if actual_size == 0:
            try:
                os.remove(file_path)
            except:
                pass
            return jsonify({'error': '保存的檔案為空'}), 400
        
        # 獲取上傳參數
        chunk_size = request.form.get('chunk_size')
        overlap_size = request.form.get('overlap_size')
        split_by = request.form.get('split_by', 'char')
        content_type = request.form.get('content_type', 'general')
        text_column = request.form.get('text_column')
        
        logger.info(f"處理參數 - chunk_size: {chunk_size}, overlap_size: {overlap_size}, split_by: {split_by}")
        
        # 驗證參數
        kwargs = {
            'split_by': split_by,
            'content_type': content_type
        }
        
        try:
            if chunk_size:
                chunk_size_int = int(chunk_size)
                if chunk_size_int <= 0:
                    raise ValueError("chunk_size 必須大於 0")
                kwargs['chunk_size'] = chunk_size_int
                
            if overlap_size:
                overlap_size_int = int(overlap_size)
                if overlap_size_int < 0:
                    raise ValueError("overlap_size 不能小於 0")
                kwargs['overlap_size'] = overlap_size_int
                
        except ValueError as ve:
            # 清理檔案
            try:
                os.remove(file_path)
            except:
                pass
            return jsonify({'error': f'參數錯誤: {str(ve)}'}), 400
        
        if text_column:
            kwargs['text_column'] = text_column
        
        # 匯入檔案到 Pinecone
        logger.info(f"開始處理檔案: {safe_filename}")
        success, message = rag_service.import_file_to_pinecone(file_path, **kwargs)
        
        # 清理上傳的檔案
        try:
            os.remove(file_path)
            logger.info(f"已清理臨時檔案: {file_path}")
        except Exception as cleanup_error:
            logger.warning(f"清理檔案失敗: {cleanup_error}")
        
        if success:
            logger.info(f"檔案處理成功: {message}")
            return jsonify({
                'success': True,
                'message': message,
                'filename': file.filename,
                'file_size': file_size
            })
        else:
            logger.error(f"檔案處理失敗: {message}")
            return jsonify({
                'error': message
            }), 500
            
    except Exception as e:
        logger.error(f"上傳檔案時發生錯誤: {str(e)}")
        import traceback
        logger.error(f"詳細錯誤: {traceback.format_exc()}")
        return jsonify({'error': f'上傳檔案時發生錯誤: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """健康檢查端點"""
    try:
        status = {
            'sentence_transformer': rag_service.sentence_model is not None,
            'pinecone': rag_service.pinecone_index is not None,
            'gemini': rag_service.gemini_model is not None
        }
        
        all_healthy = all(status.values())
        
        return jsonify({
            'healthy': all_healthy,
            'services': status
        }), 200 if all_healthy else 503
        
    except Exception as e:
        return jsonify({
            'healthy': False,
            'error': str(e)
        }), 503

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)