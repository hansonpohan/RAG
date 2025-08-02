# RAG Flask Web Application

這是一個基於 Flask 的 RAG（Retrieval Augmented Generation）演示應用程序，整合了 Pinecone 向量存儲、Google Gemini 語言模型和 SentenceTransformer 嵌入模型。

## 🚀 功能特點

- **完整的 RAG 流程**：查詢向量化 → 向量檢索 → 上下文生成 → 智能回答
- **現代化 Web 界面**：響應式設計，支持桌面和移動設備
- **實時健康監控**：自動檢查各項服務狀態
- **調試信息顯示**：可查看檢索到的文檔片段和相似度評分
- **錯誤處理**：完善的錯誤處理和用戶反饋機制

## 🏗️ 系統架構

```
用戶查詢 → 向量化 → Pinecone檢索 → 上下文組合 → Gemini生成 → 返回結果
```

### 核心組件

1. **向量嵌入**：SentenceTransformer (`all-MiniLM-L6-v2`)
2. **向量存儲**：Pinecone Vector Database
3. **語言模型**：Google Gemini Pro
4. **Web框架**：Flask

## 📋 系統要求

- Python 3.8+
- 網路連接（用於訪問 Pinecone 和 Gemini API）
- 有效的 Pinecone 和 Google API 金鑰

## 🛠️ 安裝步驟

### 1. 克隆或下載項目

```bash
git clone <repository-url>
cd rag-flask-app
```

### 2. 創建虛擬環境（推薦）

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. 安裝依賴包

```bash
pip install -r requirements.txt
```

### 4. 準備 Pinecone 索引

確保您的 Pinecone 環境中已經創建了索引：

- **索引名稱**：`text`
- **維度**：1024（對應 sentence-transformers/all-roberta-large-v1 模型）
- **距離度量**：cosine
- **雲端區域**：us-east-1

### 5. 配置 API 金鑰

#### 方法一：直接修改代碼（僅用於測試）
在 `app.py` 中更新您的 API 金鑰：

```python
PINECONE_API_KEY = "your-pinecone-api-key"
GEMINI_API_KEY = "your-gemini-api-key"
```

#### 方法二：使用環境變量（推薦）
```bash
# Windows
set PINECONE_API_KEY=your-pinecone-api-key
set GEMINI_API_KEY=your-gemini-api-key

# macOS/Linux
export PINECONE_API_KEY=your-pinecone-api-key
export GEMINI_API_KEY=your-gemini-api-key
```

然後修改 `app.py` 中的配置：
```python
import os
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
```

## 🚦 啟動應用

```bash
python app.py
```

應用將在 `http://localhost:5001` 啟動。

## 📁 項目結構

```
rag-flask-app/
│
├── app.py                 # 主要的 Flask 應用程序
├── requirements.txt       # Python 依賴包清單
├── templates/
│   └── index.html        # 前端 HTML 模板
└── README.md             # 項目說明文檔
```

## 🔧 API 端點

| 端點 | 方法 | 描述 |
|------|------|------|
| `/` | GET | 主頁面 |
| `/query` | POST | 處理 RAG 查詢請求 |
| `/health` | GET | 健康檢查端點 |

### `/query` 請求格式

```json
{
    "query": "您的問題內容"
}
```

### `/query` 回應格式

```json
{
    "success": true,
    "query": "原始查詢",
    "answer": "AI 生成的回答",
    "retrieved_chunks": [
        {
            "text": "檢索到的文檔片段",
            "score": 0.85,
            "id": "文檔ID"
        }
    ],
    "debug_info": {
        "context": "組合後的上下文",
        "num_chunks": 3
    }
}
```

## 🎯 使用方法

1. **打開瀏覽器**訪問 `http://localhost:5001`
2. **輸入問題**到文本框中
3. **點擊「發送查詢」**按鈕
4. **查看結果**：AI 會基於檢索到的相關文檔回答您的問題
5. **查看調試信息**：點擊「顯示/隱藏 調試信息」查看檢索詳情

## 📊 Pinecone 數據準備

### 數據格式要求

您的 Pinecone 索引中的文檔應包含以下元數據結構：

```json
{
    "id": "document_1",
    "values": [0.1, 0.2, ...],  // 1024 維向量
    "metadata": {
        "text": "這是文檔的實際內容文本"
    }
}
```

### 數據上傳示例

```python
from sentence_transformers import SentenceTransformer
import pinecone

# 初始化模型和 Pinecone
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
pc = pinecone.Pinecone(api_key="your-api-key")
index = pc.Index("text")

# 準備文檔
documents = [
    "人工智能是模擬人類智能的技術...",
    "機器學習是人工智能的一個分支...",
    # 更多文檔...
]

# 向量化並上傳
for i, doc in enumerate(documents):
    embedding = model.encode([doc])[0].tolist()
    index.upsert([(f"doc_{i}", embedding, {"text": doc})])
```

## 🔍 故障排除

### 常見問題

1. **服務狀態顯示異常**
   - 檢查 API 金鑰是否正確
   - 確認網路連接正常
   - 查看控制台錯誤日誌

2. **沒有找到相關文檔**
   - 確保 Pinecone 索引中有數據
   - 檢查索引名稱是否正確
   - 確認文檔元數據包含 `text` 字段

3. **Gemini API 錯誤**
   - 檢查 API 金鑰是否有效
   - 確認是否超出使用配額
   - 檢查網路是否能訪問 Google API

### 日誌查看

應用會在控制台輸出詳細的日誌信息，包括：
- 服務初始化狀態
- 查詢處理過程
- 錯誤詳情

## 🛡️ 安全注意事項

1. **API 金鑰保護**：
   - 不要將 API 金鑰提交到版本控制系統
   - 使用環境變量存儲敏感信息
   - 定期輪換 API 金鑰

2. **生產環境部署**：
   - 關閉 Flask 的 debug 模式
   - 使用 HTTPS
   - 添加輸入驗證和速率限制

## 📈 性能優化建議

1. **向量檢索優化**：
   - 調整檢索的文檔片段數量（top_k）
   - 優化向量索引配置
   - 使用更高維度的嵌入模型

2. **回應生成優化**：
   - 調整提示詞模板
   - 控制回應長度
   - 實現結果緩存

## 🤝 貢獻指南

歡迎提交 Issue 和 Pull Request 來改進這個項目！

1. Fork 本項目
2. 創建功能分支
3. 提交更改
4. 發起 Pull Request

## 📄 許可證

本項目採用 MIT 許可證。詳情請參見 LICENSE 文件。

## 📞 技術支持

如果您在使用過程中遇到問題，請：

1. 查看本 README 的故障排除部分
2. 檢查項目的 Issues 頁面
3. 提交新的 Issue 描述您的問題

---

**注意**：這是一個演示項目，主要用於學習和測試 RAG 技術。在生產環境使用前，請確保進行充分的安全性和性能測試。
