# PDF MCP Server - Complete Documentation

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation & Setup](#installation--setup)
- [Core PDF Tools](#core-pdf-tools)
- [Text Extraction & OCR](#text-extraction--ocr)
- [LLM-Powered OCR](#llm-powered-ocr)
- [Advanced Features](#advanced-features)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Performance Optimization](#performance-optimization)

## 📖 Overview

The **PDF MCP Server** is a comprehensive Model Context Protocol (MCP) server that provides advanced PDF manipulation, text extraction, and OCR capabilities. It combines traditional PDF processing tools (PDFtk, QPDF) with modern LLM-powered OCR for maximum accuracy and flexibility.

### Key Capabilities
- **20+ PDF manipulation tools** for merging, splitting, encryption, optimization
- **Multi-method text extraction** with intelligent fallback systems
- **LLM-powered OCR** using OpenRouter (Gemini 2.5 Flash) and Mistral
- **Smart hybrid processing** for cost-optimized text extraction
- **Advanced caching** with document fingerprinting
- **Async/parallel processing** with proper resource management

## 🚀 Features

### ✅ Core PDF Operations
- Merge multiple PDFs
- Split PDFs into individual pages
- Extract specific page ranges
- Rotate pages (90°, 180°, 270°)
- Encrypt/decrypt with AES-256
- Optimize for web viewing
- Update metadata fields
- Extract attachments
- Repair damaged PDFs
- Integrity checking

### ✅ Text Extraction Methods
1. **Fast Extraction** - PyMuPDF for speed
2. **Layout-Aware** - pdfplumber for tables/structure
3. **Traditional OCR** - Tesseract for scanned documents
4. **LLM OCR** - OpenRouter/Mistral for maximum accuracy
5. **Smart Hybrid** - Automatic method selection with cost optimization

### ✅ Advanced Features
- **Intelligent caching** (24h text, 30d LLM OCR)
- **Document handle management** (prevents "document closed" errors)
- **Batch processing** with rate limiting
- **Progress tracking** with detailed metrics
- **Error recovery** with graceful fallbacks
- **Multi-provider LLM support**

## 🛠️ Installation & Setup

### Prerequisites
```bash
# System dependencies
sudo apt-get install pdftk qpdf tesseract-ocr

# Python dependencies
pip install -r requirements.txt
```

### Dependencies
```
# Core MCP Framework
fastmcp==2.10.1
mcp==1.10.1

# PDF Processing
pymupdf==1.25.2
pdfplumber==0.12.0
pillow==11.1.0

# OCR
pytesseract==0.3.14

# LLM OCR
mistralai>=1.9.3
httpx==0.28.1
aiofiles==24.1.0
```

### MCP Configuration

**Basic Setup:**
```json
{
  "pdf-tools": {
    "command": "python3",
    "args": ["/path/to/pdf-mcp-server/server.py"],
    "disabledTools": []
  }
}
```

**With LLM OCR (Recommended):**
```json
{
  "pdf-tools": {
    "command": "python3",
    "args": ["/path/to/pdf-mcp-server/server.py"],
    "env": {
      "OPENROUTER_API_KEY": "${OPENROUTER_API_KEY}",
      "OPENROUTER_OCR_ENABLED": "true"
    },
    "disabledTools": []
  }
}
```

### Environment Variables
```bash
# LLM OCR Configuration
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"
export OPENROUTER_OCR_ENABLED="true"

# Alternative: Mistral OCR
export MISTRAL_API_KEY="your-mistral-key"
export MISTRAL_OCR_ENABLED="true"

# Optional: Custom workspace
export PDF_WORKSPACE="/path/to/pdf/workspace"
```

## 📁 Core PDF Tools

### Document Manipulation

#### `merge_pdfs(input_files, output_file)`
Merge multiple PDF files into one.
```python
# Example usage
result = merge_pdfs(
    input_files=["doc1.pdf", "doc2.pdf", "doc3.pdf"],
    output_file="merged_document.pdf"
)
```

#### `split_pdf(input_file, output_dir?)`
Split PDF into individual pages.
```python
result = split_pdf(
    input_file="document.pdf",
    output_dir="split_pages/"  # Optional
)
```

#### `extract_pages(input_file, page_range, output_file)`
Extract specific pages from PDF.
```python
result = extract_pages(
    input_file="document.pdf",
    page_range="1-5,10,15-20",
    output_file="extracted_pages.pdf"
)
```

#### `rotate_pages(input_file, rotation, output_file, page_range?)`
Rotate PDF pages.
```python
result = rotate_pages(
    input_file="document.pdf",
    rotation="90",  # or "right", "left", "down"
    output_file="rotated.pdf",
    page_range="1-3"  # Optional
)
```

### Security Operations

#### `encrypt_pdf(input_file, output_file, user_password, owner_password?)`
Encrypt PDF with AES-256 encryption.
```python
result = encrypt_pdf(
    input_file="document.pdf",
    output_file="encrypted.pdf",
    user_password="user123",
    owner_password="owner456"  # Optional
)
```

#### `decrypt_pdf(input_file, output_file, password)`
Decrypt password-protected PDF.
```python
result = decrypt_pdf(
    input_file="encrypted.pdf",
    output_file="decrypted.pdf",
    password="user123"
)
```

### Information & Analysis

#### `get_pdf_info(input_file)`
Get comprehensive PDF information.
```python
result = get_pdf_info("document.pdf")
# Returns: pages, metadata, structure, file info
```

#### `inspect_pdf_structure(input_file, detailed?)`
Inspect PDF internal structure.
```python
result = inspect_pdf_structure(
    input_file="document.pdf",
    detailed=True
)
```

#### `check_pdf_integrity(input_file)`
Check PDF for errors and warnings.
```python
result = check_pdf_integrity("document.pdf")
```

### Optimization & Repair

#### `optimize_pdf(input_file, output_file, compression_level?)`
Optimize PDF for web viewing.
```python
result = optimize_pdf(
    input_file="large_document.pdf",
    output_file="optimized.pdf",
    compression_level="high"  # "low", "medium", "high"
)
```

#### `repair_pdf(input_file, output_file)`
Repair damaged PDF file.
```python
result = repair_pdf(
    input_file="corrupted.pdf",
    output_file="repaired.pdf"
)
```

## 📝 Text Extraction & OCR

### Fast Text Extraction

#### `extract_text_fast(input_file, pages?)`
High-speed text extraction using PyMuPDF.
```python
result = extract_text_fast(
    input_file="document.pdf",
    pages=[0, 1, 2]  # Optional: specific pages
)
```

**Features:**
- ⚡ **Fastest method** (~1s per page)
- 📄 **Direct text extraction** from PDF structure
- 🎯 **Best for** text-based PDFs
- 💾 **Caches** results for 24 hours

### Layout-Aware Extraction

#### `extract_text_layout_aware(input_file, pages?)`
Structure-preserving extraction with table detection.
```python
result = extract_text_layout_aware(
    input_file="report_with_tables.pdf",
    pages=[0, 1]
)
```

**Features:**
- 📊 **Table detection** and preservation
- 🏗️ **Layout awareness** maintains formatting
- 📋 **Structured output** with table data
- 🎯 **Best for** reports, forms, structured documents

### Traditional OCR

#### `extract_text_ocr(input_file, pages?, language?, dpi?)`
Tesseract-based OCR for scanned documents.
```python
result = extract_text_ocr(
    input_file="scanned_document.pdf",
    pages=[0, 1],
    language="eng",  # OCR language
    dpi=300  # Image resolution
)
```

**Features:**
- 📷 **Image-to-text** conversion
- 🌍 **Multi-language** support
- ⚙️ **Configurable DPI** for quality
- 📊 **Confidence scores** for each page
- 🔄 **Parallel processing** (4 workers)

### Smart Text Extraction

#### `extract_text_smart(input_file, pages?, ocr_fallback?)`
Intelligent method selection with automatic fallback.
```python
result = extract_text_smart(
    input_file="mixed_document.pdf",
    pages=None,  # All pages
    ocr_fallback=True
)
```

**Logic Flow:**
1. 🔍 **Content detection** - Analyzes text availability
2. ⚡ **Fast extraction** - Tries PyMuPDF first
3. 🏗️ **Layout-aware** - Falls back to pdfplumber
4. 📷 **OCR** - Uses Tesseract if needed

## 🤖 LLM-Powered OCR

### LLM OCR Extraction

#### `extract_text_llm_ocr(input_file, pages?, provider?, model?, custom_prompt?)`
High-accuracy OCR using Large Language Models.
```python
result = await extract_text_llm_ocr(
    input_file="complex_document.pdf",
    pages=[0, 1],
    provider="openrouter",  # "openrouter", "mistral"
    model="google/gemini-2.5-flash-lite-preview-06-17",
    custom_prompt="Extract all text including handwritten notes"
)
```

**Supported Providers:**

#### OpenRouter (Recommended)
- 🤖 **Model**: `google/gemini-2.5-flash-lite-preview-06-17`
- 💰 **Cost**: ~$0.075 per 1M tokens
- ⚡ **Speed**: 2-4 seconds per page
- 🎯 **Accuracy**: 95%+ for complex documents
- 🔄 **Rate limits**: 10 concurrent requests

#### Mistral
- 🤖 **Model**: `mistral-ocr-latest`
- 💰 **Cost**: $1 per 1000 pages
- ⚡ **Speed**: 3-5 seconds per page
- 🎯 **Accuracy**: 90%+ for standard documents
- 🔄 **Rate limits**: 5 concurrent requests

### Smart Hybrid Processing

#### `extract_text_hybrid_smart(input_file, pages?, llm_provider?)`
Cost-optimized extraction with intelligent page routing.
```python
result = await extract_text_hybrid_smart(
    input_file="mixed_document.pdf",
    pages=None,
    llm_provider="openrouter"
)
```

**Smart Routing Logic:**
1. 🔍 **Content analysis** - Fast extraction on all pages
2. 📊 **Quality assessment** - Evaluates extraction success
3. 🎯 **Page classification**:
   - **Simple pages** (>100 chars) → Keep fast extraction
   - **Complex pages** (<100 chars) → Route to LLM OCR
4. 💰 **Cost optimization** - Only pay for LLM on difficult pages

### Text Search

#### `search_pdf_content(input_file, query, case_sensitive?, whole_words?)`
Search for text content within PDF documents.
```python
result = search_pdf_content(
    input_file="document.pdf",
    query="important keyword",
    case_sensitive=False,
    whole_words=True
)
```

### LLM-Ready Analysis

#### `analyze_pdf_content(input_file, max_chars?)`
Prepare PDF content for LLM consumption with smart chunking.
```python
result = analyze_pdf_content(
    input_file="research_paper.pdf",
    max_chars=100000
)
```

**Features:**
- 📝 **Smart chunking** by sentences
- 📋 **Page summaries** for quick overview
- 🏷️ **Metadata extraction** with statistics
- 🤖 **LLM-optimized** output format

## 🔧 Advanced Features

### Caching System

**Multi-Tier Caching:**
- 📄 **Text extraction**: 24 hours TTL
- 📷 **Traditional OCR**: 7 days TTL
- 🤖 **LLM OCR**: 30 days TTL (expensive to regenerate)

**Cache Location:** `~/.pdf_mcp_cache/`

**Cache Benefits:**
- ⚡ **Instant responses** for repeated requests
- 💰 **Cost savings** on LLM OCR
- 🔄 **Automatic cleanup** with LRU eviction

### Document Handle Management

**Problem Solved:** "Document closed" errors in async/parallel processing

**Solution:**
1. 📸 **Pre-extract images** while document is open
2. 🔒 **Close document** immediately after extraction
3. 🔄 **Process images** independently (no document access needed)

### Progress Tracking

**Detailed Metrics:**
```json
{
  "processing_time": 4.2,
  "total_pages": 10,
  "processed_pages": 10,
  "tokens_used": 1250,
  "confidence": 0.94,
  "method": "llm_ocr_openrouter",
  "from_cache": false
}
```

### Error Recovery

**Graceful Fallbacks:**
- 🤖 LLM OCR → 📷 Traditional OCR → ⚡ Fast extraction
- 🌐 OpenRouter → 🔮 Mistral → 📷 Local OCR
- 🔄 Retry logic with exponential backoff

## ⚙️ Configuration

### Workspace Configuration

#### `configure_pdf_workspace(directory_path)`
Set custom PDF workspace directory.
```python
result = configure_pdf_workspace("/custom/pdf/workspace")
```

### Status & Debugging

#### `get_llm_ocr_status()`
Check LLM OCR providers and configuration.
```python
result = get_llm_ocr_status()
```

#### `debug_environment()`
Debug environment variables and setup.
```python
result = debug_environment()
```

#### `get_server_status()`
Check server and tools availability.
```python
result = get_server_status()
```

### File Management

#### `download_pdf(url, output_file)`
Download PDF from URL.
```python
result = download_pdf(
    url="https://example.com/document.pdf",
    output_file="downloaded.pdf"
)
```

#### `count_pdfs_in_directory(directory_path?)`
Count PDF files in directory.
```python
result = count_pdfs_in_directory("/pdf/folder")
```

## 📚 API Reference

### Response Format

All tools return standardized response format:
```json
{
  "success": true,
  "method": "extraction_method",
  "total_pages": 10,
  "processed_pages": 10,
  "processing_time": 2.45,
  "file_info": {
    "path": "/path/to/file.pdf",
    "size_mb": 1.2,
    "modified": 1640995200
  },
  "pages": {
    "0": {
      "text": "Extracted text...",
      "char_count": 1250,
      "word_count": 200,
      "confidence": 0.95
    }
  },
  "full_text": "Complete extracted text...",
  "from_cache": false
}
```

### Error Handling

**Error Response Format:**
```json
{
  "success": false,
  "error": "Detailed error message",
  "file_info": {...}
}
```

**Common Errors:**
- `Invalid PDF file` - File doesn't exist or corrupted
- `LLM OCR not available` - API key not configured
- `Document closed` - Fixed with handle management
- `Rate limit exceeded` - Handled with backoff

## 🔧 Troubleshooting

### Common Issues

#### 1. "Document closed" errors
**Fixed in v2.0** with proper document handle management.

#### 2. LLM OCR 401 Unauthorized
```bash
# Check API key configuration
export OPENROUTER_API_KEY="your-key-here"
export OPENROUTER_OCR_ENABLED="true"

# Verify in MCP config
"env": {
  "OPENROUTER_API_KEY": "${OPENROUTER_API_KEY}",
  "OPENROUTER_OCR_ENABLED": "true"
}
```

#### 3. Missing dependencies
```bash
# Install system tools
sudo apt-get install pdftk qpdf tesseract-ocr

# Install Python packages
pip install -r requirements.txt
```

#### 4. Slow processing
- ✅ **Use caching** - Enable for repeated documents
- ✅ **Optimize pages** - Process specific pages only
- ✅ **Choose method** - Fast extraction for text PDFs
- ✅ **Hybrid mode** - Automatic cost optimization

### Performance Tips

#### Memory Optimization
- 📊 **Process in batches** - Large documents split processing
- 🔄 **Use page ranges** - Extract specific pages only
- 💾 **Enable caching** - Avoid reprocessing
- 🧹 **Regular cleanup** - Clear cache periodically

#### Cost Optimization
- 🎯 **Use hybrid mode** - Automatic smart routing
- 📄 **Try fast first** - Most PDFs have extractable text
- 🔄 **Cache LLM results** - 30-day TTL saves money
- 📊 **Monitor usage** - Track token consumption

## 📈 Performance Optimization

### Processing Speed by Method

| Method | Speed | Use Case |
|--------|-------|----------|
| Fast | ~1s/page | Text-based PDFs |
| Layout-aware | ~2s/page | Tables/forms |
| Traditional OCR | ~3s/page | Scanned documents |
| LLM OCR | ~4s/page | Complex/handwritten |
| Smart hybrid | ~1-4s/page | Mixed documents |

### Memory Usage

| Method | Memory/Page | Notes |
|--------|-------------|-------|
| Fast | ~1MB | Minimal overhead |
| Layout-aware | ~2MB | Table processing |
| Traditional OCR | ~5MB | Image conversion |
| LLM OCR | ~8MB | High-res images |

### Cost Analysis (LLM OCR)

| Provider | Cost | Quality | Speed |
|----------|------|---------|-------|
| OpenRouter (Gemini) | $0.075/1M tokens | 95%+ | 3-4s |
| Mistral | $1/1000 pages | 90%+ | 4-5s |
| Traditional OCR | Free | 80%+ | 3s |

## 🚀 Recent Updates

### v2.1 - OpenRouter Integration
- ✅ **Added OpenRouter support** with Gemini 2.5 Flash
- ✅ **Multi-provider LLM** system with fallbacks
- ✅ **Improved reliability** over Mistral-only approach
- ✅ **Better cost optimization**

### v2.0 - Document Handle Fix
- 🔧 **Fixed "document closed"** errors in async processing
- 🔧 **Improved memory management** with pre-extraction
- 🔧 **Enhanced error handling** with proper cleanup
- 🔧 **Added validation helpers** for document state

### v1.5 - LLM OCR Addition
- 🤖 **Added Mistral OCR** support
- 🤖 **Smart hybrid processing** for cost optimization
- 🤖 **Advanced caching** with longer TTL for LLM results
- 🤖 **Async batch processing** with rate limiting

### v1.0 - Initial Release
- 📄 **Core PDF operations** with PDFtk/QPDF
- 📝 **Multi-method text extraction**
- 🔍 **Traditional OCR** with Tesseract
- ⚙️ **MCP framework** integration

---

## 📞 Support

For issues, questions, or contributions:
- 📧 **Issues**: Report on GitHub repository
- 📖 **Documentation**: This comprehensive guide
- 🔧 **Debug tools**: Use built-in `debug_environment()` and `get_llm_ocr_status()`

**Happy PDF processing! 🎉**