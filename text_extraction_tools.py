import hashlib
import json
import time
import io
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import PDFUtils
from llm_ocr_providers import LLMOCRManager, LLMOCRResult

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None

class TextExtractionTools:
    """Advanced text extraction and OCR tools for PDF documents."""
    
    def __init__(self):
        self.utils = PDFUtils()
        self.cache_dir = Path.home() / '.pdf_mcp_cache'
        self.cache_dir.mkdir(exist_ok=True)
        self.max_cache_size = 1024 * 1024 * 1024  # 1GB
        self.cache_ttl = 24 * 3600  # 24 hours for text, 7 days for OCR
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM OCR manager
        self.llm_ocr_manager = LLMOCRManager()
    
    def _get_document_hash(self, file_path: Path) -> str:
        """Generate MD5 hash for document caching."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _get_cache_path(self, doc_hash: str, method: str, page_num: Optional[int] = None) -> Path:
        """Get cache file path for extracted content."""
        if page_num is not None:
            return self.cache_dir / f"{doc_hash}_{method}_page_{page_num}.json"
        return self.cache_dir / f"{doc_hash}_{method}_full.json"
    
    def _is_cache_valid(self, cache_path: Path, ttl: int = None) -> bool:
        """Check if cached result is still valid."""
        if not cache_path.exists():
            return False
        
        if ttl is None:
            ttl = self.cache_ttl
            
        cache_age = time.time() - cache_path.stat().st_mtime
        return cache_age < ttl
    
    def _save_cache(self, cache_path: Path, data: Dict[str, Any]) -> None:
        """Save extraction result to cache."""
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")
    
    def _load_cache(self, cache_path: Path) -> Optional[Dict[str, Any]]:
        """Load extraction result from cache."""
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load cache: {e}")
            return None
    
    def _validate_document_open(self, doc) -> bool:
        """Validate that a PyMuPDF document is still open and accessible."""
        try:
            if doc is None:
                return False
            # Try to access basic document properties
            _ = len(doc)
            return True
        except:
            return False
    
    def _detect_text_content(self, file_path: Path) -> Dict[str, Any]:
        """Detect if PDF contains extractable text or requires OCR."""
        if not fitz:
            return {"success": False, "error": "PyMuPDF not available"}
        
        doc = None
        try:
            doc = fitz.open(str(file_path))
            
            if not self._validate_document_open(doc):
                return {"success": False, "error": "Failed to open PDF document"}
            
            total_chars = 0
            total_pages = len(doc)
            sample_pages = min(5, total_pages)  # Check first 5 pages
            
            for page_num in range(sample_pages):
                if not self._validate_document_open(doc):
                    return {"success": False, "error": "Document became inaccessible during processing"}
                
                page = doc.load_page(page_num)
                text = page.get_text()
                total_chars += len(text.strip())
            
            # Heuristic: if average characters per page < 50, likely needs OCR
            avg_chars_per_page = total_chars / sample_pages if sample_pages > 0 else 0
            needs_ocr = avg_chars_per_page < 50
            
            return {
                "success": True,
                "total_pages": total_pages,
                "avg_chars_per_page": avg_chars_per_page,
                "needs_ocr": needs_ocr,
                "has_text": avg_chars_per_page > 10
            }
            
        except Exception as e:
            return {"success": False, "error": f"Document analysis failed: {str(e)}"}
        finally:
            # Ensure document is always closed
            if doc is not None:
                try:
                    doc.close()
                except:
                    pass
    
    def extract_text_fast(self, file_path: str, pages: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Fast text extraction using PyMuPDF.
        
        Args:
            file_path: PDF file path
            pages: Specific pages to extract (None for all)
            
        Returns:
            dict: Extracted text content
        """
        if not fitz:
            return {"success": False, "error": "PyMuPDF not installed. Run: pip install pymupdf"}
        
        input_path = self.utils.resolve_path(file_path)
        if not self.utils.validate_pdf_file(input_path):
            return {"success": False, "error": f"Invalid PDF file: {file_path}"}
        
        # Check cache
        doc_hash = self._get_document_hash(input_path)
        cache_key = f"pages_{'-'.join(map(str, pages))}" if pages else "all"
        cache_path = self.cache_dir / f"{doc_hash}_fast_{cache_key}.json"
        
        if self._is_cache_valid(cache_path):
            cached_result = self._load_cache(cache_path)
            if cached_result:
                cached_result["from_cache"] = True
                return cached_result
        
        try:
            start_time = time.time()
            doc = fitz.open(str(input_path))
            
            extracted_pages = {}
            total_text = ""
            
            pages_to_process = pages if pages else range(len(doc))
            
            for page_num in pages_to_process:
                if page_num >= len(doc):
                    continue
                    
                page = doc.load_page(page_num)
                text = page.get_text()
                
                extracted_pages[page_num] = {
                    "text": text,
                    "char_count": len(text),
                    "word_count": len(text.split())
                }
                total_text += text + "\n"
            
            doc.close()
            
            result = {
                "success": True,
                "method": "pymupdf_fast",
                "total_pages": len(doc),
                "processed_pages": len(extracted_pages),
                "pages": extracted_pages,
                "full_text": total_text.strip(),
                "total_chars": len(total_text),
                "total_words": len(total_text.split()),
                "processing_time": time.time() - start_time,
                "file_info": self.utils.get_file_info(input_path),
                "from_cache": False
            }
            
            # Save to cache
            self._save_cache(cache_path, result)
            return result
            
        except Exception as e:
            return {"success": False, "error": f"Text extraction failed: {str(e)}"}
    
    def extract_text_layout_aware(self, file_path: str, pages: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Layout-aware text extraction using pdfplumber.
        
        Args:
            file_path: PDF file path
            pages: Specific pages to extract (None for all)
            
        Returns:
            dict: Extracted text with layout information
        """
        if not pdfplumber:
            return {"success": False, "error": "pdfplumber not installed. Run: pip install pdfplumber"}
        
        input_path = self.utils.resolve_path(file_path)
        if not self.utils.validate_pdf_file(input_path):
            return {"success": False, "error": f"Invalid PDF file: {file_path}"}
        
        # Check cache
        doc_hash = self._get_document_hash(input_path)
        cache_key = f"pages_{'-'.join(map(str, pages))}" if pages else "all"
        cache_path = self.cache_dir / f"{doc_hash}_layout_{cache_key}.json"
        
        if self._is_cache_valid(cache_path):
            cached_result = self._load_cache(cache_path)
            if cached_result:
                cached_result["from_cache"] = True
                return cached_result
        
        try:
            start_time = time.time()
            
            with pdfplumber.open(str(input_path)) as pdf:
                extracted_pages = {}
                total_text = ""
                
                pages_to_process = pages if pages else range(len(pdf.pages))
                
                for page_num in pages_to_process:
                    if page_num >= len(pdf.pages):
                        continue
                    
                    page = pdf.pages[page_num]
                    
                    # Extract text
                    text = page.extract_text() or ""
                    
                    # Extract tables if present
                    tables = page.extract_tables()
                    table_text = ""
                    if tables:
                        for table in tables:
                            for row in table:
                                if row:
                                    table_text += " | ".join(str(cell) if cell else "" for cell in row) + "\n"
                    
                    combined_text = text + "\n" + table_text if table_text else text
                    
                    extracted_pages[page_num] = {
                        "text": text,
                        "tables": len(tables),
                        "table_text": table_text,
                        "combined_text": combined_text,
                        "char_count": len(combined_text),
                        "word_count": len(combined_text.split()),
                        "bbox": page.bbox if hasattr(page, 'bbox') else None
                    }
                    
                    total_text += combined_text + "\n"
            
            result = {
                "success": True,
                "method": "pdfplumber_layout",
                "total_pages": len(pdf.pages),
                "processed_pages": len(extracted_pages),
                "pages": extracted_pages,
                "full_text": total_text.strip(),
                "total_chars": len(total_text),
                "total_words": len(total_text.split()),
                "processing_time": time.time() - start_time,
                "file_info": self.utils.get_file_info(input_path),
                "from_cache": False
            }
            
            # Save to cache
            self._save_cache(cache_path, result)
            return result
            
        except Exception as e:
            return {"success": False, "error": f"Layout-aware extraction failed: {str(e)}"}
    
    def extract_text_ocr(self, file_path: str, pages: Optional[List[int]] = None, 
                        language: str = "eng", dpi: int = 300) -> Dict[str, Any]:
        """
        OCR-based text extraction using Tesseract.
        
        Args:
            file_path: PDF file path
            pages: Specific pages to extract (None for all)
            language: OCR language (default: eng)
            dpi: Image DPI for OCR (default: 300)
            
        Returns:
            dict: OCR extracted text content
        """
        if not pytesseract or not Image or not fitz:
            return {
                "success": False, 
                "error": "OCR dependencies not installed. Run: pip install pytesseract pillow pymupdf"
            }
        
        input_path = self.utils.resolve_path(file_path)
        if not self.utils.validate_pdf_file(input_path):
            return {"success": False, "error": f"Invalid PDF file: {file_path}"}
        
        # Check cache (longer TTL for OCR)
        doc_hash = self._get_document_hash(input_path)
        cache_key = f"pages_{'-'.join(map(str, pages))}" if pages else "all"
        cache_path = self.cache_dir / f"{doc_hash}_ocr_{language}_{dpi}_{cache_key}.json"
        
        if self._is_cache_valid(cache_path, ttl=7*24*3600):  # 7 days for OCR
            cached_result = self._load_cache(cache_path)
            if cached_result:
                cached_result["from_cache"] = True
                return cached_result
        
        try:
            start_time = time.time()
            doc = fitz.open(str(input_path))
            
            pages_to_process = pages if pages else range(len(doc))
            
            # STEP 1: Pre-extract all page images while document is open
            self.logger.info(f"Pre-extracting images for OCR processing of {len(pages_to_process)} pages...")
            page_images = {}
            
            for page_num in pages_to_process:
                if page_num >= len(doc):
                    continue
                    
                page = doc.load_page(page_num)
                # Convert page to image
                mat = fitz.Matrix(dpi/72, dpi/72)  # Scale to desired DPI
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                page_images[page_num] = img_data
            
            # STEP 2: Close document immediately after image extraction
            total_pages = len(doc)
            doc.close()
            self.logger.info(f"Document closed. Processing {len(page_images)} page images with OCR...")
            
            # STEP 3: Process images in parallel (document no longer needed)
            extracted_pages = {}
            total_text = ""
            
            def process_page_image(page_num, img_data):
                """Process a single page image with OCR"""
                try:
                    # Convert to PIL Image
                    img = Image.open(io.BytesIO(img_data))
                    
                    # Perform OCR
                    ocr_text = pytesseract.image_to_string(img, lang=language)
                    
                    # Get OCR data with confidence
                    ocr_data = pytesseract.image_to_data(img, lang=language, output_type=pytesseract.Output.DICT)
                    
                    # Calculate average confidence
                    confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
                    return {
                        "page_num": page_num,
                        "text": ocr_text,
                        "char_count": len(ocr_text),
                        "word_count": len(ocr_text.split()),
                        "confidence": avg_confidence,
                        "words_detected": len([w for w in ocr_data['text'] if w.strip()])
                    }
                except Exception as e:
                    self.logger.error(f"OCR processing failed for page {page_num}: {e}")
                    return {
                        "page_num": page_num,
                        "text": "",
                        "char_count": 0,
                        "word_count": 0,
                        "confidence": 0.0,
                        "words_detected": 0,
                        "error": str(e)
                    }
            
            # Use ThreadPoolExecutor for parallel processing of images
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(process_page_image, page_num, img_data): page_num 
                          for page_num, img_data in page_images.items()}
                
                for future in as_completed(futures):
                    result_data = future.result()
                    if result_data:
                        page_num = result_data["page_num"]
                        extracted_pages[page_num] = result_data
                        total_text += result_data["text"] + "\n"
            
            # Calculate overall statistics
            total_confidence = sum(page["confidence"] for page in extracted_pages.values())
            avg_confidence = total_confidence / len(extracted_pages) if extracted_pages else 0
            
            result = {
                "success": True,
                "method": "tesseract_ocr",
                "language": language,
                "dpi": dpi,
                "total_pages": total_pages,
                "processed_pages": len(extracted_pages),
                "pages": extracted_pages,
                "full_text": total_text.strip(),
                "total_chars": len(total_text),
                "total_words": len(total_text.split()),
                "average_confidence": round(avg_confidence, 2),
                "processing_time": time.time() - start_time,
                "file_info": self.utils.get_file_info(input_path),
                "from_cache": False
            }
            
            # Save to cache
            self._save_cache(cache_path, result)
            return result
            
        except Exception as e:
            return {"success": False, "error": f"OCR extraction failed: {str(e)}"}
    
    def extract_text_smart(self, file_path: str, pages: Optional[List[int]] = None,
                          ocr_fallback: bool = True) -> Dict[str, Any]:
        """
        Smart text extraction with automatic method selection.
        
        Args:
            file_path: PDF file path
            pages: Specific pages to extract (None for all)
            ocr_fallback: Use OCR if text extraction yields poor results
            
        Returns:
            dict: Extracted text using best available method
        """
        input_path = self.utils.resolve_path(file_path)
        if not self.utils.validate_pdf_file(input_path):
            return {"success": False, "error": f"Invalid PDF file: {file_path}"}
        
        # Detect content type
        detection = self._detect_text_content(input_path)
        if not detection["success"]:
            return detection
        
        start_time = time.time()
        methods_tried = []
        
        # Try fast extraction first
        if detection["has_text"]:
            result = self.extract_text_fast(file_path, pages)
            methods_tried.append("pymupdf_fast")
            
            if result["success"] and result["total_chars"] > 100:
                result["detection_info"] = detection
                result["methods_tried"] = methods_tried
                result["smart_processing_time"] = time.time() - start_time
                return result
        
        # Try layout-aware extraction
        if pdfplumber:
            result = self.extract_text_layout_aware(file_path, pages)
            methods_tried.append("pdfplumber_layout")
            
            if result["success"] and result["total_chars"] > 100:
                result["detection_info"] = detection
                result["methods_tried"] = methods_tried
                result["smart_processing_time"] = time.time() - start_time
                return result
        
        # Fall back to OCR if enabled and needed
        if ocr_fallback and (detection["needs_ocr"] or not detection["has_text"]):
            result = self.extract_text_ocr(file_path, pages)
            methods_tried.append("tesseract_ocr")
            
            if result["success"]:
                result["detection_info"] = detection
                result["methods_tried"] = methods_tried
                result["smart_processing_time"] = time.time() - start_time
                return result
        
        return {
            "success": False,
            "error": "All extraction methods failed or insufficient text found",
            "detection_info": detection,
            "methods_tried": methods_tried,
            "smart_processing_time": time.time() - start_time
        }
    
    def search_text_content(self, file_path: str, query: str, case_sensitive: bool = False,
                           whole_words: bool = False) -> Dict[str, Any]:
        """
        Search for text content within PDF.
        
        Args:
            file_path: PDF file path
            query: Search query
            case_sensitive: Case sensitive search
            whole_words: Match whole words only
            
        Returns:
            dict: Search results with page locations
        """
        # First extract all text
        extraction_result = self.extract_text_smart(file_path)
        
        if not extraction_result["success"]:
            return extraction_result
        
        import re
        
        search_flags = 0 if case_sensitive else re.IGNORECASE
        pattern = rf'\b{re.escape(query)}\b' if whole_words else re.escape(query)
        
        matches = []
        total_matches = 0
        
        for page_num, page_data in extraction_result["pages"].items():
            page_text = page_data.get("text", "") or page_data.get("combined_text", "")
            
            page_matches = []
            for match in re.finditer(pattern, page_text, search_flags):
                start, end = match.span()
                context_start = max(0, start - 50)
                context_end = min(len(page_text), end + 50)
                context = page_text[context_start:context_end]
                
                page_matches.append({
                    "position": start,
                    "context": context,
                    "match_text": match.group()
                })
            
            if page_matches:
                matches.append({
                    "page": page_num,
                    "matches": len(page_matches),
                    "results": page_matches
                })
                total_matches += len(page_matches)
        
        return {
            "success": True,
            "query": query,
            "total_matches": total_matches,
            "pages_with_matches": len(matches),
            "case_sensitive": case_sensitive,
            "whole_words": whole_words,
            "matches": matches,
            "extraction_method": extraction_result.get("method", "unknown"),
            "file_info": extraction_result["file_info"]
        }
    
    def analyze_content_for_llm(self, file_path: str, max_chars: int = 100000) -> Dict[str, Any]:
        """
        Analyze PDF content for LLM consumption with smart chunking.
        
        Args:
            file_path: PDF file path
            max_chars: Maximum characters to extract
            
        Returns:
            dict: Structured content analysis for LLM
        """
        extraction_result = self.extract_text_smart(file_path)
        
        if not extraction_result["success"]:
            return extraction_result
        
        full_text = extraction_result["full_text"]
        
        # Smart text chunking if too long
        chunks = []
        if len(full_text) > max_chars:
            # Split into sentences and group into chunks
            sentences = re.split(r'[.!?]+', full_text)
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < max_chars // 3:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
            
            if current_chunk:
                chunks.append(current_chunk.strip())
        else:
            chunks = [full_text]
        
        # Extract key metadata
        pages = extraction_result["pages"]
        page_summaries = []
        
        for page_num, page_data in pages.items():
            page_text = page_data.get("text", "") or page_data.get("combined_text", "")
            word_count = page_data.get("word_count", 0)
            
            # Simple page summary (first 200 chars)
            summary = page_text[:200] + "..." if len(page_text) > 200 else page_text
            
            page_summaries.append({
                "page": page_num,
                "word_count": word_count,
                "summary": summary.strip(),
                "has_tables": page_data.get("tables", 0) > 0
            })
        
        return {
            "success": True,
            "document_analysis": {
                "total_pages": extraction_result["total_pages"],
                "total_words": extraction_result["total_words"],
                "total_chars": extraction_result["total_chars"],
                "extraction_method": extraction_result["method"],
                "processing_time": extraction_result["processing_time"],
                "truncated": len(full_text) > max_chars
            },
            "content_chunks": chunks,
            "page_summaries": page_summaries,
            "file_info": extraction_result["file_info"],
            "llm_ready": True
        }
    
    async def extract_text_llm_ocr(self, file_path: str, pages: Optional[List[int]] = None, 
                                  provider: str = "mistral", model: str = "mistral-ocr-latest",
                                  custom_prompt: str = None) -> Dict[str, Any]:
        """
        LLM-based OCR text extraction using Mistral or other providers.
        
        Args:
            file_path: PDF file path
            pages: Specific pages to extract (None for all)
            provider: LLM provider ("mistral", "http")
            model: Model to use for OCR
            custom_prompt: Custom OCR prompt
            
        Returns:
            dict: LLM OCR extracted text content
        """
        if not fitz:
            return {"success": False, "error": "PyMuPDF not installed. Run: pip install pymupdf"}
        
        input_path = self.utils.resolve_path(file_path)
        if not self.utils.validate_pdf_file(input_path):
            return {"success": False, "error": f"Invalid PDF file: {file_path}"}
        
        # Check if LLM OCR is available
        if not self.llm_ocr_manager.is_provider_available(provider):
            available = self.llm_ocr_manager.get_available_providers()
            return {
                "success": False, 
                "error": f"LLM OCR provider '{provider}' not available. Available: {available}.\n"
                        f"Set MISTRAL_API_KEY and MISTRAL_OCR_ENABLED=true environment variables."
            }
        
        # Check cache
        doc_hash = self._get_document_hash(input_path)
        cache_key = f"pages_{'-'.join(map(str, pages))}" if pages else "all"
        cache_path = self.cache_dir / f"{doc_hash}_llm_ocr_{provider}_{model}_{cache_key}.json"
        
        if self._is_cache_valid(cache_path, ttl=30*24*3600):  # 30 days for LLM OCR
            cached_result = self._load_cache(cache_path)
            if cached_result:
                cached_result["from_cache"] = True
                return cached_result
        
        try:
            start_time = time.time()
            doc = fitz.open(str(input_path))
            
            pages_to_process = pages if pages else range(len(doc))
            
            # STEP 1: Pre-extract all page images while document is open
            self.logger.info(f"Pre-extracting images for {len(pages_to_process)} pages...")
            page_images = {}
            
            for page_num in pages_to_process:
                if page_num >= len(doc):
                    continue
                    
                page = doc.load_page(page_num)
                # Convert page to high-quality image
                mat = fitz.Matrix(2.0, 2.0)  # 2x scale for better OCR
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                page_images[page_num] = img_data
            
            # STEP 2: Close document immediately after image extraction
            total_pages = len(doc)
            doc.close()
            self.logger.info(f"Document closed. Processing {len(page_images)} page images with LLM OCR...")
            
            # STEP 3: Process images asynchronously (document no longer needed)
            extracted_pages = {}
            total_text = ""
            total_tokens = 0
            
            async def process_page_image_async(page_num, img_data):
                """Process a single page image with LLM OCR"""
                # Use LLM OCR
                result = await self.llm_ocr_manager.extract_text(
                    img_data, 
                    provider=provider, 
                    prompt=custom_prompt
                )
                
                if result.success:
                    return {
                        "page_num": page_num,
                        "text": result.text,
                        "char_count": len(result.text),
                        "word_count": len(result.text.split()),
                        "confidence": result.confidence,
                        "tokens_used": result.tokens_used,
                        "processing_time": result.processing_time,
                        "provider": result.provider,
                        "model": result.model
                    }
                else:
                    self.logger.error(f"LLM OCR failed for page {page_num}: {result.error}")
                    return {
                        "page_num": page_num,
                        "text": "",
                        "char_count": 0,
                        "word_count": 0,
                        "confidence": 0.0,
                        "tokens_used": 0,
                        "processing_time": result.processing_time,
                        "error": result.error
                    }
            
            # Process images in batches to avoid overwhelming the API
            batch_size = 3  # Process 3 pages concurrently
            page_items = list(page_images.items())
            
            for i in range(0, len(page_items), batch_size):
                batch = page_items[i:i + batch_size]
                
                # Process batch concurrently
                tasks = [process_page_image_async(page_num, img_data) for page_num, img_data in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect results
                for result_data in batch_results:
                    if result_data and not isinstance(result_data, Exception):
                        page_num = result_data["page_num"]
                        extracted_pages[page_num] = result_data
                        total_text += result_data["text"] + "\n"
                        total_tokens += result_data.get("tokens_used", 0)
            
            # Calculate statistics
            total_confidence = sum(page.get("confidence", 0) for page in extracted_pages.values())
            avg_confidence = total_confidence / len(extracted_pages) if extracted_pages else 0
            
            result = {
                "success": True,
                "method": f"llm_ocr_{provider}",
                "provider": provider,
                "model": model,
                "total_pages": total_pages,
                "processed_pages": len(extracted_pages),
                "pages": extracted_pages,
                "full_text": total_text.strip(),
                "total_chars": len(total_text),
                "total_words": len(total_text.split()),
                "average_confidence": round(avg_confidence, 2),
                "total_tokens_used": total_tokens,
                "processing_time": time.time() - start_time,
                "file_info": self.utils.get_file_info(input_path),
                "from_cache": False
            }
            
            # Save to cache
            self._save_cache(cache_path, result)
            return result
            
        except Exception as e:
            return {"success": False, "error": f"LLM OCR extraction failed: {str(e)}"}
    
    async def extract_text_hybrid_smart(self, file_path: str, pages: Optional[List[int]] = None,
                                       llm_provider: str = "openrouter") -> Dict[str, Any]:
        """
        Smart hybrid extraction: fast methods first, LLM OCR for complex pages.
        
        Args:
            file_path: PDF file path
            pages: Specific pages to extract
            llm_provider: LLM provider for complex pages
            
        Returns:
            dict: Hybrid extraction results
        """
        input_path = self.utils.resolve_path(file_path)
        if not self.utils.validate_pdf_file(input_path):
            return {"success": False, "error": f"Invalid PDF file: {file_path}"}
        
        start_time = time.time()
        
        # First, try fast extraction to assess content quality
        fast_result = self.extract_text_fast(file_path, pages)
        if not fast_result["success"]:
            return fast_result
        
        # Analyze extraction quality per page
        complex_pages = []
        simple_pages = []
        
        for page_num, page_data in fast_result["pages"].items():
            char_count = page_data.get("char_count", 0)
            word_count = page_data.get("word_count", 0)
            
            # Heuristic: if very low text extraction, likely needs LLM OCR
            if char_count < 100 or (word_count > 0 and char_count / word_count < 3):
                complex_pages.append(page_num)
            else:
                simple_pages.append(page_num)
        
        result = {
            "success": True,
            "method": "hybrid_smart",
            "simple_pages": simple_pages,
            "complex_pages": complex_pages,
            "fast_extraction": fast_result,
            "llm_extraction": None,
            "combined_text": fast_result["full_text"],
            "processing_time": time.time() - start_time
        }
        
        # If we have complex pages and LLM OCR is available, use it
        if complex_pages and self.llm_ocr_manager.is_provider_available(llm_provider):
            try:
                llm_result = await self.extract_text_llm_ocr(
                    file_path, 
                    pages=complex_pages, 
                    provider=llm_provider
                )
                
                if llm_result["success"]:
                    result["llm_extraction"] = llm_result
                    
                    # Combine results: use LLM OCR for complex pages, fast for simple
                    combined_pages = {}
                    combined_text = ""
                    
                    # Add simple pages from fast extraction
                    for page_num in simple_pages:
                        if page_num in fast_result["pages"]:
                            combined_pages[page_num] = fast_result["pages"][page_num]
                            combined_pages[page_num]["extraction_method"] = "fast"
                    
                    # Add complex pages from LLM OCR
                    for page_num in complex_pages:
                        if page_num in llm_result["pages"]:
                            combined_pages[page_num] = llm_result["pages"][page_num]
                            combined_pages[page_num]["extraction_method"] = "llm_ocr"
                    
                    # Rebuild full text in page order
                    sorted_pages = sorted(combined_pages.keys())
                    for page_num in sorted_pages:
                        combined_text += combined_pages[page_num]["text"] + "\n"
                    
                    result.update({
                        "combined_pages": combined_pages,
                        "combined_text": combined_text.strip(),
                        "total_chars": len(combined_text),
                        "total_words": len(combined_text.split()),
                        "tokens_used": llm_result.get("total_tokens_used", 0)
                    })
                    
            except Exception as e:
                self.logger.error(f"LLM OCR failed in hybrid mode: {e}")
                result["llm_error"] = str(e)
        
        result["processing_time"] = time.time() - start_time
        result["file_info"] = self.utils.get_file_info(input_path)
        
        return result