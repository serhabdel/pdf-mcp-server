import asyncio
import base64
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import logging

try:
    import httpx
except ImportError:
    httpx = None

try:
    from mistralai import Mistral
    from mistralai.models import UserMessage, AssistantMessage, SystemMessage
except ImportError:
    Mistral = None
    UserMessage = None
    AssistantMessage = None
    SystemMessage = None

@dataclass
class LLMOCRResult:
    """Result from LLM OCR extraction"""
    success: bool
    text: str = ""
    confidence: float = 0.0
    provider: str = ""
    model: str = ""
    tokens_used: int = 0
    processing_time: float = 0.0
    error: Optional[str] = None
    structured_data: Optional[Dict] = None
    reasoning: Optional[str] = None

class LLMOCRProvider(ABC):
    """Abstract base class for LLM OCR providers"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    async def extract_text(self, image_data: bytes, prompt: str = None) -> LLMOCRResult:
        """Extract text from image using LLM"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available and configured"""
        pass

class MistralOCRProvider(LLMOCRProvider):
    """Mistral Vision API OCR provider"""
    
    def __init__(self, api_key: str = None, model: str = "mistral-ocr-latest"):
        api_key = api_key or os.getenv("MISTRAL_API_KEY")
        super().__init__(api_key)
        self.model = model
        self.client = None
        self.rate_limiter = asyncio.Semaphore(5)  # Max 5 concurrent requests
        
        if self.api_key and Mistral:
            try:
                self.client = Mistral(api_key=self.api_key)
            except Exception as e:
                self.logger.error(f"Failed to initialize Mistral client: {e}")
    
    def is_available(self) -> bool:
        """Check if Mistral OCR is available"""
        return (
            self.api_key is not None and 
            Mistral is not None and 
            self.client is not None and
            os.getenv("MISTRAL_OCR_ENABLED", "false").lower() == "true"
        )
    
    async def extract_text(self, image_data: bytes, prompt: str = None) -> LLMOCRResult:
        """Extract text using Mistral Vision API"""
        
        if not self.is_available():
            return LLMOCRResult(
                success=False,
                error="Mistral OCR not available. Check MISTRAL_API_KEY and MISTRAL_OCR_ENABLED env vars",
                provider="mistral"
            )
        
        # Default OCR prompt
        if not prompt:
            prompt = """Extract all text from this image with high accuracy. 
                       Preserve formatting, line breaks, and structure exactly as shown.
                       If there are tables, preserve the tabular structure.
                       Return only the extracted text content without any additional commentary."""
        
        start_time = time.time()
        
        try:
            # Convert image to base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Prepare the message with image
            messages = [
                UserMessage(
                    content=[
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            }
                        }
                    ]
                )
            ]
            
            # Make API call with rate limiting
            async with self.rate_limiter:
                response = await self._make_request_with_backoff(messages)
            
            processing_time = time.time() - start_time
            
            if response and hasattr(response, 'choices') and response.choices:
                extracted_text = response.choices[0].message.content.strip()
                tokens_used = getattr(response, 'usage', {}).get('total_tokens', 0)
                
                return LLMOCRResult(
                    success=True,
                    text=extracted_text,
                    confidence=0.95,  # High confidence for LLM OCR
                    provider="mistral",
                    model=self.model,
                    tokens_used=tokens_used,
                    processing_time=processing_time
                )
            else:
                return LLMOCRResult(
                    success=False,
                    error="Empty response from Mistral API",
                    provider="mistral",
                    processing_time=processing_time
                )
                
        except Exception as e:
            return LLMOCRResult(
                success=False,
                error=f"Mistral OCR failed: {str(e)}",
                provider="mistral",
                processing_time=time.time() - start_time
            )
    
    async def _make_request_with_backoff(self, messages: List[UserMessage], max_retries: int = 3):
        """Make API request with exponential backoff"""
        
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.complete_async(
                    model=self.model,
                    messages=messages,
                    max_tokens=4000,
                    temperature=0.1  # Low temperature for consistent OCR
                )
                return response
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                
                # Exponential backoff
                wait_time = (2 ** attempt) + (0.1 * attempt)
                self.logger.warning(f"Mistral API request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
        
        return None

class OpenRouterOCRProvider(LLMOCRProvider):
    """OpenRouter API OCR provider with support for multiple vision models"""
    
    def __init__(self, api_key: str = None, model: str = "google/gemini-2.5-flash-lite-preview-06-17"):
        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        super().__init__(api_key)
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
        self.client = httpx.AsyncClient() if httpx else None
        self.rate_limiter = asyncio.Semaphore(10)  # OpenRouter can handle more concurrent requests
        
    def is_available(self) -> bool:
        """Check if OpenRouter OCR is available"""
        return (
            self.api_key is not None and 
            httpx is not None and 
            self.client is not None and
            os.getenv("OPENROUTER_OCR_ENABLED", "false").lower() == "true"
        )
    
    async def extract_text(self, image_data: bytes, prompt: str = None) -> LLMOCRResult:
        """Extract text using OpenRouter API with vision models"""
        
        if not self.is_available():
            return LLMOCRResult(
                success=False,
                error="OpenRouter OCR not available. Check OPENROUTER_API_KEY and OPENROUTER_OCR_ENABLED env vars",
                provider="openrouter"
            )
        
        # Default OCR prompt optimized for Gemini
        if not prompt:
            prompt = """Please extract all text from this image with high accuracy. 
                       Preserve formatting, line breaks, and structure exactly as shown.
                       If there are tables, preserve the tabular structure with clear column separation.
                       Return only the extracted text content without any additional commentary or markdown formatting."""
        
        start_time = time.time()
        
        try:
            # Convert image to base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Prepare OpenRouter API request (OpenAI-compatible format)
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 4000,
                "temperature": 0.1  # Low temperature for consistent OCR
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/your-repo",  # OpenRouter requires referrer
                "X-Title": "PDF MCP OCR Server"
            }
            
            # Make API call with rate limiting
            async with self.rate_limiter:
                response = await self._make_request_with_backoff(payload, headers)
            
            processing_time = time.time() - start_time
            
            if response and "choices" in response and response["choices"]:
                extracted_text = response["choices"][0]["message"]["content"].strip()
                tokens_used = response.get("usage", {}).get("total_tokens", 0)
                
                return LLMOCRResult(
                    success=True,
                    text=extracted_text,
                    confidence=0.92,  # Gemini generally has high confidence
                    provider="openrouter",
                    model=self.model,
                    tokens_used=tokens_used,
                    processing_time=processing_time
                )
            else:
                return LLMOCRResult(
                    success=False,
                    error="Empty response from OpenRouter API",
                    provider="openrouter",
                    processing_time=processing_time
                )
                
        except Exception as e:
            return LLMOCRResult(
                success=False,
                error=f"OpenRouter OCR failed: {str(e)}",
                provider="openrouter",
                processing_time=time.time() - start_time
            )
    
    async def _make_request_with_backoff(self, payload: dict, headers: dict, max_retries: int = 3):
        """Make API request with exponential backoff"""
        
        for attempt in range(max_retries):
            try:
                response = await self.client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=60.0  # OpenRouter can be slower
                )
                
                response.raise_for_status()
                return response.json()
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                
                # Exponential backoff
                wait_time = (2 ** attempt) + (0.1 * attempt)
                self.logger.warning(f"OpenRouter API request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
        
        return None

class HTTPOCRProvider(LLMOCRProvider):
    """Generic HTTP-based OCR provider (fallback for custom APIs)"""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        super().__init__(api_key)
        self.base_url = base_url or os.getenv("CUSTOM_OCR_API_URL")
        self.client = httpx.AsyncClient() if httpx else None
    
    def is_available(self) -> bool:
        return self.api_key is not None and self.base_url is not None and httpx is not None
    
    async def extract_text(self, image_data: bytes, prompt: str = None) -> LLMOCRResult:
        """Extract text using generic HTTP API"""
        
        if not self.is_available():
            return LLMOCRResult(
                success=False,
                error="Custom OCR API not configured",
                provider="http"
            )
        
        start_time = time.time()
        
        try:
            # Convert to base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            payload = {
                "image": image_b64,
                "prompt": prompt or "Extract all text from this image"
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = await self.client.post(
                f"{self.base_url}/ocr",
                json=payload,
                headers=headers,
                timeout=30.0
            )
            
            response.raise_for_status()
            result = response.json()
            
            return LLMOCRResult(
                success=True,
                text=result.get("text", ""),
                confidence=result.get("confidence", 0.8),
                provider="http",
                processing_time=time.time() - start_time,
                tokens_used=result.get("tokens_used", 0)
            )
            
        except Exception as e:
            return LLMOCRResult(
                success=False,
                error=f"HTTP OCR failed: {str(e)}",
                provider="http",
                processing_time=time.time() - start_time
            )

class LLMOCRManager:
    """Manager for multiple LLM OCR providers"""
    
    def __init__(self):
        self.providers = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize available providers
        self._init_providers()
    
    def _init_providers(self):
        """Initialize available OCR providers"""
        
        # OpenRouter OCR (prioritized - more reliable)
        openrouter_provider = OpenRouterOCRProvider()
        if openrouter_provider.is_available():
            self.providers["openrouter"] = openrouter_provider
            self.logger.info("OpenRouter OCR provider initialized")
        
        # Mistral OCR
        mistral_provider = MistralOCRProvider()
        if mistral_provider.is_available():
            self.providers["mistral"] = mistral_provider
            self.logger.info("Mistral OCR provider initialized")
        
        # Custom HTTP provider
        http_provider = HTTPOCRProvider()
        if http_provider.is_available():
            self.providers["http"] = http_provider
            self.logger.info("HTTP OCR provider initialized")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names"""
        return list(self.providers.keys())
    
    def is_provider_available(self, provider_name: str) -> bool:
        """Check if specific provider is available"""
        return provider_name in self.providers
    
    async def extract_text(self, 
                          image_data: bytes, 
                          provider: str = "mistral", 
                          prompt: str = None) -> LLMOCRResult:
        """Extract text using specified provider"""
        
        if provider not in self.providers:
            return LLMOCRResult(
                success=False,
                error=f"Provider '{provider}' not available. Available: {self.get_available_providers()}",
                provider=provider
            )
        
        return await self.providers[provider].extract_text(image_data, prompt)
    
    async def extract_text_with_fallback(self, 
                                       image_data: bytes, 
                                       providers: List[str] = None, 
                                       prompt: str = None) -> LLMOCRResult:
        """Extract text with provider fallback"""
        
        if not providers:
            providers = ["openrouter", "mistral", "http"]
        
        last_error = None
        
        for provider in providers:
            if provider in self.providers:
                result = await self.extract_text(image_data, provider, prompt)
                if result.success:
                    return result
                last_error = result.error
        
        return LLMOCRResult(
            success=False,
            error=f"All providers failed. Last error: {last_error}",
            provider="fallback"
        )