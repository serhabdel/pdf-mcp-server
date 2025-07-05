import os
import shutil
import webbrowser
from pathlib import Path
from typing import Optional, Union
import requests
from urllib.parse import urlparse

class PDFUtils:
    """Utility functions for PDF file operations and path resolution."""
    
    def __init__(self):
        self.default_dirs = [
            os.environ.get('PDF_WORKSPACE', Path.home() / 'Documents' / 'PDFs'),
            Path.home() / 'Downloads',
            Path.home() / 'Desktop',
            Path.cwd()
        ]
        
        # Ensure default PDF directory exists
        self._ensure_default_dir()
    
    def _ensure_default_dir(self):
        """Create default PDF directory if it doesn't exist."""
        pdf_dir = Path(self.default_dirs[0])
        pdf_dir.mkdir(parents=True, exist_ok=True)
    
    def resolve_path(self, user_input: str) -> Path:
        """
        Resolve user input to absolute path.
        
        Args:
            user_input: File path (absolute or relative)
            
        Returns:
            Path: Absolute path to file
        """
        path = Path(user_input)
        
        # If absolute path, use as-is
        if path.is_absolute():
            return path
        
        # Find first existing default directory
        for base_dir in self.default_dirs:
            base_path = Path(base_dir)
            if base_path.exists():
                return base_path / path
        
        # Fallback to current directory
        return Path.cwd() / path
    
    def generate_output_path(self, input_path: Path, suffix: str, extension: str = None) -> Path:
        """
        Generate output path with suffix.
        
        Args:
            input_path: Original file path
            suffix: Suffix to add (e.g., '_merged', '_encrypted')
            extension: New extension (optional)
            
        Returns:
            Path: Output file path
        """
        if extension is None:
            extension = input_path.suffix
        
        stem = input_path.stem
        parent = input_path.parent
        
        return parent / f"{stem}{suffix}{extension}"
    
    def validate_pdf_file(self, file_path: Path) -> bool:
        """
        Validate if file exists and is a PDF.
        
        Args:
            file_path: Path to validate
            
        Returns:
            bool: True if valid PDF file
        """
        if not file_path.exists():
            return False
        
        if file_path.suffix.lower() != '.pdf':
            return False
        
        return True
    
    def download_file(self, url: str, output_path: Path) -> bool:
        """
        Download file from URL.
        
        Args:
            url: URL to download from
            output_path: Where to save file
            
        Returns:
            bool: True if successful
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False
    
    def open_file_externally(self, file_path: Path) -> bool:
        """
        Open file with system default application.
        
        Args:
            file_path: File to open
            
        Returns:
            bool: True if successful
        """
        try:
            if os.name == 'nt':  # Windows
                os.startfile(str(file_path))
            elif os.name == 'posix':  # macOS/Linux
                if shutil.which('open'):  # macOS
                    os.system(f'open "{file_path}"')
                else:  # Linux
                    os.system(f'xdg-open "{file_path}"')
            return True
        except Exception as e:
            print(f"Failed to open file: {e}")
            return False
    
    def open_in_browser(self, file_path: Path) -> bool:
        """
        Open PDF in web browser.
        
        Args:
            file_path: PDF file to open
            
        Returns:
            bool: True if successful
        """
        try:
            file_url = f"file:///{file_path.as_posix()}"
            webbrowser.open(file_url)
            return True
        except Exception as e:
            print(f"Failed to open in browser: {e}")
            return False
    
    def get_file_info(self, file_path: Path) -> dict:
        """
        Get file information.
        
        Args:
            file_path: File to analyze
            
        Returns:
            dict: File information
        """
        if not file_path.exists():
            return {"error": "File not found"}
        
        stat = file_path.stat()
        return {
            "path": str(file_path),
            "size": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified": stat.st_mtime,
            "exists": True
        }
    
    def cleanup_temp_files(self, file_paths: list) -> None:
        """
        Clean up temporary files.
        
        Args:
            file_paths: List of file paths to delete
        """
        for path in file_paths:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception as e:
                print(f"Failed to delete {path}: {e}")