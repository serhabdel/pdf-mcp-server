import subprocess
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from utils import PDFUtils

class PDFtkTools:
    """PDFtk CLI wrapper for PDF operations."""
    
    def __init__(self):
        self.utils = PDFUtils()
        self.pdftk_cmd = "pdftk"
    
    def _run_command(self, args: List[str]) -> Dict[str, Any]:
        """
        Run PDFtk command and return result.
        
        Args:
            args: Command arguments
            
        Returns:
            dict: Command result with success/error info
        """
        try:
            result = subprocess.run(
                [self.pdftk_cmd] + args,
                capture_output=True,
                text=True,
                check=True
            )
            return {
                "success": True,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": f"PDFtk error: {e.stderr}",
                "stdout": e.stdout,
                "stderr": e.stderr
            }
        except FileNotFoundError:
            return {
                "success": False,
                "error": "PDFtk not found. Please install PDFtk and ensure it's in your PATH."
            }
    
    def split_pdf(self, input_file: str, output_dir: str = None) -> Dict[str, Any]:
        """
        Split PDF into individual pages.
        
        Args:
            input_file: Input PDF file path
            output_dir: Output directory (optional)
            
        Returns:
            dict: Operation result
        """
        input_path = self.utils.resolve_path(input_file)
        if not self.utils.validate_pdf_file(input_path):
            return {"success": False, "error": f"Invalid PDF file: {input_file}"}
        
        if output_dir:
            output_path = self.utils.resolve_path(output_dir)
        else:
            output_path = input_path.parent / f"{input_path.stem}_split"
        
        output_path.mkdir(parents=True, exist_ok=True)
        output_pattern = output_path / f"{input_path.stem}_page_%02d.pdf"
        
        args = [str(input_path), "burst", "output", str(output_pattern)]
        result = self._run_command(args)
        
        if result["success"]:
            result["output_directory"] = str(output_path)
            result["output_pattern"] = str(output_pattern)
        
        return result
    
    def extract_pages(self, input_file: str, page_range: str, output_file: str) -> Dict[str, Any]:
        """
        Extract specific pages from PDF.
        
        Args:
            input_file: Input PDF file path
            page_range: Page range (e.g., "1-3", "1,3,5", "1-end")
            output_file: Output PDF path
            
        Returns:
            dict: Operation result
        """
        input_path = self.utils.resolve_path(input_file)
        if not self.utils.validate_pdf_file(input_path):
            return {"success": False, "error": f"Invalid PDF file: {input_file}"}
        
        output_path = self.utils.resolve_path(output_file)
        
        args = [str(input_path), "cat", page_range, "output", str(output_path)]
        result = self._run_command(args)
        
        if result["success"]:
            result["output_file"] = str(output_path)
            result["file_info"] = self.utils.get_file_info(output_path)
        
        return result
    
    def rotate_pages(self, input_file: str, rotation: str, output_file: str, page_range: str = None) -> Dict[str, Any]:
        """
        Rotate PDF pages.
        
        Args:
            input_file: Input PDF file path
            rotation: Rotation angle (90, 180, 270, or left, right, down)
            output_file: Output PDF path
            page_range: Page range to rotate (optional, default: all pages)
            
        Returns:
            dict: Operation result
        """
        input_path = self.utils.resolve_path(input_file)
        if not self.utils.validate_pdf_file(input_path):
            return {"success": False, "error": f"Invalid PDF file: {input_file}"}
        
        output_path = self.utils.resolve_path(output_file)
        
        # Convert rotation to PDFtk format
        rotation_map = {
            "90": "right", "right": "right",
            "180": "down", "down": "down",
            "270": "left", "left": "left"
        }
        
        rotation_cmd = rotation_map.get(rotation, rotation)
        
        if page_range:
            args = [str(input_path), "cat", f"{page_range}{rotation_cmd}", "output", str(output_path)]
        else:
            args = [str(input_path), "cat", f"1-end{rotation_cmd}", "output", str(output_path)]
        
        result = self._run_command(args)
        
        if result["success"]:
            result["output_file"] = str(output_path)
            result["file_info"] = self.utils.get_file_info(output_path)
        
        return result
    
    def encrypt_pdf_basic(self, input_file: str, output_file: str, user_password: str, owner_password: str = None) -> Dict[str, Any]:
        """
        Encrypt PDF with basic password protection.
        
        Args:
            input_file: Input PDF file path
            output_file: Output encrypted PDF path
            user_password: User password (for opening)
            owner_password: Owner password (for editing, optional)
            
        Returns:
            dict: Operation result
        """
        input_path = self.utils.resolve_path(input_file)
        if not self.utils.validate_pdf_file(input_path):
            return {"success": False, "error": f"Invalid PDF file: {input_file}"}
        
        output_path = self.utils.resolve_path(output_file)
        
        args = [str(input_path), "output", str(output_path), "user_pw", user_password]
        
        if owner_password:
            args.extend(["owner_pw", owner_password])
        
        result = self._run_command(args)
        
        if result["success"]:
            result["output_file"] = str(output_path)
            result["file_info"] = self.utils.get_file_info(output_path)
        
        return result
    
    def decrypt_pdf(self, input_file: str, output_file: str, password: str) -> Dict[str, Any]:
        """
        Decrypt password-protected PDF.
        
        Args:
            input_file: Input encrypted PDF file path
            output_file: Output decrypted PDF path
            password: PDF password
            
        Returns:
            dict: Operation result
        """
        input_path = self.utils.resolve_path(input_file)
        if not self.utils.validate_pdf_file(input_path):
            return {"success": False, "error": f"Invalid PDF file: {input_file}"}
        
        output_path = self.utils.resolve_path(output_file)
        
        args = [str(input_path), "input_pw", password, "output", str(output_path)]
        result = self._run_command(args)
        
        if result["success"]:
            result["output_file"] = str(output_path)
            result["file_info"] = self.utils.get_file_info(output_path)
        
        return result
    
    def update_metadata(self, input_file: str, output_file: str, metadata: Dict[str, str]) -> Dict[str, Any]:
        """
        Update PDF metadata.
        
        Args:
            input_file: Input PDF file path
            output_file: Output PDF path
            metadata: Metadata dictionary (Title, Author, Subject, Keywords, etc.)
            
        Returns:
            dict: Operation result
        """
        input_path = self.utils.resolve_path(input_file)
        if not self.utils.validate_pdf_file(input_path):
            return {"success": False, "error": f"Invalid PDF file: {input_file}"}
        
        output_path = self.utils.resolve_path(output_file)
        
        # Create temporary metadata file
        temp_metadata_file = input_path.parent / "temp_metadata.txt"
        
        try:
            with open(temp_metadata_file, 'w') as f:
                f.write("InfoBegin\n")
                for key, value in metadata.items():
                    f.write(f"InfoKey: {key}\n")
                    f.write(f"InfoValue: {value}\n")
            
            args = [str(input_path), "update_info", str(temp_metadata_file), "output", str(output_path)]
            result = self._run_command(args)
            
            if result["success"]:
                result["output_file"] = str(output_path)
                result["file_info"] = self.utils.get_file_info(output_path)
            
            return result
        
        finally:
            # Clean up temporary file
            temp_metadata_file.unlink(missing_ok=True)