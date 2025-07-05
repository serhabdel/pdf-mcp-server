import subprocess
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from utils import PDFUtils

class QPDFTools:
    """QPDF CLI wrapper for advanced PDF operations."""
    
    def __init__(self):
        self.utils = PDFUtils()
        self.qpdf_cmd = "qpdf"
    
    def _run_command(self, args: List[str]) -> Dict[str, Any]:
        """
        Run QPDF command and return result.
        
        Args:
            args: Command arguments
            
        Returns:
            dict: Command result with success/error info
        """
        try:
            result = subprocess.run(
                [self.qpdf_cmd] + args,
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
                "error": f"QPDF error: {e.stderr}",
                "stdout": e.stdout,
                "stderr": e.stderr
            }
        except FileNotFoundError:
            return {
                "success": False,
                "error": "QPDF not found. Please install QPDF and ensure it's in your PATH."
            }
    
    def merge_pdfs(self, input_files: List[str], output_file: str) -> Dict[str, Any]:
        """
        Merge multiple PDF files using QPDF.
        
        Args:
            input_files: List of input PDF file paths
            output_file: Output merged PDF path
            
        Returns:
            dict: Operation result
        """
        resolved_inputs = []
        for file_path in input_files:
            resolved_path = self.utils.resolve_path(file_path)
            if not self.utils.validate_pdf_file(resolved_path):
                return {"success": False, "error": f"Invalid PDF file: {file_path}"}
            resolved_inputs.append(str(resolved_path))
        
        output_path = self.utils.resolve_path(output_file)
        
        # Build command: qpdf --empty output.pdf --pages file1.pdf file2.pdf file3.pdf --
        args = ["--empty", str(output_path), "--pages"]
        args.extend(resolved_inputs)
        args.append("--")
        
        result = self._run_command(args)
        
        if result["success"]:
            result["output_file"] = str(output_path)
            result["file_info"] = self.utils.get_file_info(output_path)
        
        return result
    
    def optimize_pdf(self, input_file: str, output_file: str, compression_level: str = "medium") -> Dict[str, Any]:
        """
        Optimize PDF for web viewing.
        
        Args:
            input_file: Input PDF file path
            output_file: Output optimized PDF path
            compression_level: Compression level (low, medium, high)
            
        Returns:
            dict: Operation result
        """
        input_path = self.utils.resolve_path(input_file)
        if not self.utils.validate_pdf_file(input_path):
            return {"success": False, "error": f"Invalid PDF file: {input_file}"}
        
        output_path = self.utils.resolve_path(output_file)
        
        args = [str(input_path), str(output_path)]
        
        # Add optimization flags
        if compression_level == "high":
            args.extend(["--optimize-images", "--compress-streams=y"])
        elif compression_level == "medium":
            args.extend(["--optimize-images"])
        
        args.append("--linearize")  # Web optimization
        
        result = self._run_command(args)
        
        if result["success"]:
            result["output_file"] = str(output_path)
            result["file_info"] = self.utils.get_file_info(output_path)
            result["compression_level"] = compression_level
        
        return result
    
    def encrypt_pdf(self, input_file: str, output_file: str, user_password: str, owner_password: str = None) -> Dict[str, Any]:
        """
        Encrypt PDF with AES-256 encryption.
        
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
        
        args = [
            str(input_path),
            str(output_path),
            "--encrypt",
            user_password,
            owner_password or user_password,
            "256",
            "--"
        ]
        
        result = self._run_command(args)
        
        if result["success"]:
            result["output_file"] = str(output_path)
            result["file_info"] = self.utils.get_file_info(output_path)
            result["encryption"] = "AES-256"
        
        return result
    
    def inspect_structure(self, input_file: str, detailed: bool = False) -> Dict[str, Any]:
        """
        Inspect PDF internal structure.
        
        Args:
            input_file: Input PDF file path
            detailed: Include detailed object information
            
        Returns:
            dict: PDF structure information
        """
        input_path = self.utils.resolve_path(input_file)
        if not self.utils.validate_pdf_file(input_path):
            return {"success": False, "error": f"Invalid PDF file: {input_file}"}
        
        args = [str(input_path), "--show-all-data"] if detailed else [str(input_path), "--show-data"]
        result = self._run_command(args)
        
        if result["success"]:
            result["structure_info"] = result["stdout"]
            result["file_info"] = self.utils.get_file_info(input_path)
        
        return result
    
    def check_pdf(self, input_file: str) -> Dict[str, Any]:
        """
        Check PDF for errors and warnings.
        
        Args:
            input_file: Input PDF file path
            
        Returns:
            dict: PDF check results
        """
        input_path = self.utils.resolve_path(input_file)
        if not self.utils.validate_pdf_file(input_path):
            return {"success": False, "error": f"Invalid PDF file: {input_file}"}
        
        args = [str(input_path), "--check"]
        result = self._run_command(args)
        
        # QPDF check returns non-zero for warnings, but that's not always an error
        if not result["success"] and "warning" in result["stderr"].lower():
            result["success"] = True
            result["warnings"] = result["stderr"]
        
        if result["success"]:
            result["file_info"] = self.utils.get_file_info(input_path)
        
        return result
    
    def get_pdf_info(self, input_file: str) -> Dict[str, Any]:
        """
        Get comprehensive PDF information.
        
        Args:
            input_file: Input PDF file path
            
        Returns:
            dict: PDF information
        """
        input_path = self.utils.resolve_path(input_file)
        if not self.utils.validate_pdf_file(input_path):
            return {"success": False, "error": f"Invalid PDF file: {input_file}"}
        
        args = [str(input_path), "--json"]
        result = self._run_command(args)
        
        if result["success"]:
            try:
                result["pdf_data"] = json.loads(result["stdout"])
            except json.JSONDecodeError:
                result["pdf_data"] = {"raw": result["stdout"]}
            
            result["file_info"] = self.utils.get_file_info(input_path)
        
        return result
    
    def extract_attachments(self, input_file: str, output_dir: str = None) -> Dict[str, Any]:
        """
        Extract attachments from PDF.
        
        Args:
            input_file: Input PDF file path
            output_dir: Output directory for attachments
            
        Returns:
            dict: Operation result
        """
        input_path = self.utils.resolve_path(input_file)
        if not self.utils.validate_pdf_file(input_path):
            return {"success": False, "error": f"Invalid PDF file: {input_file}"}
        
        if output_dir:
            output_path = self.utils.resolve_path(output_dir)
        else:
            output_path = input_path.parent / f"{input_path.stem}_attachments"
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        args = [str(input_path), "--show-attachment=" + str(output_path)]
        result = self._run_command(args)
        
        if result["success"]:
            result["output_directory"] = str(output_path)
        
        return result
    
    def repair_pdf(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """
        Repair damaged PDF file.
        
        Args:
            input_file: Input PDF file path
            output_file: Output repaired PDF path
            
        Returns:
            dict: Operation result
        """
        input_path = self.utils.resolve_path(input_file)
        output_path = self.utils.resolve_path(output_file)
        
        args = [str(input_path), str(output_path)]
        result = self._run_command(args)
        
        if result["success"]:
            result["output_file"] = str(output_path)
            result["file_info"] = self.utils.get_file_info(output_path)
        
        return result