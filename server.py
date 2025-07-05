import asyncio
from pathlib import Path
from fastmcp import FastMCP
from pdftk_tools import PDFtkTools
from qpdf_tools import QPDFTools
from utils import PDFUtils

# Initialize tools
mcp = FastMCP("PDF Tools Server")
pdftk = PDFtkTools()
qpdf = QPDFTools()
utils = PDFUtils()

# Core PDF Operations
@mcp.tool()
def merge_pdfs(input_files: list[str], output_file: str) -> dict:
    """
    Merge multiple PDF files into one.
    
    Args:
        input_files: List of PDF file paths to merge
        output_file: Output merged PDF file path
    
    Returns:
        Operation result with output file info
    """
    result = qpdf.merge_pdfs(input_files, output_file)
    
    if result["success"] and "output_file" in result:
        utils.open_file_externally(utils.resolve_path(result["output_file"]))
    
    return result

@mcp.tool()
def split_pdf(input_file: str, output_dir: str = None) -> dict:
    """
    Split PDF into individual pages.
    
    Args:
        input_file: Input PDF file path
        output_dir: Output directory (optional, defaults to input file directory)
    
    Returns:
        Operation result with output directory info
    """
    result = pdftk.split_pdf(input_file, output_dir)
    
    if result["success"] and "output_directory" in result:
        utils.open_file_externally(utils.resolve_path(result["output_directory"]))
    
    return result

@mcp.tool()
def extract_pages(input_file: str, page_range: str, output_file: str) -> dict:
    """
    Extract specific pages from PDF.
    
    Args:
        input_file: Input PDF file path
        page_range: Page range (e.g., "1-3", "1,3,5", "1-end")
        output_file: Output PDF file path
    
    Returns:
        Operation result with output file info
    """
    result = pdftk.extract_pages(input_file, page_range, output_file)
    
    if result["success"] and "output_file" in result:
        utils.open_file_externally(utils.resolve_path(result["output_file"]))
    
    return result

@mcp.tool()
def rotate_pages(input_file: str, rotation: str, output_file: str, page_range: str = None) -> dict:
    """
    Rotate PDF pages.
    
    Args:
        input_file: Input PDF file path
        rotation: Rotation angle (90, 180, 270, or left, right, down)
        output_file: Output PDF file path
        page_range: Page range to rotate (optional, defaults to all pages)
    
    Returns:
        Operation result with output file info
    """
    result = pdftk.rotate_pages(input_file, rotation, output_file, page_range)
    
    if result["success"] and "output_file" in result:
        utils.open_file_externally(utils.resolve_path(result["output_file"]))
    
    return result

# Security Operations
@mcp.tool()
def encrypt_pdf(input_file: str, output_file: str, user_password: str, owner_password: str = None) -> dict:
    """
    Encrypt PDF with AES-256 encryption using QPDF.
    
    Args:
        input_file: Input PDF file path
        output_file: Output encrypted PDF file path
        user_password: User password for opening the PDF
        owner_password: Owner password for editing (optional)
    
    Returns:
        Operation result with output file info
    """
    result = qpdf.encrypt_pdf(input_file, output_file, user_password, owner_password)
    
    if result["success"] and "output_file" in result:
        utils.open_file_externally(utils.resolve_path(result["output_file"]))
    
    return result

@mcp.tool()
def encrypt_pdf_basic(input_file: str, output_file: str, user_password: str, owner_password: str = None) -> dict:
    """
    Encrypt PDF with basic password protection using PDFtk.
    
    Args:
        input_file: Input PDF file path
        output_file: Output encrypted PDF file path
        user_password: User password for opening the PDF
        owner_password: Owner password for editing (optional)
    
    Returns:
        Operation result with output file info
    """
    result = pdftk.encrypt_pdf_basic(input_file, output_file, user_password, owner_password)
    
    if result["success"] and "output_file" in result:
        utils.open_file_externally(utils.resolve_path(result["output_file"]))
    
    return result

@mcp.tool()
def decrypt_pdf(input_file: str, output_file: str, password: str) -> dict:
    """
    Decrypt password-protected PDF using PDFtk.
    
    Args:
        input_file: Input encrypted PDF file path
        output_file: Output decrypted PDF file path
        password: PDF password
    
    Returns:
        Operation result with output file info
    """
    result = pdftk.decrypt_pdf(input_file, output_file, password)
    
    if result["success"] and "output_file" in result:
        utils.open_file_externally(utils.resolve_path(result["output_file"]))
    
    return result

# Information and Analysis
@mcp.tool()
def get_pdf_info(input_file: str) -> dict:
    """
    Get comprehensive PDF information using QPDF.
    
    Args:
        input_file: Input PDF file path
    
    Returns:
        Detailed PDF information in JSON format
    """
    return qpdf.get_pdf_info(input_file)

@mcp.tool()
def update_pdf_metadata(input_file: str, output_file: str, title: str = None, author: str = None, 
                       subject: str = None, keywords: str = None) -> dict:
    """
    Update PDF metadata.
    
    Args:
        input_file: Input PDF file path
        output_file: Output PDF file path
        title: PDF title
        author: PDF author
        subject: PDF subject
        keywords: PDF keywords
    
    Returns:
        Operation result with output file info
    """
    metadata = {}
    if title: metadata["Title"] = title
    if author: metadata["Author"] = author
    if subject: metadata["Subject"] = subject
    if keywords: metadata["Keywords"] = keywords
    
    result = pdftk.update_metadata(input_file, output_file, metadata)
    
    if result["success"] and "output_file" in result:
        utils.open_file_externally(utils.resolve_path(result["output_file"]))
    
    return result

# Advanced Operations
@mcp.tool()
def optimize_pdf(input_file: str, output_file: str, compression_level: str = "medium") -> dict:
    """
    Optimize PDF for web viewing using QPDF.
    
    Args:
        input_file: Input PDF file path
        output_file: Output optimized PDF file path
        compression_level: Compression level (low, medium, high)
    
    Returns:
        Operation result with output file info
    """
    result = qpdf.optimize_pdf(input_file, output_file, compression_level)
    
    if result["success"] and "output_file" in result:
        utils.open_file_externally(utils.resolve_path(result["output_file"]))
    
    return result

@mcp.tool()
def inspect_pdf_structure(input_file: str, detailed: bool = False) -> dict:
    """
    Inspect PDF internal structure using QPDF.
    
    Args:
        input_file: Input PDF file path
        detailed: Include detailed object information
    
    Returns:
        PDF structure information
    """
    return qpdf.inspect_structure(input_file, detailed)

@mcp.tool()
def check_pdf_integrity(input_file: str) -> dict:
    """
    Check PDF for errors and warnings using QPDF.
    
    Args:
        input_file: Input PDF file path
    
    Returns:
        PDF check results
    """
    return qpdf.check_pdf(input_file)

@mcp.tool()
def extract_pdf_attachments(input_file: str, output_dir: str = None) -> dict:
    """
    Extract attachments from PDF using QPDF.
    
    Args:
        input_file: Input PDF file path
        output_dir: Output directory for attachments (optional)
    
    Returns:
        Operation result with extracted attachments info
    """
    result = qpdf.extract_attachments(input_file, output_dir)
    
    if result["success"] and "output_directory" in result:
        utils.open_file_externally(utils.resolve_path(result["output_directory"]))
    
    return result

@mcp.tool()
def repair_pdf(input_file: str, output_file: str) -> dict:
    """
    Repair damaged PDF file using QPDF.
    
    Args:
        input_file: Input PDF file path
        output_file: Output repaired PDF file path
    
    Returns:
        Operation result with output file info
    """
    result = qpdf.repair_pdf(input_file, output_file)
    
    if result["success"] and "output_file" in result:
        utils.open_file_externally(utils.resolve_path(result["output_file"]))
    
    return result

# Help and Information Tools
@mcp.tool()
def get_pdf_tools_help() -> dict:
    """
    Get comprehensive help information about PDF tools and operations.
    
    Returns:
        dict: Complete guide on using the PDF MCP server
    """
    help_info = {
        "server_info": {
            "name": "PDF MCP Server",
            "version": "1.0.0",
            "description": "Complete PDF manipulation toolkit using PDFtk and QPDF",
            "tools_available": ["PDFtk", "QPDF", "File Utilities"]
        },
        "file_locations": {
            "default_directory": str(utils.default_dirs[0]),
            "supported_paths": [
                "Absolute paths: C:\\path\\to\\file.pdf",
                "Relative to default: filename.pdf",
                "Relative paths: ../folder/file.pdf"
            ],
            "output_behavior": "Files are saved in same directory as input unless specified"
        },
        "operations": {
            "basic": {
                "merge_pdfs": "Combine multiple PDFs into one",
                "split_pdf": "Split PDF into individual pages",
                "extract_pages": "Extract specific page ranges",
                "rotate_pages": "Rotate pages (90°, 180°, 270°)"
            },
            "security": {
                "encrypt_pdf": "Advanced AES-256 encryption",
                "encrypt_pdf_basic": "Basic password encryption",
                "decrypt_pdf": "Decrypt password-protected PDFs"
            },
            "optimization": {
                "optimize_pdf": "Compress and optimize for web",
                "repair_pdf": "Fix damaged PDF files",
                "check_pdf_integrity": "Validate PDF structure"
            },
            "information": {
                "get_pdf_info": "Detailed JSON metadata using QPDF",
                "update_pdf_metadata": "Update PDF metadata fields",
                "inspect_pdf_structure": "Internal PDF structure analysis",
                "extract_pdf_attachments": "Extract embedded files"
            }
        },
        "file_naming": {
            "patterns": {
                "merged": "original_merged.pdf",
                "split": "original_page_01.pdf, original_page_02.pdf",
                "encrypted": "original_encrypted.pdf",
                "optimized": "original_optimized.pdf"
            },
            "custom_output": "Always specify output filename for control"
        },
        "tips": {
            "file_paths": "Use forward slashes or double backslashes in Windows paths",
            "page_ranges": "Format: '1-3', '1,3,5', '1-end'",
            "rotations": "Use: 90, 180, 270, 'left', 'right', 'down'",
            "compression": "Levels: 'low', 'medium', 'high'",
            "preview": "Files automatically open after successful operations"
        },
        "examples": {
            "merge": "merge_pdfs(['doc1.pdf', 'doc2.pdf'], 'combined.pdf')",
            "extract": "extract_pages('document.pdf', '1-5', 'first_five.pdf')",
            "encrypt": "encrypt_pdf('file.pdf', 'secure.pdf', 'password123')",
            "info": "get_pdf_info('document.pdf')"
        }
    }
    return {"success": True, "help": help_info}

@mcp.tool()
def list_default_directories() -> dict:
    """
    Show default PDF directories and their status.
    
    Returns:
        dict: Information about default directories
    """
    dirs_info = []
    for i, dir_path in enumerate(utils.default_dirs):
        path_obj = Path(dir_path)
        dirs_info.append({
            "priority": i + 1,
            "path": str(path_obj),
            "exists": path_obj.exists(),
            "is_default": i == 0,
            "description": [
                "Primary PDF workspace (PDF_WORKSPACE env var)",
                "Downloads folder",
                "Desktop folder", 
                "Current working directory"
            ][i]
        })
    
    return {
        "success": True,
        "directories": dirs_info,
        "note": "Files are resolved in priority order. First existing directory is used."
    }

@mcp.tool()
def get_server_status() -> dict:
    """
    Check server status and tool availability.
    
    Returns:
        dict: Server and tools status
    """
    # Test PDFtk
    pdftk_status = pdftk._run_command(["--version"])
    
    # Test QPDF  
    qpdf_status = qpdf._run_command(["--version"])
    
    return {
        "success": True,
        "server_status": "Running",
        "tools": {
            "pdftk": {
                "available": pdftk_status["success"],
                "version": pdftk_status.get("stdout", "").strip() if pdftk_status["success"] else None,
                "error": pdftk_status.get("error") if not pdftk_status["success"] else None
            },
            "qpdf": {
                "available": qpdf_status["success"], 
                "version": qpdf_status.get("stdout", "").strip() if qpdf_status["success"] else None,
                "error": qpdf_status.get("error") if not qpdf_status["success"] else None
            }
        },
        "default_directory": str(utils.default_dirs[0]),
        "operations_count": 16
    }

# File Utility Operations
@mcp.tool()
def download_pdf(url: str, output_file: str) -> dict:
    """
    Download PDF from URL.
    
    Args:
        url: PDF URL to download
        output_file: Local file path to save PDF
    
    Returns:
        Download result with file info
    """
    output_path = utils.resolve_path(output_file)
    success = utils.download_file(url, output_path)
    
    if success:
        file_info = utils.get_file_info(output_path)
        utils.open_file_externally(output_path)
        return {
            "success": True,
            "output_file": str(output_path),
            "file_info": file_info
        }
    else:
        return {
            "success": False,
            "error": f"Failed to download PDF from {url}"
        }

@mcp.tool()
def open_pdf_preview(file_path: str, browser: bool = False) -> dict:
    """
    Open PDF file for preview.
    
    Args:
        file_path: PDF file path to open
        browser: Open in web browser instead of default application
    
    Returns:
        Operation result
    """
    resolved_path = utils.resolve_path(file_path)
    
    if not utils.validate_pdf_file(resolved_path):
        return {"success": False, "error": f"Invalid PDF file: {file_path}"}
    
    if browser:
        success = utils.open_in_browser(resolved_path)
    else:
        success = utils.open_file_externally(resolved_path)
    
    return {
        "success": success,
        "file_path": str(resolved_path),
        "opened_in": "browser" if browser else "default_application"
    }

@mcp.tool()
def get_file_info(file_path: str) -> dict:
    """
    Get file information.
    
    Args:
        file_path: File path to analyze
    
    Returns:
        File information
    """
    resolved_path = utils.resolve_path(file_path)
    return utils.get_file_info(resolved_path)

@mcp.tool()
def configure_pdf_workspace(directory_path: str) -> dict:
    """
    Set custom PDF workspace directory.
    
    Args:
        directory_path: Path to new PDF workspace directory
        
    Returns:
        dict: Configuration result
    """
    import os
    from pathlib import Path
    
    new_path = Path(directory_path)
    
    # Create directory if it doesn't exist
    try:
        new_path.mkdir(parents=True, exist_ok=True)
        
        # Update environment variable
        os.environ['PDF_WORKSPACE'] = str(new_path)
        
        # Update utils default directories
        utils.default_dirs[0] = new_path
        utils._ensure_default_dir()
        
        # Update tools' utils instances
        pdftk.utils.default_dirs[0] = new_path
        qpdf.utils.default_dirs[0] = new_path
        
        return {
            "success": True,
            "message": f"PDF workspace set to: {new_path}",
            "previous_directory": str(utils.default_dirs[0]),
            "new_directory": str(new_path)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to set workspace: {str(e)}"
        }

@mcp.tool()
def count_pdfs_in_directory(directory_path: str = None) -> dict:
    """
    Count PDF files in specified or default directory.
    
    Args:
        directory_path: Directory to scan (optional, uses default if not provided)
        
    Returns:
        dict: PDF count and file list
    """
    from pathlib import Path
    
    if directory_path:
        scan_path = utils.resolve_path(directory_path)
    else:
        scan_path = Path(utils.default_dirs[0])
    
    if not scan_path.exists():
        return {
            "success": False,
            "error": f"Directory does not exist: {scan_path}"
        }
    
    if not scan_path.is_dir():
        return {
            "success": False,
            "error": f"Path is not a directory: {scan_path}"
        }
    
    pdf_files = list(scan_path.glob("*.pdf"))
    
    file_info = []
    for pdf_file in pdf_files:
        file_stat = pdf_file.stat()
        file_info.append({
            "name": pdf_file.name,
            "size_mb": round(file_stat.st_size / (1024 * 1024), 2),
            "modified": file_stat.st_mtime
        })
    
    return {
        "success": True,
        "directory": str(scan_path),
        "pdf_count": len(pdf_files),
        "files": file_info,
        "total_size_mb": round(sum(f["size_mb"] for f in file_info), 2)
    }

if __name__ == "__main__":
    mcp.run()