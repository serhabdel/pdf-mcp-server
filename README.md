# PDF MCP Server

[![GitHub stars](https://img.shields.io/github/stars/Sohaib-2/pdf-mcp-server?style=social)](https://github.com/Sohaib-2/pdf-mcp-server/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/Sohaib-2/pdf-mcp-server)](https://github.com/Sohaib-2/pdf-mcp-server/issues)
[![GitHub license](https://img.shields.io/github/license/Sohaib-2/pdf-mcp-server)](https://github.com/Sohaib-2/pdf-mcp-server/blob/main/LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> Transform PDF manipulation with AI-powered natural language commands through Claude integration

Comprehensive PDF toolkit that integrates seamlessly with Claude AI via MCP (Model Context Protocol). Perform complex PDF operations using simple conversational commands - merge, split, encrypt, optimize, and analyze PDFs effortlessly.

![Demo](demo.gif)

## 🚀 Quick Start

### Clone & Setup
```bash
git clone https://github.com/Sohaib-2/pdf-mcp-server.git
cd pdf-mcp-server
```

### Option 1: With Virtual Environment (Recommended)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux  
source .venv/bin/activate

pip install -r requirements.txt
```

### Option 2: Without Virtual Environment
```bash
pip install fastmcp requests pathlib
```

### Install PDF Tools
**PDFtk:**
```bash
# Ubuntu/Debian
sudo apt-get install pdftk
# macOS
brew install pdftk-java
# Windows: Download from https://www.pdflabs.com/tools/pdftk-the-pdf-toolkit/
```

**QPDF:**
```bash
# Ubuntu/Debian
sudo apt-get install qpdf
# macOS
brew install qpdf
# Windows: Download from https://qpdf.sourceforge.io/
```

## 🔧 Claude Desktop Integration

1. **Locate Claude config file:**
   - **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
   - **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

2. **Add PDF MCP Server:**

**With Virtual Environment:**
```json
{
  "mcpServers": {
    "pdf-tools": {
      "command": "C:\\path\\to\\pdf-mcp-server\\.venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\pdf-mcp-server\\server.py"]
    }
  }
}
```

**Without Virtual Environment:**
```json
{
  "mcpServers": {
    "pdf-tools": {
      "command": "python",
      "args": ["C:\\path\\to\\pdf-mcp-server\\server.py"]
    }
  }
}
```

**macOS/Linux with venv:**
```json
{
  "mcpServers": {
    "pdf-tools": {
      "command": "/path/to/pdf-mcp-server/.venv/bin/python",
      "args": ["/path/to/pdf-mcp-server/server.py"]
    }
  }
}
```

3. **Restart Claude Desktop**

4. **Start using natural language:**
   - *"Merge these 3 PDFs into one document"*
   - *"Encrypt my report with password protection"*
   - *"Extract pages 1-10 from this manual"*

## 📚 Complete Tool Reference

### Core Operations
| Tool | Description | Example |
|------|-------------|---------|
| `merge_pdfs` | Combine multiple PDFs | `merge_pdfs(['doc1.pdf', 'doc2.pdf'], 'combined.pdf')` |
| `split_pdf` | Split into individual pages | `split_pdf('document.pdf', './pages/')` |
| `extract_pages` | Extract specific page ranges | `extract_pages('book.pdf', '1-5,10,15-20', 'excerpt.pdf')` |
| `rotate_pages` | Rotate pages by degrees | `rotate_pages('scan.pdf', '90', 'rotated.pdf', '1-3')` |

### Security & Encryption
| Tool | Description | Example |
|------|-------------|---------|
| `encrypt_pdf` | AES-256 encryption | `encrypt_pdf('file.pdf', 'secure.pdf', 'password123')` |
| `encrypt_pdf_basic` | Basic password protection | `encrypt_pdf_basic('doc.pdf', 'protected.pdf', 'pass', 'admin')` |
| `decrypt_pdf` | Remove password protection | `decrypt_pdf('locked.pdf', 'unlocked.pdf', 'password')` |

### Optimization & Repair
| Tool | Description | Example |
|------|-------------|---------|
| `optimize_pdf` | Compress for web/email | `optimize_pdf('large.pdf', 'small.pdf', 'high')` |
| `repair_pdf` | Fix corrupted PDFs | `repair_pdf('broken.pdf', 'fixed.pdf')` |
| `check_pdf_integrity` | Validate PDF structure | `check_pdf_integrity('suspicious.pdf')` |

### Information & Analysis
| Tool | Description | Example |
|------|-------------|---------|
| `get_pdf_info` | Detailed metadata (JSON) | `get_pdf_info('document.pdf')` |
| `update_pdf_metadata` | Modify title/author/etc | `update_pdf_metadata('file.pdf', 'updated.pdf', title='New Title')` |
| `inspect_pdf_structure` | Internal structure analysis | `inspect_pdf_structure('complex.pdf', detailed=True)` |
| `extract_pdf_attachments` | Extract embedded files | `extract_pdf_attachments('portfolio.pdf', './attachments/')` |

### File Management
| Tool | Description | Example |
|------|-------------|---------|
| `download_pdf` | Download from URL | `download_pdf('https://example.com/file.pdf', 'local.pdf')` |
| `open_pdf_preview` | Open with system viewer | `open_pdf_preview('report.pdf', browser=False)` |
| `get_file_info` | File size/path details | `get_file_info('document.pdf')` |
| `configure_pdf_workspace` | Set working directory | `configure_pdf_workspace('/path/to/workspace')` |
| `count_pdfs_in_directory` | List PDFs in folder | `count_pdfs_in_directory('./pdf_folder/')` |

### System Management
| Tool | Description | Purpose |
|------|-------------|---------|
| `get_server_status` | Check tool availability | Verify PDFtk/QPDF installation |
| `list_default_directories` | Show search paths | Debug file resolution issues |
| `get_pdf_tools_help` | Complete documentation | In-app help reference |

## 💬 Natural Language Examples

**Document Management:**
- *"Combine all my research papers into one bibliography"*
- *"Split this 100-page manual into chapters"*
- *"Extract the executive summary from pages 2-4"*

**Security Operations:**
- *"Encrypt this contract with military-grade protection"*
- *"Remove password from this locked document"*
- *"Add owner permissions to prevent editing"*

**File Optimization:**
- *"Optimize all PDFs in my downloads folder"*
- *"Fix this corrupted presentation file"*

**Advanced Analysis:**
- *"Show me detailed metadata about this academic paper"*
- *"Analyze the internal structure for security audit"*

## 🗂️ File Path Handling

**Flexible path resolution:**
- **Absolute:** `C:\Documents\file.pdf`
- **Relative:** `../pdfs/document.pdf`
- **Filename only:** `report.pdf` (searches default directories)

**Default search order:**
1. `PDF_WORKSPACE` environment variable
2. `~/Documents/PDFs`
3. `~/Downloads`
4. `~/Desktop`
5. Current working directory

## ⚙️ Configuration

**Custom workspace:**
```python
configure_pdf_workspace('/path/to/your/pdfs')
```



**Check installation:**
```python
get_server_status()  # Verify PDFtk and QPDF availability
```

## 🛠️ Troubleshooting

**Common issues:**

| Problem | Solution |
|---------|----------|
| `PDFtk not found` | Install PDFtk and add to PATH |
| `QPDF error` | Install QPDF via package manager |
| `File not found` | Use `list_default_directories()` to check search paths |
| `Permission denied` | Run with appropriate file permissions |
| `Invalid PDF` | Use `check_pdf_integrity()` to validate file |

**Debug commands:**
```python
get_server_status()           # Check tool installation
list_default_directories()    # Verify search paths  
get_pdf_info('file.pdf')      # Validate PDF structure
```

## 🏗️ Architecture

```
pdf-mcp-server/
├── server.py              # FastMCP server with 16 tools
├── pdftk_tools.py         # PDFtk CLI wrapper
├── qpdf_tools.py          # QPDF CLI wrapper  
├── utils.py               # File utilities & path resolution
└── requirements.txt       # Python dependencies
```

**Built with:**
- [FastMCP](https://github.com/jlowin/fastmcp) - MCP server framework
- [PDFtk](https://www.pdflabs.com/tools/pdftk-the-pdf-toolkit/) - PDF manipulation
- [QPDF](https://qpdf.sourceforge.io/) - Advanced PDF processing

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License

## 👨‍💻 Author

**Sohaib-2** - [GitHub](https://github.com/Sohaib-2)

## 🌟 Acknowledgments

- [Anthropic](https://www.anthropic.com/) for MCP Protocol
- [PDFtk](https://www.pdflabs.com/) and [QPDF](https://qpdf.sourceforge.io/) teams
- Open source community for inspiration

---

⭐ **Star this repo** if it helped you! | 🐛 **Report issues** | 💡 **Request features**