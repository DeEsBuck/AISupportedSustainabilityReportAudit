import logging as log
import pandas as pd
from pathlib import Path
import helper.Helpy
import re
import json
import openpyxl
import pdfplumber
import pymupdf
from langchain.text_splitter import MarkdownTextSplitter

log_name = log.getLogger()
log.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=log.INFO)

class ExtractPDFReport():
    '''
    Testwise and rudimentär Raw-Data-PDF-Text-Extractor
    Uses Packages pdfplumber and pymupdf with MarkdownTextSplitter from langchain.text_splitter
    extract_text_from_pdf for pdfplumber.open(self.path)
    and
    extract_all_text(self) for pymupdf.open(self.path)
    Beispiel: Extrahiere alle Paragraphen-Referenzen wie "§ 5", "Art. 7", etc.
    regex_extract(self, full_text)
    Pipeline exports: tuple(full_text, md_chunks, regex_hits)
    '''
    
    def __init__(self, path: str, output_path: str):
        self.path = path
        self.out_path = output_path
    
    def __str__(self):
        return f'{self.__class__.__name__}({self.path})'
    
    def extract_text_from_pdf(self):
        with pdfplumber.open(self.path) as pdf:
            return [page.extract_text() for page in pdf.pages if page.extract_text()]
    
    def extract_all_text(self):
        with pymupdf.open(self.path) as doc:
            return "\n".join([page.get_text() for page in doc])
    
    def split_text_markdown(self, full_text):
        splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
        return splitter.split_text(full_text)
    
    def regex_extract(self, full_text):
        # Beispiel: Extrahiere alle Paragraphen-Referenzen wie "§ 5", "Art. 7", etc.
        matches = re.findall(r"(§ ?\d+[a-z]?|Art\. ?\d+[a-z]?)", full_text)
        return list(set(matches))  # Duplikate entfernen
    
    def export_outputs(self, full_text, md_chunks, regex_hits, output_base_path=None):
        if output_base_path:
            base = Path(output_base_path)
        else:
            base = Path(self.out_path)
        
        output_json = f"{base}.json"
        output_md = f"{base}.md"
        output_xlsx = f"{base}.xlsx"
        
        # Speichern als Markdown
        with open(output_md, "w", encoding="utf-8") as md_file:
            md_file.write("\n\n---\n\n".join(md_chunks))
        
        # JSON-Speicherung mit Regex-Matches
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump({
                "full_text": full_text,
                "chunks": md_chunks,
                "regex_matches": regex_hits
            }, f, indent=2, ensure_ascii=False)
        
        # Excel-Speicherung
        df = pd.DataFrame(md_chunks, columns=["chunk"])
        df["regex_hits"] = ", ".join(regex_hits)
        df.to_excel(output_xlsx, index=False)
        
        log.info(f'✅ Alles exportiert:\n📄 {output_json}\n📝 {output_md}\n📊 {output_xlsx}')
        return (output_json, output_md, output_xlsx)
    
    def run_pipeline(self):
        #texts = self.extract_text_from_pdf()
        # or
        texts = self.extract_all_text()
        full_text = "\n".join(texts)
        
        # Text splitten (MarkdownTextSplitter)
        md_chunks = self.split_text_markdown(full_text)
        
        # Regex-Analyse
        regex_hits = self.regex_extract(full_text)
        
        # Speichern als JSON, Markdown, Excel
        self.export_outputs(
            full_text=full_text,
            md_chunks=md_chunks,
            regex_hits=regex_hits
        )
        return (full_text, md_chunks, regex_hits)


