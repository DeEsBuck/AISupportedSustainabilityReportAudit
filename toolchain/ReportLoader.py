import os
import re
import json
import pymupdf
import pdfplumber
import openpyxl
import helper.Helpy
import logging as log
import numpy as np
import pandas as pd
from pathlib import Path
from helper.Konfiguration import DirectoryTree
from langchain.text_splitter import MarkdownTextSplitter

log_name = log.getLogger()
log.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=log.INFO)

class ImportJSONReport:
    '''
    loops through the json datapoints and streams blocks correctly as llm prompts.
    tool keep context with embedding and cross-encoding
    '''
    
    def __init__(self, path: str, output_path: str):
        self.path = Path(path)
        self.out_path = Path(output_path)
        self.df = pd.DataFrame()
    
    def __str__(self):
        return f'{self.__class__.__name__}({self.path})'
    
    def _load_json(self):
        '''
        Load and comprehend structured report output.
        :return: dict if successful, else creates a JSON file 'a.json' in the directory and returns default content
        '''
        json_file = self.path
        try:
            if os.path.isfile(json_file):
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data
            else:
                # If not a file, treat as directory
                directory = json_file if os.path.isdir(json_file) else os.path.dirname(json_file)
                if not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                output_file = os.path.join(directory, f'{DirectoryTree.API_TEMP_FILE}')
                self.out_path = output_file
                default_content = {}
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(default_content, f, ensure_ascii=False, indent=2)
                log.info(f"Created new JSON file at {output_file}")
                return default_content
        except Exception as e:
            log.error(f"Error handling path {json_file}: {e}")
            return None
    
    
    def _flatten_blocks(self, blocks, context):
        '''
        Flatten the list of blocks, combining with current context.
        '''
        rows = []
        for block in blocks:
            row = context.copy()
            row.update({
                'block_type': block.get('type', ''),
                'block_content': block.get('content', ''),
                'block_page': block.get('page', ''),
                'block_index': block.get('index', ''),
                'block_table_path': block.get('table_path', ''),
                'block_table_html': block.get('table_html', ''),
                'block_table_caption': block.get('table_caption', ''),
                'block_table_footnote': block.get('table_footnote', ''),
            })
            rows.append(row)
        return rows
    
    def _traverse(self, data, context, rows):
        '''
        Recursively traverse the JSON structure and collect flattened rows.
        '''
        if isinstance(data, dict):
            # Copy context to avoid mutation
            ctx = context.copy()
            # Collect known context keys
            for key in ['level', 'section', 'ESRS', 'DR', 'heading', 'block_page']:
                if key in data:
                    ctx[key] = data[key]
            # Handle blocks
            if 'blocks' in data:
                rows.extend(self._flatten_blocks(data['blocks'], ctx))
            # Recurse into 'data'
            if 'data' in data:
                self._traverse(data['data'], ctx, rows)
        elif isinstance(data, list):
            for item in data:
                self._traverse(item, context, rows)
    
    
    def load_and_flatten_json(self, context):
        '''
        Loads the JSON file, flattens it, and returns a pandas DataFrame.
        '''
        rows = []
        data = self._load_json()
        self._traverse(data.get('data', []), {
            'max_level': data.get('max_level', None)
        }, rows)
        self.df = pd.DataFrame(rows)
        return self.df
    
    
    def con_text_block(self, ctx):
        '''
        Gather and count for title first level, one text-block and dig for sub text-blocks.
        Save specific key-meta traversal, as encoder and decoder.
        Includes block_page for traceability.
        :return: dictionary with structure summary and encoder/decoder mapping
        '''
        # Load and flatten data
        df = self.load_and_flatten_json(ctx)
        if not isinstance(df, pd.DataFrame):
            return {"error": "No DataFrame loaded"}
         
        # Ensure index columns exist (including block_page)
        idx_cols = ["max_level", "level", "section", "ESRS", "DR", "heading", "block_page"]
        for col in idx_cols:
            if col not in df.columns:
                df[col] = np.nan
        
        # MultiIndex
        index = pd.MultiIndex.from_frame(df[idx_cols])
        df.index = index
        
        # First-level titles (level == 2 and block_type == 'title')
        first_level_titles = df[(df['level'] == 2) & (df['block_type'] == 'title')][
            ['block_content', 'block_page']].drop_duplicates().to_dict('records')
        
        # Text blocks at first level
        first_level_text_blocks = df[(df['level'] == 2) & (df['block_type'] == 'text')]
        text_block_count = first_level_text_blocks.shape[0]
        text_blocks = first_level_text_blocks[['block_content', 'block_page']].to_dict('records')
        
        # Sub text blocks (level > 2 and block_type == 'text')
        sub_text_blocks = df[(df['level'] > 2) & (df['block_type'] == 'text')]
        sub_text_block_count = sub_text_blocks.shape[0]
        sub_texts = sub_text_blocks[['block_content', 'block_page']].to_dict('records')
        
        # Encoder: index to block_content, Decoder: block_content to index
        encoder = {i: row['block_content'] for i, row in df.iterrows()}
        decoder = {v: k for k, v in encoder.items()}
        
        # Structured result
        result = {
            "first_level_titles": first_level_titles,
            "first_level_text_block_count": text_block_count,
            "first_level_text_blocks": text_blocks,
            "sub_text_block_count": sub_text_block_count,
            "sub_text_blocks": sub_texts,
            "encoder": encoder,
            "decoder": decoder,
            "multiindex_columns": idx_cols,
        }
        
        return result
    
    
    @staticmethod
    def create_structured_json_single(section):
        """
        Create a structured JSON object from a single section.
        The section should be a dict with keys: index, textabschnitt, code, heading, title, seite.
        """
        entry = {
            "Index": section.get("index"),
            "Code": section.get("code"),
            "Heading": section.get("heading"),
            "Title": section.get("title"),
            "Textabschnitt": section.get("textabschnitt"),
            "Seite": section.get("seite")
        }
        return json.dumps(entry, ensure_ascii=False, indent=2)
    
    
    @staticmethod
    def create_structured_json(sections):
        """
        Create a structured JSON list from a list of sections.
        Each section should be a dict with keys: index, textabschnitt, code, heading, title, seite.
        """
        result = []
        for section in sections:
            entry = {
                "Index": section.get("index"),
                "Code": section.get("code"),
                "Heading": section.get("heading"),
                "Title": section.get("title"),
                "Textabschnitt": section.get("textabschnitt"),
                "Seite": section.get("seite")
            }
            result.append(entry)
        return json.dumps(result, ensure_ascii=False, indent=2)

    
