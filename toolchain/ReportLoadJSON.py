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
        :return: dict if successful, else None
        '''
        try:
            json_file = self.path
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading JSON file: {e}")
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
    
    def export_to_json(self, keys, data):
        """
        Takes a list of keys and a list of value lists,
        and returns a JSON string where each entry is a dict mapping keys to values.
        :param keys: List of key names (e.g. ["Index", "Textabschnitt", "Code", "Heading", "Title", "Seite"])
        :param data: List of lists (or tuples), where each sublist contains values corresponding to the keys
        :return: JSON string of the structured list
        """
        result = []
        for row in data:
            entry = dict(zip(keys, row))
            result.append(entry)
        return json.dumps(result, ensure_ascii=False, indent=2)

    
