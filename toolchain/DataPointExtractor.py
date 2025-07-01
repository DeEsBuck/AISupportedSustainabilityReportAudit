import re
import json
import openpyxl
import pdfplumber
import pymupdf
import helper.Helpy
import logging as log
import pandas as pd
import numpy as np
from pathlib import Path
from langchain.text_splitter import MarkdownTextSplitter

log_name = log.getLogger()
log.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=log.INFO)

class ExtractDataPoints():
    '''
    Extrahiert strukturierte Datenpunkte aus Excel/CSV/JSON und exportiert in verschiedene Formate.
    self.path = Path(file_path) // read_data_file: Src_file
    self.out_path = Path(output_path) // export_data_output: folder/extracted
    self.in_path = Path(input_path) // export_data_prompts: folder/prompts
    self.sheet_name = sheet_name //
    self.data = pd.DataFrame() // result Dataframe
    '''

    def __init__(self, file_path: str, output_path: str, input_path: str, sheet_name: str):
        self.path = Path(file_path)
        self.out_path = Path(output_path)
        self.in_path = Path(input_path)
        self.sheet_name = sheet_name
        self.data_frame = pd.DataFrame()
        self.data_xbrl = 'https://xbrl.efrag.org/e-esrs/'
        self.data_code = '0'
        self.column_names = [
            "ID", "ESRS", "DR", "Paragraph", "Related AR", "Name", "Data Type", "Linked Regulations", "Voluntary", "Code"
        ]


    def __str__(self):
        return f'{self.__class__.__name__}({self.path}, {self.out_path}, {self.input_path}, {self.sheet_name})'


    def read_data_file(self):
        '''
        Liest und bereinigt Excel, CSV oder JSON-Datei.
        '''
        if not self.path.exists():
            log.error(f"Datei nicht gefunden: {self.path}")
            raise FileNotFoundError(self.path)

        suffix = self.path.suffix.lower()
        if suffix in ['.xls', '.xlsx']:
            df_raw = pd.read_excel(self.path, engine="openpyxl", sheet_name=self.sheet_name)
        elif suffix == '.csv':
            df_raw = pd.read_csv(self.path)
        elif suffix == '.json':
            df_raw = pd.read_json(self.path, lines=True)
        else:
            raise ValueError(f"Nicht unterstützter Dateityp: {suffix}")

        df = df_raw[1:].copy()
        
        # Setze Spaltennamen aus der ersten Zeile
        cols = list(df_raw.iloc[0])
        seen = {}
        unique_cols = []
        for col in cols:
            if pd.isna(col) or col == "":
                col = "Index"
            col = str(col).strip()
            if col in seen:
                seen[col] += 1
                col = f"{col}_{seen[col]}"
            else:
                seen[col] = 0
            unique_cols.append(col)
    
        df.columns = unique_cols
        df.columns = [str(c).strip() for c in df.columns]
        df = df.dropna(how="all", axis=0)
        df = df.dropna(how="all", axis=1)
        
        # Mapping (falls notwendig)
        column_map = {
            "ID": "ID",
            "ESRS": "ESRS",
            "DR": "DR",
            "Paragraph": "Paragraph",
            "Related AR": "Related AR",
            "Name": "Name",
            "Data Type": "Data Type",
            "Code": "Code"
        }
        df.rename(columns=column_map, inplace=True)

        self.data_frame = pd.DataFrame(df, columns=self.column_names)
        return self.data_frame
    
    
    def convert_data_output(self):
        '''
        Erstellt alle Ausgabeformate (Markdown, JSON, Excel, CSV) aus dem DataFrame.
        '''
        df = pd.DataFrame(self.data_frame, columns=self.column_names)
        
        # Markdown
        markdown_output = df.to_markdown(index=False)
        with open(self.out_path.with_suffix('.md'), "w", encoding="utf-8") as f:
            f.write(markdown_output)
        
        # JSON
        json_output = df.to_dict(orient="records")
        with open(self.out_path.with_suffix('.json'), "w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)
        
        # Excel
        excel_output = df.to_excel(self.out_path.with_suffix('.xlsx'), index=False, engine='openpyxl')
        
        # CSV
        csv_output = df.to_csv(self.out_path.with_suffix('.csv'), index=False, encoding="utf-8")
        
        log.info("✅ Formate exportiert: markdown, json, excel, csv")
        return (json_output, markdown_output, excel_output, csv_output)
    
    
    def export_data_prompts(self):
        '''
        Exportiert Daten in JSONL-Format (für LLMs) und Prompt/Completion-Format.
        '''
        # JSONL für LLMs
        with open(self.in_path.with_suffix('.jsonl'), "w", encoding="utf-8") as jsonl_file:
            for record in self.data_frame.to_dict(orient="records"):
                jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        # TODO Anpassen, nicht Paragraph sondern Text und vorgegeben codierung:
        #  [Index - Textabschnitt - Code (z.B. G1-1_01) - Heading - Title - Seite]
        # Prompt/Completion
        with open(self.in_path.with_name(self.in_path.stem + "_prompts.jsonl"), "w", encoding="utf-8") as f:
            for record in self.data_frame.to_dict(orient="records"):
                prompt = f"Paragraph {record['Paragraph']} – {record['Name']}"
                completion = f"{record.get('Data Type', '')}".strip()
                f.write(json.dumps({"prompt": prompt, "completion": completion}) + "\n")

        log.info("✅ Export abgeschlossen: JSONL + Prompt-Format")
        return (jsonl_file, f)
    
    
    ## TODO: To extract Datapoints href XBRL machine readable Standard https://www.xbrl.org
    def _get_link_if_exists(self, cell: str) -> str | None:
        try:
            return cell.hyperlink.target
        except AttributeError:
            return None
    
    
    def extract_hyperlinks_from_xlsx(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        ## TODO: To extract Datapoints href XBRL machine readable Standard https://www.xbrl.org
        self.path = file_name
        self.sheet_name = sheet_name
        self.column_names = columns_to_parse
        row_header = 2
        :return: df
        '''
        ws = openpyxl.load_workbook(self.path)[self.sheet_name]
        for column in self.column_names:
            column_index = list(df.columns).index(column) + 1
            df[column] = [
                self._get_link_if_exists(ws.cell(row=row_offset + i, column=column_index))
                for i in range(len(df[column]))
            ]
        return df
    
    """
    self.path = Path(file_path) // read_data_file: Src_file
    self.out_path = Path(output_path) // export_data_output: folder / extracted
    self.in_path = Path(input_path) // export_data_prompts: folder / prompts
    self.sheet_name = sheet_name //
    self.data = pd.DataFrame() // result
    Dataframe
    """

    def get_datapoints_prompts(self) -> str:
        '''
        # JSONL für LLMs
        first wanna paraphrase. feed the llm with the defined Datapoints (DP)
        :return:
        '''
        try:
            with open(self.in_path.with_suffix('.jsonl'), "r", encoding="utf-8") as jsonl_file:
                return jsonl_file.read()
        except Exception as e:
            log.exception(f'{e}')
    
    
    def get_datapoints_finetuning(self) -> str:
        '''
        # Prompt/Completion
        :return:
        '''
        try:
            with open(self.in_path.with_name(self.in_path.stem + "_prompts.jsonl"), "r", encoding="utf-8") as jsonl_file:
                return jsonl_file.read()
        except Exception as e:
            log.exception(f'{e}')
    
    
    def transform_to_frame(self, to_frame: list[str]) -> pd.DataFrame:
        try:
            json_lines = to_frame[0].splitlines()
            records = [json.loads(line) for line in json_lines]
            data_frame = pd.DataFrame(records, columns=self.column_names)
            return data_frame
        except Exception as e:
            log.exception("Could not transform", exc_info=e)
    
    
    def run_pipeline(self):
        '''
        Führt den kompletten Datenverarbeitungsprozess aus –
        inkl. Markdown, JSON, Excel, CSV, JSONL & Prompt-Format.
        '''
        self.read_data_file()
        self.convert_data_output()  # ohne Parameter, alle Formate standardmäßig
        self.export_data_prompts()
        return
