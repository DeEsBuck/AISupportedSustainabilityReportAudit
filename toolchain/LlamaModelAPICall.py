import os
import json
import torch
import logging as log
import pandas as pd
import numpy as np
import lmstudio as lms
import helper.Helpy as hp
import torch.nn.functional as F
from helper.Helpy import safe_strip, oba
from helper.Konfiguration import DirectoryTree
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoTokenizer, AutoModel
from toolchain.ReportLoader import ImportJSONReport
from toolchain.DataPointExtractor import ExtractDataPoints

from datetime import datetime

log_name = log.getLogger()
log.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=log.INFO)

"""
# The above code requires the Llama 3.2 1B model. If you don't have the model, run the following command in the terminal to download it.
# >_ lms get llama-3.2-1b-instruct
# >_ lms load llama-3.2-1b-instruct
>_ To use the model in the API/SDK, use the identifier "llama-3.2-1b-instruct"
# === 1. Modellkonfiguration ===
MAX_TOKENS = 256
TEMPERATURE = 0
:return:
"""

## Konfiguration ensure_directories

# === 1. Modellkonfiguration ===
MODEL="llama-3.2-1b-instruct"
TOP_P=0.1
TOP_K=0
MESSAGES=['']
TEMPERATURE=0
MAX_TOKENS=65
STREAM=True
STOP='NULL'
PRESENCE_PENALTY=0
FREQUENCY_PENALTY=0
LOGIT_BIAS={}
REPEAT_PENALTY=0
SEED=0
METADATA={}
TTL=15

# === 2. Modell laden ===
# Getting Local Models
client = lms.get_default_client()
print("✅ Client geladen:", client)
print("🚀 Modell wird geladen...")
model = client.llm.load_new_instance(MODEL, config={
    "contextLength": 1024,
    "gpu": {
        "ratio": 0.1,
    }
})
print("✅ Modell geladen:", model)

dir = DirectoryTree.ENUMDIR
datapoints = ExtractDataPoints(dir[4], dir[6], dir[7], DirectoryTree.SHEET_NAME)
prompts = datapoints.read_data_file()

report = ImportJSONReport(dir[0], dir[2])
feed = report.con_text_block(DirectoryTree.SHEET_NAME)
response_keys = ["Index", "Textabschnitt", "Code", "Heading", "Title", "Seite"]
df = pd.DataFrame(feed,columns=response_keys)
completion_sections = []

# === 3. Prompt-Funktion ===
def call_lmstudio(prompt = prompts):
    try:
        response = model.completion(
            prompt=prompt,
            expected_completions=completion_sections,
        )
        print(response.choices[0].messages)
        return hp.safe_strip(response.text)
    except Exception as e:
        return f"[Fehler beim Modellaufruf: {e}]"
    

# === 4. Hauptfunktion ===
def promptRequest():
    df1 = df.merge(prompts, how="cross",left_index="Index", right_index="Code") # brauch ich als dataset datapoints x report sections
    results = []
    
    # TODO: überarbeiten finetuning
    for i, row in df1.iterrows():
        completion_index = hp.safe_strip(row.get("Index"))
        completion_code = hp.safe_strip(row.get("ID"))
        completion_id = hp.safe_strip(row.get("Code"))
        esrs = hp.safe_strip(row.get("ESRS"))
        paragraph = hp.safe_strip(row.get("Paragraph"))
        name = hp.safe_strip(row.get("Name"))
        disc_requirement = hp.safe_strip(row.get("DR"))
        ar = hp.safe_strip(row.get("Related AR"))
        linked_reg = hp.safe_strip(row.get("Linked Regulations"))
        voluntary = hp.safe_strip(row.get("Voluntary"))
        type = hp.safe_strip(row.get("Data Type"))
        completion_section = hp.safe_strip(row.get("Textabschnitt"))
        completion_heading = hp.safe_strip(row.get("Heading"))
        completion_title = hp.safe_strip(row.get("Title"))
        completion_page = hp.safe_strip(row.get("Seite"))
        
        if not completion_index and not completion_code:
            continue  # Leere Zeile überspringen
            # TODO Weiteres Error Handling
            
        completion_sections.append({
            "completion_index": completion_index,
            "completion_section": completion_section,
            "completion_code": completion_code,
            "completion_heading": completion_heading,
            "completion_title": completion_title,
            "completion_page": completion_page
        })
        
        # TODO: Prompt formulieren; Datenpunkte überreichen und mit vereinfachten ESG-Report gegenüberstellen
        # embedding   datapoints x report
        # response_keys = ["Index", "Textabschnitt", "Code", "Heading", "Title", "Seite"]
        prompt = f"{completion_index} {completion_section} {completion_code} {completion_heading} {completion_title} {completion_page}"  #  | {completion_name}
        # print(f"🔍 Sende an LLM: {prompt[:80]}...")
        lm_response = call_lmstudio(prompt)
        
        results.append({
            "prompt": prompt,
            "expected_completions": completion_sections,
            "model_response": lm_response
        })
    
    if df1.empty:
        print("⚠️ DataFrame ist leer – sende Testprompt...")
        test_response = call_lmstudio("Was ist die Aufgabe von ESRS G1?")
        print("📨 Antwort:", test_response)
    
    cheat = ImportJSONReport.create_structured_json(completion_sections)
    
    # === 5. Ergebnisse speichern ===
    with open(dir[1], "w", encoding="utf-8") as f:
        for entry in cheat:  # results
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    log.info('✅ Alle Prompts wurden verarbeitet und gespeichert.')




