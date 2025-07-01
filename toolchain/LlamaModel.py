import os
import json
import torch
import logging as log
import numpy as np
import lmstudio as lms
import helper.Helpy as hp
import torch.nn.functional as F
from helper.Helpy import safe_strip, oba
from helper.Konfiguration import DirectoryTree
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoTokenizer, AutoModel
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

# === 3. Prompt-Funktion ===
def call_lmstudio(prompt = datapoints):
    try:
        response = model.complete(
            prompt=prompt,
            expected_completions=[esrs, index, paragraph, completion_type],
        )
        # print(response.choices[0].messages)
        return hp.safe_strip(response.text)
    except Exception as e:
        return f"[Fehler beim Modellaufruf: {e}]"
    

# === 4. Hauptfunktion ===
def promptRequest():
    df = prompts
    results = []
    
    # TODO: überarbeiten finetuning
    for i, row in df.iterrows():
        index = hp.safe_strip(row.get("ID"))
        esrs = hp.safe_strip(row.get("ESRS"))
        paragraph = hp.safe_strip(row.get("Paragraph"))
        name = hp.safe_strip(row.get("Name"))
        disc_requirement = hp.safe_strip(row.get("DR"))
        ar = hp.safe_strip(row.get("Related AR"))
        linked_reg = hp.safe_strip(row.get("Linked Regulations"))
        voluntary = hp.safe_strip(row.get("Voluntary"))
        completion_type = hp.safe_strip(row.get("Data Type"))
        
        if not index and not code:
            continue  # Leere Zeile überspringen
            # TODO Weiteres Error Handling
        
        # TODO: Prompt formulieren; Datenpunkte überreichen und mit vereinfachten ESG-Report gegenüberstellen
        # embedding
        prompt = f"{i} {esrs} {index} Paragraph {paragraph} – {name}"
        # print(f"🔍 Sende an LLM: {prompt[:80]}...")
        # print(prompt)
        lm_response = call_lmstudio(prompt)
        
        results.append({
            "prompt": prompt,
            "expected_completions": [esrs, index, paragraph, completion_type],
            "model_response": lm_response
        })
    
    if df.empty:
        print("⚠️ DataFrame ist leer – sende Testprompt...")
        test_response = call_lmstudio("Was ist die Aufgabe von ESRS G1?")
        print("📨 Antwort:", test_response)
    
    # === 5. Ergebnisse speichern ===
    with open(dir[3], "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    log.info('✅ Alle Prompts wurden verarbeitet und gespeichert.')




