import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Protocol

import torch
import torch.nn.functional as F
import msgspec
import pandas as pd
import numpy as np
import lmstudio as lms

from lmstudio import (
    BaseModel,
    ModelSchema,
    ToolFunctionDef,
    LMStudioPredictionError,
    ToolCallRequest
)
from helper.match_prompts_feed import build_block_responses
from helper.Helpy import safe_strip, oba, parse_prompts, prompt_line
from helper.Konfiguration import DirectoryTree
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoTokenizer, AutoModel
from toolchain.ReportLoader import ImportJSONReport
from toolchain.DataPointExtractor import ExtractDataPoints

# --- Logging setup ---
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# --- Schema Definitions ---
model_json_schema = []
response_schema = {
        "type": "object",
        "properties": {
            "first_level_titles": {"type": "string"},
            "first_level_text_block_count": {"type": "integer"},
            "first_level_text_blocks": {"type": "string"},
            "sub_text_block_count": {"type": "integer"},
            "sub_text_blocks": {"type": "string"},
            "encoder": {"type": "string"},
            "decoder": {"type": "string"},
            "multiindex_columns": {"type": "string"},
        },
        "required": ["first_level_titles", "encoder", "decoder"],
        "bosToken": {
            "bosToken": "<BOS>"
        }
    }

response_keys = {
        "type": "object",
        "properties": {
            "Index": {"type": "integer"},
            "Textabschnitt": {"type": "string"},
            "Code": {"type": "string"},
            "Heading": {"type": "string"},
            "Title": {"type": "string"},
            "Seite": {"type": "integer"},
        },
        "required": ["Index", "Textabschnitt", "Code", "Heading", "Title", "Seite"],
        "bosToken": {
            "bosToken": "<BOS>"
        }
    }

# --- Data Loading ---
dir_paths = DirectoryTree.DIRS
dirs = DirectoryTree.DIRS
datapoints = ExtractDataPoints(dirs[4],
                                   dirs[5],
                                   dirs[6],
                                   DirectoryTree.SHEET_NAME)
prompts = datapoints.read_data_file()
report = ImportJSONReport(dir_paths[0], dir_paths[1])
feed = report.con_text_block(DirectoryTree.SHEET_NAME)
logger.info(f"Data loaded. Prompts: {prompts}, Feed: {feed}")

# --- Schema Definitions ---
# --- Step 1: Parse Prompt Strings into a DataFrame ---
# --- Step 2: Merge feed and prompts DataFrames on 'ID' ---
# --- Step 3: Build Response Schema Per Block ---
idx_cols = ["max_level", "level", "section", "ESRS", "DR", "heading", "block_page"]
#feed_df = pd.DataFrame(feed, columns=idx_cols)
feed_df = report.load_and_flatten_json(DirectoryTree.SHEET_NAME)
response_key = report.create_structured_json(feed_df)
assert 'Index' in feed_df.columns, f"feed_df missing 'Index' column, columns: {feed_df.columns}"

# If you have JSONL string for prompts
prompts_df = parse_prompts(prompts)  # ensure this is a DataFrame with 'Index'
assert 'Index' in prompts_df.columns, f"prompts_df missing 'Index' column, columns: {prompts_df.columns}"

# Confirm columns
print("feed_df columns:", feed_df.columns)
print("prompts_df columns:", prompts_df.columns)

merged = pd.merge(feed_df, prompts_df, on="Index", how="left", suffixes=('_feed', '_prompt'))
responses = build_block_responses(merged, feed, bos_token_value="<BOS>")
for resp in responses:
    print(json.dumps(resp, ensure_ascii=False, indent=2))
    
# --- Tool Definitions ---
def create_file(name: str, content: str) -> str:
    """Create a file with the given name and content."""
    dest_path = Path(name)
    if dest_path.exists():
        return f"Error: File '{name}' already exists."
    try:
        dest_path.write_text(content, encoding="utf-8")
        return "File created."
    except Exception as exc:
        return f"Error: {exc!r}"

def _raise_exc_in_client(exc: LMStudioPredictionError, request: ToolCallRequest | None) -> None:
    raise exc

def does_chat_fit_in_context(model: lms.LLM, chat: lms.Chat) -> bool:
    """Check if the chat fits in the model context window."""
    formatted = model.apply_prompt_template(chat)
    token_count = len(model.tokenize(formatted))
    context_length = model.get_context_length()
    return token_count < context_length

# --- Model Loading ---
def init():
    # Configuration
    SERVER_API_HOST = os.getenv("SERVER_API_HOST", "http://localhost:1234")
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama-3.2-1b-instruct")
    TOOL_MODEL = os.getenv("TOOL_MODEL", "qwen2.5")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-qwen3-embedding-0.6b")
    config = {
        "maxTokens": 120,
        "temperature": 0,
        "messages": [''],
        "stream": True,
        "stopStrings": ["\n\n"],
        "maxPredictedTokens": 65,
            "contextLength": 1024,
        "gpu": {
            "ratio": 0.1,
        }
    }
    ttl = 900

    lms.configure_default_client(SERVER_API_HOST)
    client = lms.get_default_client()
    logger.info("✅ Client loaded")

    logger.info("🚀 Loading models...")
    model = client.llm.load_new_instance(DEFAULT_MODEL, config=config, ttl=ttl)
    model_complete = client.llm.load_new_instance(TOOL_MODEL, config=config, ttl=ttl)
    model_embedding = client.embedding.load_new_instance(EMBED_MODEL, config=config, ttl=ttl)

    logger.info(f"Context length: {model_complete.get_context_length()}")

    chat = lms.Chat("You are a task focused AI assistant")
    chat_h = lms.Chat.from_history({
        "messages": [
            {"role": "user", "content": "I am an Auditor and want to obtain ESRS Datapoints acccording to ESG reports I requested."},
            {"role": "assistant", "content": "I will assist you as i can."},
        ]
    })

    received = []
    try:
        while True:
            user_input = input("You (quit/exit): ").strip()
            if user_input.lower() in {"quit", "exit"}:
                break
            chat.add_user_message(user_input)
            print("Bot: ", end="", flush=True)
            result = model.respond(
                chat,
                config={"temperature": 0.6, "maxTokens": 50},
                on_message=chat_h
            )
            print(result)
            stats = result.stats
            print(f"Accepted {stats.accepted_draft_tokens_count}/{stats.predicted_tokens_count} tokens")
            print(result.parsed)
            print("Model used:", result.model_info.display_name)
            print("Predicted tokens:", stats.predicted_tokens_count)
            print("Time to first token (seconds):", stats.time_to_first_token_sec)
            print("Stop reason:", stats.stop_reason)
            print("Fits in context:", does_chat_fit_in_context(model, chat))
            
            embedding_vector = model_embedding.embed(chat_h)
            logger.info(f"Embedding vector: {embedding_vector}")

    except Exception as exc:
        logger.error(f"Error in chat loop: {exc}")
    
    # Completion Example
    complete = model_complete.complete(
        "My name is",
        on_prompt_processing_progress=lambda progress: print(f"{progress * 100:.1f}% complete"),
        on_message=received.append,
    )
    for fragment in complete:
        print(fragment.content, end="", flush=True)
    if hasattr(complete, "result"):
        print(complete.result())

    # Tool use Example
    complete_tool = model_complete.act(
        "Please create a file named Datapoint-Report_log.md with your understanding of the meaning of life.",
        [create_file],
        handle_invalid_tool_request=_raise_exc_in_client,
    )

    # Cleanup
    model.unload()
    model_complete.unload()
    model_embedding.unload()

