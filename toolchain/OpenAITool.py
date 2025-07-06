import os
import sys
import json
import itertools
import shutil
import threading
import time
import urllib.parse
import urllib.request
import logging as lg
import numpy as np
import lmstudio as lms
from openai import OpenAI
from dotenv import load_dotenv
from helper.Konfiguration import DirectoryTree
from helper.match_prompts_feed import build_block_responses
from toolchain.ReportLoader import ImportJSONReport
from toolchain.DataPointExtractor import ExtractDataPoints
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

# >_ lms get qwen2.5-7b-instruct-1
# >_ > bartowski/Qwen2.5-7B-Instruct-1M-GGUF
# >_ > Q4_K_S 4.46 GB Qwen2.5 7B Instruct 1M [Partial GPU offload possible] > Recommended
# >_ lms load qwen2.5-7b-instruct-1m
# >_ To use the model in the API/SDK, use the identifier "qwen2.5-7b-instruct-1m"

log_name = lg.getLogger()
log_name.setLevel(lg.INFO)

## Konfiguration ensure_directories
dir = DirectoryTree.DIRS
api_key = os.getenv("OPENAI_API_KEY")
# Initialize LM Studio client
client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
MODEL = "qwen2.5-7b-instruct-1m"

def match_esg_block(search_query: str) -> dict:
    '''
    search_query (datapoints trained LLM from LlamaModel) for Report tool (fetch Report Context depending Datablocks)
    :param search_query:
    :return:
    '''
    try:
        # TODO: write a parser and traverse KV for searching and Embedding for analyzing
        # Search for most relevant article
        report = ImportJSONReport(dir[0], dir[2])
        feed = report.con_text_block(DirectoryTree.SHEET_NAME)
        response_keys = build_block_responses(feed, report, bos_token_value="<BOS>")
        df = pd.DataFrame(data=feed, columns=response_keys)
        results = {
            "search": search_query,
        }
        with open(dir[2], "w", encoding="utf-8") as f:
            for entry in results:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                create_structured_json_single()
    except Exception as e:
        return {"status": "error", "message": str(e)}
        

# Define tool for LM Studio
ESG_TOOL = {
    "type": "function",
    "function": {
        "name": "match_esg_block",
        "description": (
            "Search Report Text-Blocks and fetch the ESRS-Datapoints for matching or relevant text-blocks. "
            "Always use this if the user is asking for something that is likely on specific ESRS query needs to allocate on Report page. "
            "If the user has a typo in their search query, correct it before searching."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "search_query": {
                    "type": "string",
                    "description": "Search query for finding equivalent or likely ESRS-Datapoints. Search each Block in ESG Reports which fit to the context",
                },
            },
            "required": ["search_query"],
        },
    },
}

def fetch_esrs_content(search_query: str) -> dict:
    '''
        search_query (datapoints trained LLM from LlamaModel) for Report tool (fetch Report Context depending Datablocks)
        :param search_query:
        :return:
        '''
    try:
        # TODO: write a parser and traverse KV for searching and Embedding for analyzing
        # Search for most relevant article
        report = ImportJSONReport(dir[0], dir[2])
        feed = report.con_text_block(DirectoryTree.SHEET_NAME)
        response_keys = ["Index", "Textabschnitt", "Code", "Heading", "Title", "Seite"]
        df = pd.DataFrame(data=feed, columns=response_keys)
        results = {
            "search": search_query,
        }
        with open(dir[2], "w", encoding="utf-8") as f:
            for entry in results:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                create_structured_json_single()
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Define tool for LM Studio
ESRS_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_esrs_content",
        "description": (
            "Fetch ESRS data points for retrieving relevant sustainability reports. "
            "Always use this if the user is asking for something that is likely on specific ESRS query needs to allocate on Report page. "
            "Get Context according to data points an prompt as text block "
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "search_query": {
                    "type": "string",
                    "description": "Search query for finding the ESRS-Datapoints in text blocks",
                },
            },
            "required": ["search_query"],
        },
    },
}

# Class for displaying the state of model processing
class Spinner:
    def __init__(self, message="Processing..."):
        self.spinner = itertools.cycle(["-", "/", "|", "\\"])
        self.busy = False
        self.delay = 0.1
        self.message = message
        self.thread = None

    def write(self, text):
        sys.stdout.write(text)
        sys.stdout.flush()

    def _spin(self):
        while self.busy:
            self.write(f"\r{self.message} {next(self.spinner)}")
            time.sleep(self.delay)
        self.write("\r\033[K")  # Clear the line

    def __enter__(self):
        self.busy = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.busy = False
        time.sleep(self.delay)
        if self.thread:
            self.thread.join()
        self.write("\r")  # Move cursor to beginning of line


def chat_loop():
    """
    Main chat loop that processes user input and handles tool calls.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant that can retrieve Sustainability Reports ESG-Reports. "
                "When asked about a topic, you can retrieve ESRS specific articles in Reports"
                "and cite information from them."
            ),
        }
    ]
    
    print(
        "Assistant: "
        "Hi! I can access requested Report to help answer your questions about ESG-Disclosures, "
        "European Sustainability Reporting Standards (ESRS) (data points) "
        "explore report Insights!"
    )
    print("(Type 'quit' to exit)")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "quit":
            break
        
        messages.append({"role": "user", "content": user_input})
        try:
            with Spinner("Thinking..."):
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=[ESG_TOOL, ESRS_TOOL],
                )

            if response.choices[0].message.tool_calls:
                # Handle all tool calls
                tool_calls = response.choices[0].message.tool_calls

                # Add all tool calls to messages
                messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": tool_call.type,
                                "function": tool_call.function,
                            }
                            for tool_call in tool_calls
                        ],
                    }
                )
            else:
                # Handle regular response
                print("\nAssistant:", response.choices[0].message.content)
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.choices[0].message.content,
                    }
                )
                
        except Exception as e:
            # TODO: Error details: Error code: 404 - {'error': {'message': 'Failed to load model "qwen2.5-7b-instruct-1m". Error: Failed to load model', 'type': 'invalid_request_error', 'param': 'model', 'code': 'model_not_found'}}
            print(
                f"\nError chatting with the LM Studio server!\n\n"
                f"Please ensure:\n"
                f"1. LM Studio server is running at 127.0.0.1:1234 (hostname:port)\n"
                f"2. Model '{MODEL}' is downloaded\n"
                f"3. Model '{MODEL}' is loaded, or that just-in-time model loading is enabled\n\n"
                f"Error details: {str(e)}\n"
                "See https://lmstudio.ai/docs/basics/server for more information"
            )
            exit(1)
