import pandas as pd
import json

def align_report_blocks(feed_df, prompts_df, report):
    """
    Merge feed, prompts, and report content on 'ID' and output structured blocks for LLM search/query.
    Returns a list of dicts, each representing a data block/search object.
    """
    # Merge feed and prompts on 'ID'
    merged = pd.merge(feed_df, prompts_df, on="ID", how="left", suffixes=('_feed', '_prompt'))
    
    # Build a lookup for report block content (example: using first_level_text_blocks)
    # You may need to adjust this if your report structure is different.
    report_blocks = {}
    for block in report.get("first_level_text_blocks", []):
        # Try to extract ID from block_content, or map by order, or use external mapping.
        # Here, we assume you have an explicit 'ID' in each block (adjust as needed!).
        # If not, you may need to match by code, page, or position.
        block_id = block.get("ID") if isinstance(block, dict) else None
        if not block_id:
            continue  # Skip if no ID
        report_blocks[block_id] = block

    # Build the LLM search/query blocks
    results = []
    for idx, row in merged.iterrows():
        block_id = row["ID"]
        # Fetch report content if exists
        block_content = report_blocks.get(block_id, {})

        result = {
            "ID": block_id,
            "ESRS": row.get("ESRS"),
            "DR": row.get("DR"),
            "Paragraph": row.get("Paragraph"),
            "Name": row.get("Name"),
            "Data Type": row.get("Data Type"),
            # Add report context (text, page, etc.)
            "report_block_content": block_content.get("block_content") if isinstance(block_content, dict) else "",
            "report_block_page": block_content.get("block_page") if isinstance(block_content, dict) else "",
            # Optionally add more report-wide context
            "first_level_titles": report.get("first_level_titles", ""),
            "encoder": report.get("encoder", ""),
            "decoder": report.get("decoder", ""),
            "multiindex_columns": report.get("multiindex_columns", ""),
            "bosToken": {"bosToken": "<BOS>"},
        }
        results.append(result)
    return results

# Example usage:
# feed = ... # Your DataFrame from 'Feed'
# prompts_str = ... # Your raw JSONL string from 'Prompts'
# report = ... # Your parsed report dict

# Step 1: Parse prompts/datapoints
prompts_df = parse_prompts(prompts_str)

# Step 2: Align and enrich blocks
search_blocks = align_report_blocks(feed, prompts_df, report)

# Step 3: Use for LLM search/query
# Example: print one block
print(json.dumps(search_blocks[0], ensure_ascii=False, indent=2))

# Step 4: Optionally write all for retrieval or LLM index
with open("llm_search_blocks.jsonl", "w", encoding="utf-8") as f:
    for block in search_blocks:
        f.write(json.dumps(block, ensure_ascii=False) + "\n")