import pandas as pd
import json

def build_block_responses(merged_df, report, bos_token_value="<BOS>"):
    responses = []
    for idx, row in merged_df.iterrows():
        block_id = row.get('Index')
        # You can also pull out other values from the row as needed
        response = {
            "first_level_titles": str(report.get("first_level_titles", "")),
            "first_level_text_block_count": report.get("first_level_text_block_count", 0),
            "first_level_text_blocks": str(report.get("first_level_text_blocks", "")),
            "sub_text_block_count": report.get("sub_text_block_count", 0),
            "sub_text_blocks": str(report.get("sub_text_blocks", "")),
            "encoder": str(report.get("encoder", "")),
            "decoder": str(report.get("decoder", "")),
            "multiindex_columns": str(report.get("multiindex_columns", "")),
            "bosToken": {"bosToken": bos_token_value},
            # Include block code/ID for traceability
            "block_id": block_id,
            # Optionally add prompt fields for traceability
            "Index": row["Index"],
            "Name": row.get("Name", None),
            "Paragraph": row.get("Paragraph", None),
            "Data Type": row.get("Data Type", None),
        }
        responses.append(response)
    return responses

# Usage:
# responses = build_block_responses(merged, report, bos_token_value="<BOS>")
# for resp in responses:
#     print(json.dumps(resp, ensure_ascii=False, indent=2))
