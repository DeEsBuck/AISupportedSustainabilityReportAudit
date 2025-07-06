import logging as log
import pandas as pd
import helper.Helpy as hp
from helper.Konfiguration import DirectoryTree
from helper.ReportExtractor import ExtractPDFReport
from toolchain.ReportLoader import ImportJSONReport
from toolchain.DataPointExtractor import ExtractDataPoints
from toolchain.OpenAITool import chat_loop
from toolchain.ModelAPICall import init

log_name = log.getLogger('')
log.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=log.INFO)

if __name__ == "__main__":
    ## Konfiguration ensure_directories
    #DirectoryTree.ensure_directories('.', [dir[2]])
    
    dirs = DirectoryTree.DIRS
    datapoints = ExtractDataPoints(dirs[4],
                                   dirs[5],
                                   dirs[6],
                                   DirectoryTree.SHEET_NAME)
    datapoints.run_pipeline()
    prompts = [datapoints.get_datapoints_prompts()]
    feed = datapoints.transform_to_frame(prompts)
    
    report = ImportJSONReport(dirs[0], dirs[1])
    block = report.con_text_block(DirectoryTree.SHEET_NAME)
    log.info(f"Data loaded. Prompts: {prompts}, Feed: {feed}, Report: {block}")
    
    ## LlamaManager APPLICATION with Session Handler
    # loop
    chat_loop()
    
