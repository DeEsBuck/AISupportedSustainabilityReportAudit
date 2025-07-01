import logging as log
import helper.Helpy as hp
from helper.Konfiguration import DirectoryTree
from helper.ReportExtractor import ExtractPDFReport
from toolchain.ReportLoadJSON import ImportJSONReport
from toolchain.DataPointExtractor import ExtractDataPoints
from toolchain.LlamaModel import call_lmstudio, promptRequest
from toolchain.OpenAI import chat_loop

log_name = log.getLogger('')
log.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=log.INFO)

if __name__ == "__main__":
    ## Konfiguration ensure_directories

    dir = DirectoryTree.ENUMDIR
    #DirectoryTree.ensure_directories('.', [dir[4], dir[5], dir[2], dir[1]])
    
    datapoints = ExtractDataPoints(dir[4], dir[5], dir[6], DirectoryTree.SHEET_NAME)
    datapoints.run_pipeline()
    prompts = [datapoints.get_datapoints_prompts()]
    feed = datapoints.transform_to_frame(prompts)
    
    report = ImportJSONReport(dir[0], dir[2])
    report.con_text_block(DirectoryTree.SHEET_NAME)
    
    ## LlamaManager APPLICATION with Session Handler
    call_lmstudio()
    promptRequest()
    # loop
    chat_loop()
    
