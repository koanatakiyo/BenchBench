import asyncio, json, yaml
from pathlib import Path
from benchbench.scripts.llm_manager import LLMManager

# Load LLM credentials/config (contains OPENAI key, etc.)
llm_config_path = Path(__file__).resolve().parents[3] / "config" / "llm.yaml"
cfg = yaml.safe_load(llm_config_path.read_text())
llm = LLMManager(cfg)

async def main():
    status = await llm.openai_get_batch("batch_6937d77098e88190afefa662a5a91490")
    print("status:", json.dumps(status, indent=2))

    ofid = status.get("output_file_id")
    if ofid:
        text = await llm.openai_download_batch_output(ofid)
        print(text[:2000])
    elif status.get("error_file_id"):
        # Optional: fetch the error file for details
        err_text = await llm.openai_download_batch_output(status["error_file_id"])
        print(err_text[:2000])

asyncio.run(main())