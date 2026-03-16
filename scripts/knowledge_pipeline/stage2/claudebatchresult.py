import asyncio, json, yaml
from benchbench.scripts.llm_manager import LLMManager

cfg = yaml.safe_load(open("benchbench/config.yaml"))
llm = LLMManager(cfg)

async def main():
    status = await llm.anthropic_get_batch("msgbatch_01Uwa1BHtmUYT2SwiqDcoPPq")
    print("status:", json.dumps(status, indent=2))
    if status.get("status") == "completed":
        results = await llm.anthropic_download_batch_output("msgbatch_01Uwa1BHtmUYT2SwiqDcoPPq")
        print(results[:3])  # first few items

asyncio.run(main())