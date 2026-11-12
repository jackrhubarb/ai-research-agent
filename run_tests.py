import os
import csv
import json
import asyncio
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from browser_use import Agent, Browser, BrowserProfile
from cleaner import extract_clean_content

load_dotenv()

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
SUMMARY_CSV = RESULTS_DIR / "summary.csv"

# 1. LLM Engine Selector
if os.getenv("ANTHROPIC_API_KEY"):
    from browser_use import ChatAnthropic
    print("🤖 Using Anthropic Claude...")
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key=os.getenv("ANTHROPIC_API_KEY"))
elif os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
    from browser_use import ChatGoogle
    print("🤖 Using Google Gemini...")
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    # Using Gemini 3.8 Flash (fastest, most economical frontier vision model)
    llm = ChatGoogle(model="gemini-2.5-flash-lite", api_key=google_key)
else:
    raise ValueError("❌ No valid API key found in .env!")

# 2. Browser profile for cloud container
def get_browser_profile():
    return BrowserProfile(
        headless=True,
        args=[
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--use-mock-keychain",
            "--password-store=basic",
            "--disable-software-rasterizer"
        ]
    )

ROLE_TASK = (
    "You are an expert literature review professional with specialization in systematic searching "
    "and evidence mapping. Your task is to identify and list all the academic literature that directly "
    "addresses the research question provided below. Perform a comprehensive, systematic search to "
    "identify all potentially relevant academic sources, maximizing coverage (high recall) while "
    "including only those directly relevant to the following research question (high precision):\n\n"
)

OUTPUT_FORMAT = (
    "Return the results in standard BibTeX format. Use the correct BibTeX entry type (e.g., @article). "
    "Each entry must include the following fields: title, author (listing all authors in BibTeX format), "
    "year, and doi (if available). Do not fabricate any sources or information – include only genuine "
    "academic sources. Provide only the BibTeX output with no extra text."
)

def build_prompt_by_format(format_type: int, query_text: str, include_ci: bool, seed_count: int) -> str:
    ci_block = ""
    if include_ci:
        target_count = int(seed_count * 1.5)
        ci_block = (
            f"\n\nProvide at least {target_count} relevant sources in your best effort (more if available). "
            f"If in doubt about inclusion, rather include than exclude."
        )

    if format_type == 1:
        return query_text
    elif format_type == 2:
        return f"{query_text}{ci_block}\n\n{OUTPUT_FORMAT}"
    elif format_type == 3:
        return f"{ROLE_TASK}{query_text}{ci_block}\n\n{OUTPUT_FORMAT}"

    return query_text

def build_file_name(tool: dict, prompt_item: dict, outcome_type: str = "bib") -> Path:
    sys_id = tool.get("system_id", "unknown")
    func = tool.get("function", "search")
    model_ver = tool.get("model_version", "unknown")
    premium = tool.get("premium", "free")

    seed_id = prompt_item.get("seedpaper_id", "seed")
    prompt_id = prompt_item.get("prompt_id", "p0")
    prompt_ver = prompt_item.get("prompt_version", "v1")

    now = datetime.now()
    date_str = now.strftime("%y%m%d")
    time_str = now.strftime("%H%M%S")

    base = f"{sys_id}.{func}_{model_ver}_{premium}_{seed_id}_{prompt_id}_{prompt_ver}_{date_str}_{time_str}"

    if outcome_type == "bib":
        return RESULTS_DIR / f"{base}.bib"
    elif outcome_type == "0":
        return RESULTS_DIR / f"{base}_0.txt"
    else:
        return RESULTS_DIR / f"{base}_na.txt"

def harvest_bibtex_from_history(history) -> str:
    res = history.final_result()
    if res and "@" in res:
        return res

    try:
        for item in reversed(history.extracted_content()):
            if item and "@" in str(item):
                return str(item)
    except Exception:
        pass

    try:
        for step in reversed(history.history):
            if hasattr(step, "result") and step.result:
                for r in step.result:
                    if hasattr(r, "extracted_content") and r.extracted_content and "@" in r.extracted_content:
                        return r.extracted_content
            if hasattr(step, "model_output") and step.model_output:
                text = str(step.model_output)
                if "@" in text:
                    return text
    except Exception:
        pass

    return res or ""

def log_summary(tool_name: str, seed_id: str, prompt_id: str, status: str, citation_count: int, file_path: str):
    file_exists = SUMMARY_CSV.exists()
    with open(SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Tool", "Seed_ID", "Prompt_ID", "Status", "Citations_Found", "Output_File"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            tool_name,
            seed_id,
            prompt_id,
            status,
            citation_count,
            file_path
        ])

async def test_tool_prompt(tool: dict, prompt_item: dict):
    tool_name = tool["name"]
    url = tool["url"]
    notes = tool.get("notes", "")
    format_type = tool.get("format_type", 1)

    seed_id = prompt_item["seedpaper_id"]
    prompt_id = prompt_item["prompt_id"]
    query_text = prompt_item["query_text"]
    include_ci = prompt_item.get("include_ci", False)
    seed_count = prompt_item.get("seed_included_count", 20)

    print(f"\n==========================================")
    print(f"▶ Testing: {tool_name} | Seed: {seed_id} | Prompt: {prompt_id} (Format {format_type})")
    print(f"==========================================")

    final_query = build_prompt_by_format(format_type, query_text, include_ci, seed_count)

    task_prompt = f"""
    1. Navigate to: {url}
    2. Specific site notes: {notes}
    3. Dismiss any cookie banners or overlays immediately.
    4. Input the following research query into the search or chat box:
\"\"\"
{final_query}
\"\"\"
    5. Press Enter or click Submit.
    6. If a sign-up popup appears, press Escape or close it.
    7. Wait until the generation finishes or results load.
    8. Extract all returned academic papers in valid BibTeX format.
    9. Call the done action with the complete BibTeX citations.
    """

    browser = Browser(browser_profile=get_browser_profile())
    
    # max_steps=7 prevents runaway credit burn and cuts runtime in half
    agent = Agent(
        task=task_prompt,
        llm=llm,
        browser=browser,
        max_steps=7,
    )

    try:
        history = await agent.run()
        raw_output = harvest_bibtex_from_history(history)
        clean_bibtex = extract_clean_content(raw_output, "bibtex")

        final_text = clean_bibtex if clean_bibtex else raw_output
        citation_count = final_text.count("@") if final_text else 0

        if citation_count > 0:
            output_file = build_file_name(tool, prompt_item, "bib")
            status = "SUCCESS"
        else:
            output_file = build_file_name(tool, prompt_item, "0")
            status = "ZERO_RESULTS"
            final_text = "% System computed query but returned 0 valid citations."

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_text)

        print(f"✔ Finished {tool_name}: Saved to {output_file.name} ({citation_count} citations)")
        log_summary(tool_name, seed_id, prompt_id, status, citation_count, str(output_file))

    except Exception as e:
        output_file = build_file_name(tool, prompt_item, "na")
        error_reason = f"System failed to compute query. Error details:\n{e}"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(error_reason)

        print(f"✖ Non-computing on {tool_name}: Saved reason to {output_file.name}")
        log_summary(tool_name, seed_id, prompt_id, f"NA_ERROR", 0, str(output_file))
    finally:
        await browser.close()

async def main():
    with open("prompts.json", "r", encoding="utf-8") as f:
        prompts = json.load(f)

    with open("tools.json", "r", encoding="utf-8") as f:
        tools = json.load(f)

    active_tools = [t for t in tools if t.get("enabled", True)]
    print(f"🚀 Loaded {len(active_tools)} Active Tools × {len(prompts)} Benchmark Prompts")

    for tool in active_tools:
        for prompt_item in prompts:
            await test_tool_prompt(tool, prompt_item)
            await asyncio.sleep(2)

    print("\n🎉 Benchmark run completed! Check the 'results/' folder.")

if __name__ == "__main__":
    asyncio.run(main())