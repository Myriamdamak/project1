"""Shared helpers: Gemini client, prompt loading, JSONL IO, parallelism."""
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dotenv import load_dotenv


from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")

MODEL = os.environ["MODEL"]

_client = None


def client():
    global _client
    if _client is None:
        _client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1)
            ),
        )
    return _client

def generate(prompt, temperature=0.0, json_mode=False, retries=1):
    cfg = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json" if json_mode else None,
    )
    for attempt in range(retries):
        time.sleep(15)

        try:
            resp = client().models.generate_content(model=MODEL, contents=prompt, config=cfg)
            if resp.text:
                return resp.text.strip()
        except Exception as e:  # rate limits, transient errors
            print(f"[retry {attempt + 1}] {e}")
        time.sleep(2 ** attempt)
    raise RuntimeError("Gemini call failed after retries")


def generate_json(prompt, temperature=0.0):
    text = generate(prompt, temperature=temperature, json_mode=True)
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    return json.loads(text)


def load_prompt(name, **kwargs):
    """Fill {placeholders}. Uses replace(), so braces inside code are safe."""
    text = (ROOT / "prompts" / f"{name}.txt").read_text()
    for k, v in kwargs.items():
        text = text.replace("{" + k + "}", str(v))
    return text


def scoring_guide():
    return (ROOT / "data" / "scoring_guide.md").read_text()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parallel_map(fn, items, workers=4):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))
