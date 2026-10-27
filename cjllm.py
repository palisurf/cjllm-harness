"""
cjllm.py — CJ vLLM Search & Inference Harness (cjllm-search-harness)

Author: Christopher Jones (@palisurf)
Repository: https://github.com/palisurf/cjllm-harness

A single-file, model-agnostic vLLM inference and search-augmented generation harness
featuring automatic reasoning/tokenizer profile detection, multi-tier live web/quote/weather/table
extraction, rolling summary-buffer memory compression, and live TTFT/TPS telemetry.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import html
import json
import os
import queue
import re
import readline
import threading
import time
import warnings
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo
from openai import OpenAI
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

# Suppress deprecation warnings from legacy package imports
warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")

# Prioritize native `ddgs` import over legacy `duckduckgo_search`
try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

try:
    import primp
except ImportError:
    primp = None

console = Console()

# --- CLI ARGUMENTS (All Optional, Defaults to Local vLLM Port 8000) ---
parser = argparse.ArgumentParser(
    prog="cjllm.py",
    description="cjllm — CJ vLLM Search & Inference Harness (@palisurf | https://github.com/palisurf/cjllm-search-harness)",
)
parser.add_argument(
    "--url",
    default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
    help="OpenAI-compatible base URL (default: http://localhost:8000/v1)",
)
parser.add_argument(
    "--api-key",
    default=os.environ.get("VLLM_API_KEY", "none"),
    help="API key for endpoint (default: none)",
)
parser.add_argument(
    "--model",
    default=os.environ.get("VLLM_MODEL", None),
    help="Model ID override (default: auto-detected from /v1/models)",
)
parser.add_argument(
    "--max-pairs",
    type=int,
    default=30,
    help="High-water mark of verbatim user/assistant turn pairs before batch compression (default: 30)",
)
parser.add_argument(
    "--max-verbatim-tokens",
    type=int,
    default=20000,
    help="High-water mark of approximate verbatim history tokens before batch compression (default: 20000)",
)
parser.add_argument(
    "--evict-batch",
    type=int,
    default=8,
    help="Number of oldest turn pairs to fold in a single batch when high-water mark is reached (default: 8)",
)
parser.add_argument(
    "--cache",
    default="chat_cache.json",
    help="Default session cache file (default: chat_cache.json)",
)
parser.add_argument(
    "--think",
    action="store_true",
    help="Start session with Chain-of-Thought reasoning enabled",
)
parser.add_argument(
    "--tz",
    default=os.environ.get("CJLLM_TZ", None),
    help="IANA timezone override for local clock (default: OS local timezone, or America/Los_Angeles if OS is Etc/UTC)",
)
args, _ = parser.parse_known_args()


def resolve_local_timezone(tz_override: str | None = None):
    """
    Resolves the user's local timezone. Many headless Linux servers and Docker containers
    default their OS clock to `Etc/UTC`, which shifts the calendar date forward by 1 day
    every evening after 5:00 PM Pacific Time. When the OS timezone is generic UTC/GMT and no
    explicit override is given, defaults to `America/Los_Angeles` (Pacific Time).
    """
    if tz_override:
        try:
            return ZoneInfo(tz_override)
        except Exception:
            pass
    try:
        sys_now = datetime.now().astimezone()
        tz_name = (sys_now.tzname() or "").upper()
        if tz_name not in ("UTC", "ETC/UTC", "GMT", "UCT", ""):
            return sys_now.tzinfo
    except Exception:
        pass
    try:
        return ZoneInfo("America/Los_Angeles")
    except Exception:
        return timezone(timedelta(hours=-7), name="PDT")


LOCAL_TZ = resolve_local_timezone(args.tz)

# Initialize OpenAI client pointing to local inference engine
client = OpenAI(
    base_url=args.url,
    api_key=args.api_key,
)

DEFAULT_CACHE_FILE = args.cache
COMMAND_HISTORY_FILE = ".chat_prompt_history"
MAX_PAIRS = args.max_pairs
MAX_VERBATIM_TOKENS = args.max_verbatim_tokens
EVICT_BATCH_PAIRS = max(1, min(args.evict_batch, max(1, MAX_PAIRS // 2)))

# --- DEEPSEEK-R1-DISTILL / QWEN3 TOKENIZER BRIDGE CONSTANTS ---
# When a container serves DeepSeek-R1-Distill-Qwen-32B weights under the Qwen/Qwen3-32B tokenizer:
#   Token ID 151643 = <｜end▁of▁sentence｜> (mapped to "<|endoftext|>" in Qwen3 tokenizer)
#   Token ID 151648 = <think>               (mapped to "<|box_start|>" in Qwen3 tokenizer)
#   Token ID 151649 = </think>              (mapped to "<|box_end|>" in Qwen3 tokenizer)
DS_EOS_TOKEN_ID = 151643
DS_THINK_START_TOKEN_ID = 151648
DS_THINK_END_TOKEN_ID = 151649

DS_THINK_OPEN_STR = "<|box_start|>\n"
DS_THINK_CLOSE_STR = "<|box_end|>"
DS_NO_THINK_PREFILL = "<|box_start|>\nI have reviewed the context and will now provide a direct, accurate response.\n<|box_end|>\n\n"
DS_UTILITY_PREFILL = "<|box_start|>\nI have evaluated the input and will output only the required result.\n<|box_end|>\n\n"


def detect_model_profile(openai_client: OpenAI, model_override: str | None = None) -> dict:
    """
    Auto-discovers the active model from `/v1/models` and probes its tokenizer/reasoning behavior.
    Returns a profile dict supporting three automatic modes:
      1. 'deepseek_qwen3_bridge' : DeepSeek-R1-Distill weights running under Qwen/Qwen3 tokenizer (<|box_start|> / <|box_end|>)
      2. 'native_think'          : Native DeepSeek-R1 / Qwen3 / QwQ tokenizer (<think> / </think>)
      3. 'standard_instruct'     : Standard instruct models (Llama-3.1/3.3 8B/70B, Mistral, etc.)
    """
    model_id = model_override or "Qwen/Qwen3-32B"
    max_model_len = 32768

    try:
        models_resp = openai_client.models.list()
        if models_resp.data:
            first_model = models_resp.data[0]
            if not model_override:
                model_id = first_model.id
            raw_max_len = getattr(first_model, "max_model_len", None)
            if isinstance(raw_max_len, int) and raw_max_len > 0:
                max_model_len = raw_max_len
    except Exception:
        pass

    max_gen_tokens = min(64000, max(2048, max_model_len // 2))
    model_lower = model_id.lower()

    # Fast path for known non-reasoning instruct families (Llama, Mistral, Gemma, Phi)
    if any(fam in model_lower for fam in ("llama", "mistral", "mixtral", "gemma", "phi-")):
        return {
            "model_id": model_id,
            "mode": "standard_instruct",
            "description": "Standard Instruct (Native Chat Template)",
            "max_model_len": max_model_len,
            "max_gen_tokens": max_gen_tokens,
            "think_open": "<think>\n",
            "think_close": "</think>",
            "no_think_prefill": None,
            "no_think_stop_ids": None,
            "think_stop_ids": None,
        }

    # Probe 1: Check if model closes <|box_start|> with <|box_end|> (DeepSeek-R1-Distill on Qwen3 tokenizer)
    try:
        probe_resp = openai_client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": "Say OK."},
                {"role": "assistant", "content": "<|box_start|>\nThe user wants me to say OK.\n"},
            ],
            extra_body={
                "continue_final_message": True,
                "add_generation_prompt": False,
                "skip_special_tokens": False,
            },
            temperature=0.0,
            max_tokens=3,
        )
        probe_text = probe_resp.choices[0].message.content or ""
        if "<|box_end|>" in probe_text:
            return {
                "model_id": model_id,
                "mode": "deepseek_qwen3_bridge",
                "description": "DeepSeek-R1-Distill on Qwen3 Tokenizer Bridge (<|box_start|> / <|box_end|>)",
                "max_model_len": max_model_len,
                "max_gen_tokens": max_gen_tokens,
                "think_open": DS_THINK_OPEN_STR,
                "think_close": DS_THINK_CLOSE_STR,
                "no_think_prefill": DS_NO_THINK_PREFILL,
                "utility_prefill": DS_UTILITY_PREFILL,
                "no_think_stop_ids": [DS_THINK_END_TOKEN_ID, DS_EOS_TOKEN_ID],
                "think_stop_ids": [DS_EOS_TOKEN_ID],
            }
    except Exception:
        pass

    # Probe 2: Check if model closes <think> with </think> (Native DeepSeek-R1 / Qwen3 / QwQ)
    try:
        probe_resp = openai_client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": "Say OK."},
                {"role": "assistant", "content": "<think>\nThe user wants me to say OK.\n"},
            ],
            extra_body={
                "continue_final_message": True,
                "add_generation_prompt": False,
                "skip_special_tokens": False,
            },
            temperature=0.0,
            max_tokens=3,
        )
        probe_text = probe_resp.choices[0].message.content or ""
        if "</think>" in probe_text:
            return {
                "model_id": model_id,
                "mode": "native_think",
                "description": "Native Reasoning Tokenizer (<think> / </think>)",
                "max_model_len": max_model_len,
                "max_gen_tokens": max_gen_tokens,
                "think_open": "<think>\n",
                "think_close": "</think>",
                "no_think_prefill": "<think>\nI have reviewed the context and will now provide a direct, accurate response.\n</think>\n\n",
                "utility_prefill": "<think>\nI have evaluated the input and will output only the required result.\n</think>\n\n",
                "no_think_stop_ids": None,
                "think_stop_ids": None,
            }
    except Exception:
        pass

    # Fallback: Standard Instruct Mode
    return {
        "model_id": model_id,
        "mode": "standard_instruct",
        "description": "Standard Instruct (Native Chat Template)",
        "max_model_len": max_model_len,
        "max_gen_tokens": max_gen_tokens,
        "think_open": "<think>\n",
        "think_close": "</think>",
        "no_think_prefill": None,
        "utility_prefill": None,
        "no_think_stop_ids": None,
        "think_stop_ids": None,
    }


MODEL_PROFILE = detect_model_profile(client, args.model)
MODEL_NAME = MODEL_PROFILE["model_id"]


def prepare_request_messages_and_extra(
    messages: list[dict],
    use_thinking: bool = False,
    is_utility_call: bool = False,
) -> tuple[list[dict], dict]:
    """
    Adapts the message list and vLLM `extra_body` parameters to the active MODEL_PROFILE:
      - In 'deepseek_qwen3_bridge' or 'native_think' mode: injects the appropriate assistant prefill
        and stop token IDs so non-thinking calls skip CoT and thinking calls stream CoT cleanly.
      - In 'standard_instruct' mode (e.g., Llama-3.1-70B/8B): sends standard messages with no prefill,
        or adds a `<think>...</think>` reasoning instruction if the user explicitly requested `/think`.
    """
    req_messages = [dict(m) for m in messages]
    mode = MODEL_PROFILE["mode"]

    if mode in ("deepseek_qwen3_bridge", "native_think"):
        if use_thinking and not is_utility_call:
            req_messages.append({"role": "assistant", "content": MODEL_PROFILE["think_open"]})
            extra = {
                "repetition_penalty": 1.08,
                "continue_final_message": True,
                "add_generation_prompt": False,
                "skip_special_tokens": False,
            }
            if MODEL_PROFILE["think_stop_ids"]:
                extra["stop_token_ids"] = MODEL_PROFILE["think_stop_ids"]
        else:
            prefill = MODEL_PROFILE.get("utility_prefill") if is_utility_call else MODEL_PROFILE["no_think_prefill"]
            req_messages.append({"role": "assistant", "content": prefill})
            extra = {
                "continue_final_message": True,
                "add_generation_prompt": False,
            }
            if not is_utility_call:
                extra["repetition_penalty"] = 1.08
            if MODEL_PROFILE["no_think_stop_ids"]:
                extra["stop_token_ids"] = MODEL_PROFILE["no_think_stop_ids"]
        return req_messages, extra

    # Standard Instruct Mode (Llama-3.1 8B/70B, Mistral, etc.)
    extra = {} if is_utility_call else {"repetition_penalty": 1.05}
    if use_thinking and not is_utility_call and req_messages:
        req_messages[-1] = {
            "role": req_messages[-1]["role"],
            "content": (
                req_messages[-1]["content"]
                + "\n\n(First, think step-by-step inside `<think>...</think>` tags, then provide your final answer after `</think>`.)"
            ),
        }
    return req_messages, extra

# Common web boilerplate patterns to filter out
BOILERPLATE_PATTERNS = [
    r"cookie policy",
    r"privacy policy",
    r"enable javascript",
    r"javascript is disabled",
    r"all rights reserved",
    r"subscribe to our newsletter",
    r"skip to main content",
    r"accept all cookies",
    r"terms of use",
]

# --- Terminal Arrow Key History Setup ---
readline.set_history_length(1000)

if os.path.exists(COMMAND_HISTORY_FILE):
    try:
        readline.read_history_file(COMMAND_HISTORY_FILE)
    except Exception:
        pass


def strip_thinking_and_dedup(text: str) -> str:
    """
    Cleans raw model output by:
    1. Stripping any pre-</think> (or <|box_end|>) reasoning blocks.
    2. Removing residual special token strings (<|endoftext|>, <|box_start|>, etc.).
    3. Deduplicating back-to-back repeated answers separated by triple newlines.
    """
    if not text:
        return ""

    # If the model emitted a thinking block closed by <|box_end|> (151649) or </think>, keep only the final answer
    if DS_THINK_CLOSE_STR in text:
        text = text.split(DS_THINK_CLOSE_STR)[-1]
    if "</think>" in text:
        text = text.split("</think>")[-1]

    # Remove any residual special token strings
    for tok in ("<|box_start|>", "<|box_end|>", "<|endoftext|>", "<think>", "</think>"):
        text = text.replace(tok, "")

    text = text.strip()

    # Guard against back-to-back duplicate paragraphs separated by \n\n\n
    if "\n\n\n" in text:
        parts = [p.strip() for p in text.split("\n\n\n") if p.strip()]
        if len(parts) == 2 and (parts[0] == parts[1] or parts[0][:80] == parts[1][:80]):
            text = parts[-1]

    return text


def format_thinking_markdown(raw_text: str) -> str:
    """Formats live streaming text when thinking mode is enabled, visually separating CoT from the final answer."""
    cleaned = raw_text.replace("<|endoftext|>", "").replace("<|box_start|>", "").replace("<think>", "")
    close_delim = MODEL_PROFILE.get("think_close", "</think>")
    if close_delim in cleaned:
        think_part, answer_part = cleaned.split(close_delim, 1)
    elif "</think>" in cleaned:
        think_part, answer_part = cleaned.split("</think>", 1)
    else:
        think_part = cleaned.strip()
        if not think_part:
            return "*Thinking...*"
        quoted_think = "\n".join(f"> *{line}*" if line.strip() else ">" for line in think_part.splitlines())
        return f"**Thinking Process:**\n{quoted_think}"

    think_part = think_part.strip()
    answer_part = answer_part.strip()
    if think_part:
        quoted_think = "\n".join(f"> *{line}*" if line.strip() else ">" for line in think_part.splitlines())
        return f"**Thinking Process:**\n{quoted_think}\n\n---\n\n{answer_part}"
    return answer_part


def sync_readline_history(history: list) -> None:
    """Syncs loaded conversation turns into terminal Up/Down arrow history."""
    for msg in history:
        if msg.get("role") == "user" and msg.get("content"):
            content = msg["content"]
            if "User Question:" in content:
                content = content.split("User Question:")[-1].strip()
            readline.add_history(content)


def save_cmd_history() -> None:
    """Saves terminal up/down arrow prompt history to disk."""
    try:
        readline.write_history_file(COMMAND_HISTORY_FILE)
    except Exception:
        pass


def de_hydrate_cache(history: list, summary: str, filepath: str, silent: bool = False) -> None:
    """De-hydrates (saves) conversation history and summary buffer to a JSON file."""
    try:
        data = {
            "history": history,
            "summary": summary,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        if not silent:
            console.print(f"[bold green][Cache][/bold green] Saved context to [yellow]'{filepath}'[/yellow].")
    except Exception as e:
        if not silent:
            console.print(f"[bold red][Cache Error][/bold red] Failed to save context: {e}")


def re_hydrate_cache(filepath: str) -> tuple[list | None, str]:
    """Re-hydrates (loads) conversation history and summary buffer from a JSON file."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                history = data.get("history", [])
                summary = data.get("summary", "")
            else:
                history = data
                summary = ""

            # Sanitize any legacy duplicated responses in loaded history
            for msg in history:
                if msg.get("role") == "assistant" and msg.get("content"):
                    msg["content"] = strip_thinking_and_dedup(msg["content"])

            console.print(f"[bold green][Cache][/bold green] Loaded context from [yellow]'{filepath}'[/yellow].")
            sync_readline_history(history)
            return history, summary
        except Exception as e:
            console.print(f"[bold red][Cache Error][/bold red] Failed to load context: {e}")
    else:
        console.print(f"[bold yellow][Cache][/bold yellow] File [yellow]'{filepath}'[/yellow] not found.")
    return None, ""


def clean_text(text: str) -> str:
    """Decodes HTML entities and normalizes whitespace."""
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_junk_snippet(text: str) -> bool:
    """Filters out low-value boilerplate snippets and junk text."""
    text_lower = text.lower()
    if len(text.strip()) < 30:
        return True
    return any(re.search(pattern, text_lower) for pattern in BOILERPLATE_PATTERNS)


def _extract_yahoo_stock_card(html_text: str, label: str = "") -> list[str]:
    """Helper to parse Yahoo Search compStockChart blocks from raw HTML."""
    out = []
    prefix = f"[Live Market Quote Card — {label}]" if label else "[Live Market Quote Card]"
    for m in re.finditer(r'<div[^>]*class="[^"]*compStockChart[^"]*"', html_text):
        idx = m.start()
        chunk = html_text[max(0, idx - 600) : idx + 9000]
        chunk = re.sub(r"^[^<]*>", " ", chunk)
        chunk = re.sub(r"<[^>]*$", " ", chunk)
        chunk = re.sub(r"<style[^>]*>.*?</style>", " ", chunk, flags=re.S)
        chunk = re.sub(r"<script[^>]*>.*?</script>", " ", chunk, flags=re.S)
        chunk = re.sub(r"<svg[^>]*>.*?</svg>", " ", chunk, flags=re.S)
        chunk = re.sub(r'<div class="fin-watchlist.*?</div></div></div>', " ", chunk, flags=re.S)
        chunk = re.sub(r'<ul role="tabList".*?</ul>', " ", chunk, flags=re.S)
        text = clean_text(re.sub(r"<[^>]+>", " ", chunk))
        for stop_marker in ["People also search", "Explore AI results", "More about "]:
            if stop_marker in text:
                text = text.split(stop_marker)[0].strip()
        text = text.replace("Full chart", "| Key Stats:")
        if len(text) > 30:
            out.append(f"{prefix} {text[:600]}")
    return out


def _extract_yahoo_weather_card(html_text: str, label: str = "") -> list[str]:
    """Helper to parse Yahoo Search compWeatherInfo / weatherCarousel blocks from raw HTML."""
    weather_parts = []
    prefix = f"[Live Weather & Multi-Day Forecast Card — {label}]" if label else "[Live Weather & Multi-Day Forecast Card]"
    for m in re.finditer(r'class="[^"]*compWeatherInfo[^"]*"', html_text):
        idx = m.start()
        chunk = html_text[idx : idx + 35000]
        chunk = re.sub(r"data-forecast='[^']*'", "", chunk)
        chunk = re.sub(r'data-forecast="[^"]*"', "", chunk)
        chunk = re.sub(r"data-yga='[^']*'", "", chunk)
        chunk = re.sub(r"<svg[^>]*>.*?<title[^>]*>(.*?)</title>.*?</svg>", r" (\1) ", chunk, flags=re.S)
        chunk = re.sub(r"<svg[^>]*>.*?</svg>", " ", chunk, flags=re.S)
        chunk = re.sub(r"<style[^>]*>.*?</style>", " ", chunk, flags=re.S)
        chunk = re.sub(r"<script[^>]*>.*?</script>", " ", chunk, flags=re.S)
        chunk = re.sub(r"^[^<]*>", " ", chunk)
        chunk = re.sub(r"<[^>]*$", " ", chunk)
        w_txt = clean_text(re.sub(r"<[^>]+>", " ", chunk))
        for stop_marker in ["Temperature Precipitation Wind", "People also search", "Explore AI results"]:
            if stop_marker in w_txt:
                w_txt = w_txt.split(stop_marker)[0].strip()
        if len(w_txt) > 20:
            weather_parts.append(w_txt[:500])
    if weather_parts:
        return [f"{prefix} " + " | Forecast: ".join(weather_parts[:2])]
    return []


_TICKER_STOPWORDS = {
    "THE", "AND", "FOR", "WITH", "FROM", "THAT", "THIS", "WHAT", "WHEN", "WHERE", "WHICH",
    "HOW", "WHY", "WHO", "ARE", "WAS", "WERE", "TODAY", "NOW", "LIVE", "DATA", "PRICE",
    "STOCK", "QUOTE", "INDEX", "MARKET", "CAP", "PE", "EPS", "ETF", "IPO", "CEO", "CFO",
    "GDP", "CPI", "FED", "SEC", "NYSE", "NASDAQ", "DOW", "DJIA", "SP", "USD", "USA", "US",
    "UK", "EU", "NOAA", "NDBC", "UTC", "GMT", "PDT", "PST", "EDT", "EST", "CDT", "CST",
    "VS", "OR", "IN", "ON", "AT", "TO", "OF", "BY", "ALL", "TOP", "NEW", "HIGH", "LOW",
    "CA", "HI", "NY", "TX", "FL", "WA", "OR", "AZ", "NV", "CO", "MA", "IL", "PA", "OH",
}

_KNOWN_COMPANY_TICKERS = {
    "nvidia": ("NVDA", "NVDA stock quote"),
    "apple": ("AAPL", "AAPL stock quote"),
    "microsoft": ("MSFT", "MSFT stock quote"),
    "amd": ("AMD", "AMD stock quote"),
    "broadcom": ("AVGO", "AVGO stock quote"),
    "alphabet": ("GOOGL", "GOOGL stock quote"),
    "google": ("GOOGL", "GOOGL stock quote"),
    "amazon": ("AMZN", "AMZN stock quote"),
    "meta": ("META", "META stock quote"),
    "tesla": ("TSLA", "TSLA stock quote"),
    "intel": ("INTC", "INTC stock quote"),
    "tsmc": ("TSM", "TSM stock quote"),
    "micron": ("MU", "MU stock quote"),
    "palantir": ("PLTR", "PLTR stock quote"),
    "generac": ("GNRC", "GNRC stock quote"),
}


def fetch_yahoo_rich_cards(query: str) -> list[str]:
    """
    Extracts rich top-of-page Yahoo Search cards (live stock/index/crypto/commodity quotes,
    multi-city weather forecasts, and structured fact tables) that DDGS().text() skips.
    Automatically expands:
      - Multi-index / macro queries (Dow, S&P 500, Nasdaq, Russell 2000, Bitcoin, Gold, 10Y Treasury)
      - Multi-ticker stock comparisons (e.g., 'NVDA AAPL AMD stock price today')
      - Multi-location weather comparisons (e.g., 'weather Malibu CA vs Honolulu HI')
    """
    cards = []
    if primp is None:
        return cards

    try:
        http_client = primp.Client(impersonate="random", timeout=5)
        q_lower = query.lower()

        # 1. Check for multi-index, macro-asset, or multi-ticker financial queries
        finance_targets: list[tuple[str, str]] = []
        seen_labels: set[str] = set()

        def _add_fin_target(lbl: str, q_str: str):
            if lbl not in seen_labels:
                seen_labels.add(lbl)
                finance_targets.append((lbl, q_str))

        if re.search(r"\b(dow|djia|dji)\b", q_lower):
            _add_fin_target("Dow Jones (^DJI)", "Dow Jones Industrial Average quote")
        if re.search(r"\b(s&p|sp\s*500|s\s*p\s*500|\bsp\b|gspc)\b", q_lower):
            _add_fin_target("S&P 500 (^GSPC)", "S&P 500 index quote")
        if re.search(r"\b(nasdaq|ixic)\b", q_lower):
            _add_fin_target("Nasdaq Composite (^IXIC)", "Nasdaq Composite quote")
        if re.search(r"\b(russell\s*2000|rut)\b", q_lower):
            _add_fin_target("Russell 2000 (^RUT)", "Russell 2000 index quote")
        if re.search(r"\b(bitcoin|btc)\b", q_lower):
            _add_fin_target("Bitcoin (BTC-USD)", "BTC-USD price quote")
        if re.search(r"\b(ethereum|eth)\b", q_lower):
            _add_fin_target("Ethereum (ETH-USD)", "ETH-USD price quote")
        if re.search(r"\b(gold\s+price|gold\s+futures|\bgold\b)\b", q_lower) and any(
            k in q_lower for k in ("price", "quote", "move", "today", "market", "bitcoin", "treasury", "yield", "oil", "silver")
        ):
            _add_fin_target("Gold (GC=F)", "Gold futures GC=F quote")
        if re.search(r"\b(10[-\s]*year\s+treasury|10y\s+yield|treasury\s+yield|tnx)\b", q_lower):
            _add_fin_target("10-Year Treasury Yield (^TNX)", "^TNX 10 year treasury yield quote")

        # Detect explicit uppercase stock tickers (e.g., NVDA, AAPL, AMD) or known company names
        has_fin_context = any(
            w in q_lower
            for w in ("stock", "price", "quote", "share", "market", "cap", "pe", "ticker", "compare", "today", "earnings", "close")
        )
        for comp_name, (tk, tk_q) in _KNOWN_COMPANY_TICKERS.items():
            if re.search(rf"\b{comp_name}\b", q_lower):
                _add_fin_target(tk, tk_q)

        if has_fin_context:
            for tk in re.findall(r"\b([A-Z]{2,5})\b", query):
                if tk not in _TICKER_STOPWORDS:
                    _add_fin_target(tk, f"{tk} stock quote")

        if len(finance_targets) >= 2:
            for idx_label, idx_query in finance_targets[:5]:
                try:
                    r_idx = http_client.get(f"https://search.yahoo.com/search?p={quote_plus(idx_query)}")
                    if r_idx.status_code == 200 and r_idx.text:
                        cards.extend(_extract_yahoo_stock_card(r_idx.text, label=idx_label))
                except Exception:
                    pass

        # 2. Check for multi-location weather comparison queries (e.g. "weather Malibu CA vs Honolulu HI")
        weather_cards: list[str] = []
        if any(w in q_lower for w in ("weather", "forecast", "temperature")):
            loc_cleaned = re.sub(
                r"\b(compare|current|weather|forecast|temperature|in|for|between|today|right now|3-day|5-day|7-day|what|is|the)\b",
                " ",
                query,
                flags=re.I,
            )
            loc_parts = [
                p.strip(" ,.;:?!-")
                or ""
                for p in re.split(r"\b(?:vs\.?|versus|and|compared\s+to|or)\b|;", loc_cleaned, flags=re.I)
            ]
            loc_parts = [re.sub(r"\s+", " ", p).strip() for p in loc_parts if len(p.strip()) >= 3]
            if len(loc_parts) >= 2:
                for loc in loc_parts[:3]:
                    try:
                        r_w = http_client.get(f"https://search.yahoo.com/search?p={quote_plus('weather ' + loc)}")
                        if r_w.status_code == 200 and r_w.text:
                            weather_cards.extend(_extract_yahoo_weather_card(r_w.text, label=loc))
                    except Exception:
                        pass

        url = f"https://search.yahoo.com/search?p={quote_plus(query)}"
        resp = http_client.get(url)
        if resp.status_code != 200 or not resp.text:
            cards.extend(weather_cards)
            return cards
        html_text = resp.text

        # Single-target fallback for Finance Quote Card
        if not cards:
            cards.extend(_extract_yahoo_stock_card(html_text))

        # Single-target fallback for Weather Card
        if weather_cards:
            cards.extend(weather_cards)
        else:
            cards.extend(_extract_yahoo_weather_card(html_text))

        # 3. Structured Result Fact Tables (grp-talgo-facts)
        fact_blocks = []
        for m in re.finditer(r'<div class="grp grp-talgo-facts[^"]*">(.*?)</div>', html_text, flags=re.S):
            ft = clean_text(re.sub(r"<[^>]+>", " ", m.group(1)))
            if ft and ft not in fact_blocks:
                fact_blocks.append(ft)
        if fact_blocks:
            cards.append("[Structured Financial/Entity Facts] " + " ; ".join(fact_blocks[:3]))
    except Exception:
        pass

    return cards


def fetch_top_page_table_or_excerpt(text_results: list[dict], query: str = "", max_chars: int = 3800) -> str:
    """
    Two-pass structured webpage extractor using `primp`:
      - Pass 1 (Tabular Priority): Scans up to 4 candidate URLs (including auto-resolved
        5-digit NOAA NDBC buoy station pages) for structured HTML `<table>` data across
        up to 6 tables per page so a non-tabular rank-1 article never shadows a real table at rank 2-4.
      - Pass 2 (Plain-Text Excerpt Fallback): If no candidate URL contains a valid HTML `<table>`,
        extracts a cleaned article excerpt from the top reachable candidate.
    """
    if primp is None or (not text_results and not query):
        return ""

    preferred_domains = (
        "ndbc.noaa.gov",
        "weather.gov",
        "swellinfo.com",
        "deepswell.com",
        "surf-forecast.com",
        "tideschart.com",
        "tidetime.org",
        "usharbors.com",
        "thetidechart.com",
        "companiesmarketcap.com",
        "stockanalysis.com",
        "wikipedia.org",
        "slickcharts.com",
        "finance.yahoo.com",
        "forbes.com",
        "statranker.org",
    )
    candidate_urls = []

    # Automatically resolve NOAA NDBC 5-digit buoy station IDs from query or search results (e.g., 46221, 46054)
    all_meta_sources = [query] + [
        f"{res.get('title', '')} {res.get('href', '')} {res.get('body', '')}" for res in (text_results or [])
    ]
    combined_all = " ".join(all_meta_sources)
    if any(k in combined_all.lower() for k in ("buoy", "ndbc", "noaa", "swell", "station", "wave")):
        for station_match in re.findall(r"\b(46\d{3}|41\d{3}|42\d{3}|44\d{3}|51\d{3})\b", combined_all):
            ndbc_url = f"https://www.ndbc.noaa.gov/station_page.php?station={station_match}"
            if ndbc_url not in candidate_urls:
                candidate_urls.append(ndbc_url)

    for res in text_results or []:
        href = res.get("href", "")
        if href and any(d in href for d in preferred_domains) and href not in candidate_urls:
            candidate_urls.append(href)
    for res in text_results or []:
        href = res.get("href", "")
        if href and href not in candidate_urls:
            candidate_urls.append(href)

    http_client = primp.Client(impersonate="random", timeout=5)
    extracted_blocks = []
    fallback_excerpt = ""

    for url in candidate_urls[:4]:
        try:
            resp = http_client.get(url)
            if resp.status_code != 200 or not resp.text:
                continue
            raw = resp.text
            for tag in ("script", "style", "svg", "noscript", "header", "footer", "nav", "sup"):
                raw = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", " ", raw, flags=re.S | re.I)
            # Strip MediaWiki data-mw JSON attributes (which can contain '>' characters that break tag stripping)
            raw = re.sub(r"\sdata-mw='[^']*'", "", raw)
            raw = re.sub(r'\sdata-mw="[^"]*"', "", raw)

            # Pass 1: Extract structured HTML <table> rows across up to 6 tables per page
            tables = re.findall(r"<table([^>]*)>(.*?)</table>", raw, flags=re.S | re.I)
            if tables:
                rows_out = []
                for tbl_attrs, tbl in tables[:8]:
                    if re.search(r"\b(navbox|ambox|mbox|metadata|sistersitebox|toc)\b", tbl_attrs, flags=re.I):
                        continue
                    tbl_rows_list = []
                    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", tbl, flags=re.S | re.I):
                        cells = [
                            clean_text(re.sub(r"<[^>]+>", " ", cell))
                            for cell in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, flags=re.S | re.I)
                        ]
                        cells = [c for c in cells if c and not c.startswith("#") and "DCDCDC" not in c]
                        if len(cells) >= 2:
                            row_line = " | ".join(cells)
                            if any(
                                skip_phrase in row_line
                                for skip_phrase in ("Unit of Measure:", "Change units", "Click on the graph")
                            ):
                                continue
                            tbl_rows_list.append(row_line)
                        if len(tbl_rows_list) >= 120:
                            break
                    # Keep both head (top rankings) and tail (latest entries in chronological tables like US Presidents)
                    if len(tbl_rows_list) > 20:
                        selected_rows = (
                            tbl_rows_list[:8]
                            + ["... [Earlier/middle rows omitted] ..."]
                            + tbl_rows_list[-12:]
                        )
                    else:
                        selected_rows = tbl_rows_list[:20]
                    rows_out.extend(selected_rows)
                    # Stop once we have captured a substantial primary data/chronological table on this page
                    if len(tbl_rows_list) >= 15 or len(rows_out) >= 45:
                        break
                if len(rows_out) >= 3:
                    char_limit = max_chars // 2 if extracted_blocks else max_chars
                    full_table_text = "\n".join(rows_out)
                    if len(full_table_text) > char_limit:
                        half = (char_limit - 40) // 2
                        table_text = (
                            full_table_text[:half]
                            + "\n... [Middle table content omitted] ...\n"
                            + full_table_text[-half:]
                        )
                    else:
                        table_text = full_table_text
                    extracted_blocks.append(f"[Live Extracted Table from {url}]\n{table_text}")
                    if len(extracted_blocks) >= 2:
                        break

            # Record plain-text fallback from the first valid page in case no URL has a <table>
            if not fallback_excerpt:
                body_text = clean_text(re.sub(r"<[^>]+>", " ", raw))
                if len(body_text) > 250 and not is_junk_snippet(body_text[:400]):
                    fallback_excerpt = f"[Live Webpage Excerpt from {url}]\n{body_text[:1500]}"
        except Exception:
            continue

    if extracted_blocks:
        return "\n\n".join(extracted_blocks)[:max_chars]
    return fallback_excerpt[:max_chars]


def get_dual_clock_str() -> tuple[str, str, str]:
    """
    Returns (today_date_str, local_clock_str, utc_clock_str) using the resolved LOCAL_TZ
    (e.g. America/Los_Angeles PDT/PST even on headless Etc/UTC Linux servers) with both
    12-hour and 24-hour representations.
    """
    now_local = datetime.now(LOCAL_TZ)
    now_utc = datetime.now(timezone.utc)
    today_str = now_local.strftime("%A, %B %d, %Y")
    local_clock_str = now_local.strftime("%A, %B %d, %Y %I:%M:%S %p (%H:%M:%S) %Z")
    utc_clock_str = now_utc.strftime("%A, %B %d, %Y %I:%M:%S %p (%H:%M:%S) UTC/GMT")
    return today_str, local_clock_str, utc_clock_str


WORLD_CLOCK_ZONES: list[tuple[str, str, tuple[str, ...]]] = [
    ("Pacific Time (Los Angeles / Malibu)", "America/Los_Angeles", ("pacific", "pdt", "pst", "los angeles", "malibu", "california", "san francisco", "seattle")),
    ("Mountain Time (Denver / Phoenix)", "America/Denver", ("mountain", "mdt", "mst", "denver", "utah", "colorado")),
    ("Central Time (Chicago / Dallas)", "America/Chicago", ("central", "cdt", "cst", "chicago", "dallas", "houston", "texas")),
    ("Eastern Time (New York / Miami)", "America/New_York", ("eastern", "edt", "est", "new york", "nyc", "boston", "miami", "washington")),
    ("Hawaii Time (Honolulu)", "Pacific/Honolulu", ("hawaii", "hst", "honolulu", "maui", "lahaina")),
    ("UTC / GMT", "UTC", ("utc", "gmt", "zulu", "universal")),
    ("London (UK)", "Europe/London", ("london", "uk", "britain", "bst")),
    ("Central Europe (Paris / Berlin / Zurich)", "Europe/Paris", ("paris", "berlin", "cet", "cest", "zurich", "madrid", "rome", "europe")),
    ("Tokyo (Japan)", "Asia/Tokyo", ("tokyo", "japan", "jst")),
    ("Seoul (South Korea)", "Asia/Seoul", ("seoul", "korea", "kst")),
    ("Taipei / Hong Kong / Singapore", "Asia/Taipei", ("taipei", "taiwan", "hong kong", "singapore", "beijing", "shanghai")),
    ("India (IST)", "Asia/Kolkata", ("india", "ist", "mumbai", "bangalore", "delhi")),
    ("Sydney (Australia)", "Australia/Sydney", ("sydney", "australia", "melbourne", "aest", "aedt")),
]


def is_clock_skill_query(user_input: str) -> bool:
    """
    Detects queries asking for the current time, current date, day of the week,
    calendar offsets ('what day is tomorrow/yesterday'), or world timezone conversions.
    """
    u_lower = user_input.lower().strip()
    clock_patterns = (
        r"\b(what\s+time\s+is\s+it|what'?s\s+the\s+(?:current\s+)?time|current\s+time|time\s+is\s+it\s+now|time\s+right\s+now|time\s+now)\b",
        r"\b(what\s+is\s+the\s+(?:current\s+)?(?:date|time)|current\s+date(?:\s*/\s*time|\s+and\s+time)?|today'?s\s+date|what\s+date\s+is\s+(?:it|today))\b",
        r"\b(what\s+day\s+(?:of\s+the\s+week\s+)?is\s+(?:it|today|tomorrow|yesterday))\b",
        r"\b(what\s+time\s+is\s+it\s+in\s+[a-z\s]+)\b",
        r"\b(current\s+time\s+in\s+[a-z\s]+)\b",
        r"\b(tell\s+me\s+the\s+(?:current\s+)?(?:time|date))\b",
    )
    return any(re.search(pat, u_lower) for pat in clock_patterns)


def is_external_data_query(user_input: str) -> bool:
    """Returns True if the query also asks for external web data (weather, surf, tides, stocks, news, buoys)."""
    u_lower = user_input.lower()
    return bool(
        re.search(
            r"\b(weather|forecast|surf|swell|tide|tides|buoy|ndbc|stock|price|market|dow|nasdaq|s&p|news|score|standings|table|largest|tallest)\b",
            u_lower,
        )
    )


def build_live_clock_skill_context(user_input: str) -> tuple[str, str]:
    """
    Executes the built-in Current Time, Date & World Clock Skill using the system's atomic
    NTP-synchronized clock and IANA zoneinfo database.
    Returns (short_console_status, detailed_skill_block).
    """
    now_local = datetime.now(LOCAL_TZ)
    now_utc = datetime.now(timezone.utc)
    yesterday_local = now_local - timedelta(days=1)
    tomorrow_local = now_local + timedelta(days=1)

    short_status = (
        f"{now_local.strftime('%A, %B %d, %Y • %I:%M:%S %p %Z (%H:%M:%S)')} | "
        f"UTC: {now_utc.strftime('%H:%M:%S UTC (%a %b %d)')}"
    )

    u_lower = user_input.lower()
    world_lines = []
    for label, iana_name, keywords in WORLD_CLOCK_ZONES:
        # Always include US zones + UTC, plus any international zone matching the prompt
        if iana_name in ("America/Los_Angeles", "America/New_York", "UTC") or any(k in u_lower for k in keywords):
            try:
                z_dt = now_utc.astimezone(ZoneInfo(iana_name))
                world_lines.append(f"  • {label}: {z_dt.strftime('%A, %B %d, %Y at %I:%M:%S %p %Z (%H:%M:%S)')}")
            except Exception:
                continue

    skill_block = (
        "--- [SKILL: LIVE SYSTEM CLOCK & CALENDAR (AUTHORITATIVE REAL-TIME GROUNDING)] ---\n"
        f"• Current Local Date & Time : {now_local.strftime('%A, %B %d, %Y at %I:%M:%S %p %Z (%H:%M:%S, UTC%z)')}\n"
        f"• Current UTC / GMT Time    : {now_utc.strftime('%A, %B %d, %Y at %I:%M:%S %p UTC (%H:%M:%S GMT)')}\n"
        f"• Unix Epoch Timestamp      : {int(now_utc.timestamp())}\n"
        "• Local Calendar Reference  :\n"
        f"  - Yesterday : {yesterday_local.strftime('%A, %B %d, %Y')}\n"
        f"  - Today     : {now_local.strftime('%A, %B %d, %Y')}\n"
        f"  - Tomorrow  : {tomorrow_local.strftime('%A, %B %d, %Y')}\n"
    )
    if world_lines:
        skill_block += "• World Clock Reference     :\n" + "\n".join(world_lines) + "\n"
    skill_block += "--- [END LIVE SYSTEM CLOCK SKILL] ---"
    return short_status, skill_block


def fetch_web_context(
    query: str,
    max_results: int = 5,
    max_snippet_chars: int = 300,
    max_total_chars: int = 4500,
) -> str:
    """
    Fetches rich answer cards, live HTML tables from top results, organic web snippets,
    and recent news snippets, filtering out garbage and boilerplate.
    """
    if DDGS is None:
        return "Web search dependency missing. Run `pip install ddgs`."

    try:
        today_str, local_clock_str, utc_clock_str = get_dual_clock_str()
        ddgs_client = DDGS()
        rich_cards = fetch_yahoo_rich_cards(query)

        try:
            text_results = list(ddgs_client.text(query, max_results=max_results))
        except Exception:
            text_results = []

        # If DDGS returns empty when a year suffix (e.g. '2026') is attached, retry without the trailing year
        if len(text_results) < 2 and re.search(r"\b202\d\b", query):
            stripped_q = re.sub(r"\b202\d\b", "", query).strip()
            if stripped_q:
                try:
                    extra_res = list(ddgs_client.text(stripped_q, max_results=max_results))
                    for er in extra_res:
                        if all(er.get("href") != tr.get("href") for tr in text_results):
                            text_results.append(er)
                except Exception:
                    pass

        try:
            news_results = list(ddgs_client.news(query, max_results=3))
        except Exception:
            news_results = []

        table_excerpt = fetch_top_page_table_or_excerpt(text_results, query=query)

        if not rich_cards and not table_excerpt and not text_results and not news_results:
            return "No web search results found."

        context = (
            f"--- CLEANED WEB SEARCH DATA (Retrieved Local: {local_clock_str} | Current UTC/GMT: {utc_clock_str}) "
            f"FOR QUERY: '{query}' ---\n"
        )
        valid_count = 0

        # Prepend rich cards (e.g., live closing prices, after-hours quotes, key stats)
        for card in rich_cards:
            valid_count += 1
            context += f"[{valid_count}] {card}\n\n"

        # Prepend live extracted HTML table when present (e.g., top-20 rankings, market cap tables)
        if table_excerpt:
            valid_count += 1
            context += f"[{valid_count}] {table_excerpt}\n\n"

        # Add organic search snippets
        for res in text_results:
            raw_snippet = res.get("body", "")
            cleaned_snippet = clean_text(raw_snippet)

            if is_junk_snippet(cleaned_snippet):
                continue

            title = clean_text(res.get("title", "No Title"))
            href = res.get("href", "")

            if len(cleaned_snippet) > max_snippet_chars:
                cleaned_snippet = cleaned_snippet[:max_snippet_chars] + "..."

            valid_count += 1
            source_line = f" ({href})" if href else ""
            context += f"[{valid_count}] {title}{source_line}\nSnippet: {cleaned_snippet}\n\n"

        # Add recent news snippets (vital for stock movements, headlines, and breaking events)
        for nres in news_results:
            raw_snippet = nres.get("body", "")
            cleaned_snippet = clean_text(raw_snippet)
            if is_junk_snippet(cleaned_snippet):
                continue

            title = clean_text(nres.get("title", "News Item"))
            date_str = clean_text(nres.get("date", ""))[:10]
            url_str = nres.get("url", "")
            if len(cleaned_snippet) > max_snippet_chars:
                cleaned_snippet = cleaned_snippet[:max_snippet_chars] + "..."

            valid_count += 1
            meta = f" ({date_str} - {url_str})" if date_str or url_str else ""
            context += f"[{valid_count}] [News] {title}{meta}\nSnippet: {cleaned_snippet}\n\n"

        if valid_count == 0:
            return "No high-quality search results retrieved after filtering."

        context += "--- END WEB DATA ---\n"

        if len(context) > max_total_chars:
            context = context[:max_total_chars] + "\n[Context Truncated]\n--- END WEB DATA ---\n"

        return context
    except Exception as e:
        return f"Failed to retrieve web search context: {e}"


def requires_post_cutoff_search(user_input: str) -> bool:
    """
    Detects queries about time-sensitive officeholders, political leaders, CEOs, champions,
    or 'last N / current / latest' sequences that can change after the model's 2024 pre-training
    cutoff and therefore MUST trigger a live web search.
    """
    u_lower = user_input.lower().strip()
    if is_clock_skill_query(user_input) and not is_external_data_query(user_input):
        return False
    office_or_title_pat = (
        r"\b(president|presidents|vice\s+president|prime\s+minister|governor|mayor|senator|speaker|"
        r"chancellor|pope|monarch|king|queen|ceo|cfo|chairman|fed\s+chair|secretary\s+of|administration|"
        r"champion|champions|winner|winners|super\s+bowl|world\s+series|world\s+cup|nba\s+finals|"
        r"stanley\s+cup|olympics|richest|most\s+valuable|largest\s+market\s+cap)\b"
    )
    temporal_qualifier_pat = (
        r"\b(current|currently|present|incumbent|now|today|latest|recent|most\s+recent|"
        r"last\s+\d+|past\s+\d+|who\s+is\s+the|who\s+are\s+the|2025|2026)\b"
    )
    return bool(re.search(office_or_title_pat, u_lower) and re.search(temporal_qualifier_pat, u_lower))


def determine_search_query(user_input: str, history: list) -> str | None:
    """
    Fast, context-aware classification pass:
    - Deterministically skips pure coding/math/physics prompts that don't request live external lookup.
    - Enforces post-2024 cutoff verification for current/recent officeholders, presidents, CEOs, and champions.
    - Injects today's system date so 'current'/'today'/'latest' queries use the real year (not training cutoff).
    - Uses up to the last 3 conversation turn pairs (600 chars/turn) + extracted station/ticker IDs
      to resolve multi-hop pronouns ('that buoy', 'the second one', 'their', 'it').
    - Automatically adapts to the active MODEL_PROFILE (skipping CoT when supported).
    """
    u_lower = user_input.lower().strip()
    must_search_post_cutoff = requires_post_cutoff_search(user_input)

    # Fast deterministic guard: pure clock/date/day-of-week queries are answered by the built-in Current Time Skill
    if is_clock_skill_query(user_input) and not is_external_data_query(user_input):
        return None

    # Fast deterministic route for US President queries (avoids LLM router hallucinating pre-2025 officeholders into the query)
    if must_search_post_cutoff and re.search(r"\bpresidents?\b", u_lower) and not re.search(
        r"\b(france|french|mexico|mexican|russia|russian|china|chinese|ukraine|ukrainian|brazil|brazilian|iran|iranian|korea|korean|taiwan|india|indian|argentina|turkey|turkish|poland|polish|philippines|indonesia|south\s+africa|nigeria|egypt|colombia|venezuela|company|university|college|bank|fed|federal\s+reserve|senate|council|board|club)\b",
        u_lower,
    ):
        return "List of presidents of the United States Wikipedia"

    # Fast deterministic guard against pure coding / math / textbook physics false positives
    is_pure_code_or_math = (
        re.search(
            r"\b(write\s+a\s+(?:python|bash|c\+\+|rust|javascript|js|sql|html)\b|write\s+a\s+(?:script|function|program|class)\b|implement\s+a\s+function|electric\s+current\s+in\s+a\s+circuit|ohm'?s\s+law|solve\s+for\s+[a-z]\b|derivative\s+of\b|integrate\b)",
            u_lower,
        )
        and not re.search(r"\b(search\s+the\s+web|pull\s+live|noaa|ndbc|buoy\s+\d{5}|yahoo\s+finance)\b", u_lower)
    )
    if is_pure_code_or_math:
        return None

    today_str, _, _ = get_dual_clock_str()
    recent_turns = history[1:][-6:]  # Up to the last 3 user/assistant pairs
    context_lines = []
    recent_buoy_matches: list[tuple[str, str]] = []  # (station_id, surrounding_clause)

    for msg in recent_turns:
        role_label = "User" if msg.get("role") == "user" else "Assistant"
        raw_content = msg.get("content", "")
        snippet = raw_content[:600].replace("\n", " ")
        context_lines.append(f"{role_label}: {snippet}")
        for m_b in re.finditer(r"([^.!\n]{0,45}\b(46\d{3}|41\d{3}|42\d{3}|44\d{3}|51\d{3})\b[^.!\n]{0,65})", raw_content):
            clause = clean_text(m_b.group(1))
            st_id = m_b.group(2)
            if all(st_id != existing_id for existing_id, _ in recent_buoy_matches):
                recent_buoy_matches.append((st_id, clause))

    entity_hint = ""
    if recent_buoy_matches:
        entity_hint = "\nReferenced NOAA Buoy Stations in Context: " + " ; ".join(
            f"Station {sid} ({cl})" for sid, cl in recent_buoy_matches
        )

    if context_lines:
        eval_input = (
            "Recent Conversation Context:\n"
            + "\n".join(context_lines)
            + entity_hint
            + f"\n\nLatest User Prompt: {user_input}"
        )
    else:
        eval_input = f"Latest User Prompt: {user_input}"

    try:
        base_messages = [
            {
                "role": "system",
                "content": (
                    f"You are a search intent evaluator. Today's date is {today_str}.\n"
                    "Determine if the Latest User Prompt requires real-time web search data, "
                    "stock prices, financial quotes, rankings/tables, recent news, current weather, surf/swell/tide/buoy reports, specific URLs/repositories, product availability, or external specs.\n"
                    f"- CRITICAL CUTOFF RULE: Your internal pre-training knowledge ended in 2024, whereas today is {today_str}. NEVER return 'NONE' for any question about current, incumbent, present, latest, or 'last N' political leaders, presidents ('who is the current president', 'last 5 presidents'), prime ministers, governors, CEOs, sports champions, award winners, or events that could have changed since 2024 — EVEN IF a previous turn in Recent Conversation Context already stated an answer! When 'president' is asked without a country, assume 'United States' (e.g. 'list of Presidents of the United States').\n"
                    "- Use Recent Conversation Context to resolve pronouns, implicit topic references (e.g., 'for today only instead', 'that buoy', 'the second one'), or when previous turns failed to provide the requested data.\n"
                    "- When Recent Conversation Context contains a specific 5-digit NOAA buoy station ID (e.g., 46221, 46054, 46218) that matches the buoy referenced by the user, ALWAYS include that exact 5-digit station ID (e.g. 'NOAA buoy 46054 wave height water temp') in your search query.\n"
                    f"- When the user asks for 'current', 'latest', or 'today', use the current date/year ({today_str}) in your search query (NEVER use 2023 or 2024).\n"
                    "- If the user asks to pull data, search, update, verify, or narrow a previous real-time data request (such as 'for today only instead'), ALWAYS output a search query combining the topic from Recent Conversation Context with the Latest User Prompt.\n"
                    "- If search IS NOT needed (pure math, coding, abstract reasoning, or timeless pre-2024 historical/scientific facts), reply ONLY with 'NONE'.\n"
                    "- If search IS needed, reply ONLY with a concise 2-10 word web search query (NO explanations or preamble)."
                ),
            },
            {"role": "user", "content": eval_input},
        ]
        req_messages, extra_params = prepare_request_messages_and_extra(
            base_messages, use_thinking=False, is_utility_call=True
        )
        create_kwargs = {
            "model": MODEL_NAME,
            "messages": req_messages,
            "temperature": 0.0,
            "max_tokens": 45,
        }
        if extra_params:
            create_kwargs["extra_body"] = extra_params

        response = client.chat.completions.create(**create_kwargs)
        res_text = strip_thinking_and_dedup(response.choices[0].message.content or "")
        if not res_text:
            return None

        # If the model included a label like "Search Query:", extract what follows
        for marker in ("Search Query:", "search query:", "Query:", "query:"):
            if marker in res_text:
                res_text = res_text.split(marker)[-1].strip()

        lines = [ln.strip().strip("*`\"'") for ln in res_text.splitlines() if ln.strip()]
        res_text = lines[-1] if lines else ""

        if not res_text or res_text.upper().startswith("NONE") or res_text.upper() == "NONE":
            if must_search_post_cutoff:
                if "president" in u_lower and any(x in u_lower for x in ("usa", "u.s", "united states", "current", "last")):
                    return "list of Presidents of the United States Wikipedia"
                return f"{user_input} {datetime.now(LOCAL_TZ).year}"
            return None

        # Guard against leaked reasoning sentences or declarative answers ("The current ... is ...") being sent to DuckDuckGo
        lower_q = res_text.lower()
        is_declarative_answer = bool(
            re.match(r"^the\s+(current|last|latest|present)\b.*\b(is|are|was|were)\b", lower_q)
            or res_text.endswith(".")
        )
        if is_declarative_answer or len(res_text.split()) > 12 or any(
            phrase in lower_q
            for phrase in ("to determine", "i'll analyze", "i will analyze", "the user is asking", "user's request")
        ):
            cleaned_fallback = re.sub(
                r"\b(show me|who|is|are|was|were|the|with|for|create|a|table|this|please|format|in|go|and|pull|information|from|publically|available|data|on)\b",
                " ",
                user_input,
                flags=re.I,
            )
            cleaned_fallback = re.sub(r"\s+", " ", cleaned_fallback).strip()
            res_text = f"{cleaned_fallback} {datetime.now(LOCAL_TZ).year}".strip()[:80] if cleaned_fallback else user_input[:80]

        res_text = res_text.replace('"', "").replace("'", "")

        # Ensure generic "president" queries default to United States President rather than film/book titles
        if "president" in u_lower and not any(
            c in res_text.lower() for c in ("united states", "u.s", "usa", "france", "mexico", "russia", "china", "ukraine", "brazil", "iran", "korea", "taiwan")
        ):
            res_text = f"United States {res_text}"

        # Post-processing: if query is about a buoy from recent context but lacks the 5-digit ID, attach the matching ID
        if recent_buoy_matches and "buoy" in res_text.lower() and not re.search(r"\b\d{5}\b", res_text):
            q_words = set(re.findall(r"[a-z]{4,}", (res_text + " " + " ".join(context_lines[-2:])).lower()))
            best_id = None
            best_overlap = 0
            for sid, cl in recent_buoy_matches:
                cl_words = set(re.findall(r"[a-z]{4,}", cl.lower())) - {"station", "located", "buoy", "near"}
                overlap = len(q_words & cl_words)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_id = sid
            if not best_id and len(recent_buoy_matches) == 1:
                best_id = recent_buoy_matches[0][0]
            if best_id:
                res_text = f"NOAA buoy {best_id} {res_text}"

        return res_text
    except Exception:
        return None


def needs_quantitative_reasoning(user_input: str) -> bool:
    """
    Detects prompts that require multi-step arithmetic, portfolio share allocation,
    unit conversion, or physics/engineering formula evaluation so the harness can
    instruct the model to show explicit intermediate equations even when think_mode is OFF.
    """
    u_lower = user_input.lower()
    quant_patterns = (
        r"\b(calculate|compute|derive|solve|formula|equation|wavelength|phase\s+speed)\b",
        r"\b(how\s+many\s+(?:whole\s+)?shares|left\s+over|remainder|split\s+\d+\s*/\s*\d+|allocation)\b",
        r"\b(ratio\s+of|percentage\s+difference|compound\s+annual|cagr|standard\s+deviation)\b",
        r"\b(convert\b.*\bto\b.*\band\b|in\s+both\s+(?:utc|gmt|meters|feet|knots))\b",
        r"\b(minutes\s+ago|hours\s+ago|how\s+long\s+ago)\b",
        r"\$\s*\d[\d,]*(?:\.\d+)?",
    )
    return any(re.search(pat, u_lower) for pat in quant_patterns)


def estimate_history_tokens(history: list[dict]) -> int:
    """Approximates verbatim history token count (~4 chars/token) excluding the system prompt."""
    return sum(len(m.get("content", "")) for m in history[1:]) // 4


def compress_memory_batch(existing_summary: str, evicted_pairs: list[tuple[str, str]]) -> str:
    """
    Folds a batch of overflowing (user, assistant) turn pairs into the summary buffer in a
    single non-thinking LLM call, preserving all exact numerical figures, tickers, timestamps,
    and station IDs verbatim while avoiding per-turn KV prefix-cache invalidation.
    """
    if not evicted_pairs:
        return existing_summary

    turns_block = "\n\n".join(
        f"Turn {idx + 1}:\nUser: {u_txt}\nAssistant Answer: {a_txt}"
        for idx, (u_txt, a_txt) in enumerate(evicted_pairs)
    )
    prompt = (
        f"Existing Conversation Summary:\n{existing_summary if existing_summary else 'None'}\n\n"
        f"New Turns to Integrate ({len(evicted_pairs)} turn(s)):\n{turns_block}\n\n"
        "Produce an updated, dense bulleted memory log (up to 15 concise bullets) that:\n"
        "1. Retains all existing facts and topics from Existing Conversation Summary.\n"
        "2. Integrates the New Turns while preserving EVERY exact number, stock price, point/percentage change, "
        "wave height, wave period, wave direction, temperature, date/time, ticker symbol, and 5-digit buoy station ID verbatim.\n"
        "3. NEVER replaces specific numbers with vague prose (for example, write "
        "'Buoy 46221: WVHT 3.6 ft (1.1 m), DPD 14 sec, APD 6.9 sec, MWD 284 deg, WTMP 63.1°F; "
        "^DJI 47,951.85 (-0.28%), ^GSPC 7,089.14 (-0.25%), ^IXIC 23,825.46 (-0.83%)' instead of 'provided buoy and index readings').\n"
        "Output ONLY the bulleted list with no preamble or meta-commentary."
    )
    try:
        console.print(
            f"[dim]Batch-compressing {len(evicted_pairs)} oldest turn pair(s) into rolling summary buffer...[/dim]"
        )
        base_messages = [
            {
                "role": "system",
                "content": (
                    "You are a high-precision numerical conversation memory archivist. "
                    "You never drop or round away exact numbers, tickers, or station IDs when summarizing."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        req_messages, extra_params = prepare_request_messages_and_extra(
            base_messages, use_thinking=False, is_utility_call=True
        )
        create_kwargs = {
            "model": MODEL_NAME,
            "messages": req_messages,
            "temperature": 0.0,
            "max_tokens": 700,
        }
        if extra_params:
            create_kwargs["extra_body"] = extra_params

        response = client.chat.completions.create(**create_kwargs)
        cleaned = strip_thinking_and_dedup(response.choices[0].message.content or "")
        if cleaned:
            return cleaned
    except Exception:
        pass

    fallback_entries = "\n".join(
        f"- User asked: '{u_txt}' | Assistant answered: '{a_txt[:240]}...'" for u_txt, a_txt in evicted_pairs
    )
    return f"{existing_summary}\n{fallback_entries}" if existing_summary else fallback_entries


def compress_memory(existing_summary: str, old_user: str, old_assistant: str) -> str:
    """Single-pair wrapper around compress_memory_batch for backward compatibility."""
    return compress_memory_batch(existing_summary, [(old_user, old_assistant)])


def race_router_and_speculative_stream(
    user_input: str,
    history: list[dict],
    spec_active_payload: list[dict],
    spec_extra_body: dict,
    effective_thinking: bool,
) -> tuple[str | None, queue.Queue | None, dict | None]:
    """
    Feature B — Speculative Router + Direct-Answer Racing:
    Fires `determine_search_query` (Slot 1) and the direct-response stream (Slot 2) simultaneously.
    - If Slot 1 returns `None` (no web search needed), Slot 2 is already prefilled and streaming tokens,
      hiding the ~250-400ms search-router latency completely.
    - If Slot 1 returns a search query, Slot 2 is immediately aborted/closed (<1ms) to free the server slot.
    """
    u_lower = user_input.lower().strip()
    if (is_clock_skill_query(user_input) and not is_external_data_query(user_input)) or (
        re.search(
            r"\b(write\s+a\s+(?:python|bash|c\+\+|rust|javascript|js|sql|html)\b|write\s+a\s+(?:script|function|program|class)\b|implement\s+a\s+function|electric\s+current\s+in\s+a\s+circuit|ohm'?s\s+law|solve\s+for\s+[a-z]\b|derivative\s+of\b|integrate\b)",
            u_lower,
        )
        and not re.search(r"\b(search\s+the\s+web|pull\s+live|noaa|ndbc|buoy\s+\d{5}|yahoo\s+finance)\b", u_lower)
    ):
        # Deterministic direct query: no router LLM call needed at all
        return None, None, None

    # If the prompt deterministically requires post-2024 cutoff verification, skip speculative direct stream
    if requires_post_cutoff_search(user_input):
        return determine_search_query(user_input, history), None, None

    spec_queue: queue.Queue = queue.Queue()
    abort_event = threading.Event()
    spec_meta = {
        "start_time": time.perf_counter(),
        "first_token_time": None,
        "abort_event": abort_event,
    }

    def _spec_stream_worker():
        stream_obj = None
        try:
            stream_kwargs = {
                "model": MODEL_NAME,
                "messages": spec_active_payload,
                "temperature": 0.6 if effective_thinking else 0.7,
                "max_tokens": MODEL_PROFILE["max_gen_tokens"],
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if spec_extra_body:
                stream_kwargs["extra_body"] = spec_extra_body

            stream_obj = client.chat.completions.create(**stream_kwargs)
            for chunk in stream_obj:
                if abort_event.is_set():
                    break
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta.content or ""
                    if delta and spec_meta["first_token_time"] is None:
                        spec_meta["first_token_time"] = time.perf_counter()
                spec_queue.put(("chunk", chunk))
        except Exception as e:
            if not abort_event.is_set():
                spec_queue.put(("error", e))
        finally:
            if abort_event.is_set() and stream_obj is not None:
                try:
                    stream_obj.close()
                except Exception:
                    pass
            spec_queue.put(("done", None))

    worker_thread = threading.Thread(target=_spec_stream_worker, daemon=True)
    worker_thread.start()

    search_query = determine_search_query(user_input, history)
    if search_query is not None:
        # Web search is required -> immediately abort speculative direct stream
        abort_event.set()
        return search_query, None, None

    return None, spec_queue, spec_meta


def detect_auto_parallel_sections(user_input: str) -> list[str] | None:
    """
    Automatically detects when a user prompt naturally decomposes into 3 to 6 independent parallel
    sections without requiring any slash command:
      1. Explicit numbered sub-questions: `1) ... 2) ... 3) ...`
      2. Multi-ticker / multi-company comparisons (>= 3 tickers/companies)
      3. Multi-topic comma-separated explanations (`Explain A, B, C, and D`)
    """
    # 1. Explicit numbered sub-tasks: "1) ... 2) ... 3) ..."
    numbered = re.findall(
        r"(?:^|\s)(?:\d+[\)\.]\s*)([^0-9\n][^\n\?;]{6,120}?)(?=(?:\s+\d+[\)\.]\s+)|$)",
        user_input.strip(),
    )
    if 3 <= len(numbered) <= 6:
        return [clean_text(item).rstrip(".,;") for item in numbered]

    u_lower = user_input.lower().strip()

    # 2. Multi-ticker / multi-company comparison of >= 3 tickers
    if re.search(r"\b(compare|comparison|analyze|breakdown|break\s+down|contrast|detailed|evaluate|outlook)\b", u_lower):
        tickers = []
        for tok in re.findall(r"\b[A-Z]{2,5}\b", user_input):
            if tok not in _TICKER_STOPWORDS and tok not in tickers:
                tickers.append(tok)
        for name_key, (sym, _) in _KNOWN_COMPANY_TICKERS.items():
            if re.search(rf"\b{re.escape(name_key)}\b", u_lower) and sym not in tickers:
                tickers.append(sym)
        if 3 <= len(tickers) <= 5:
            sections = [f"{sym} — Live Quote, Performance & Key Metrics Analysis" for sym in tickers]
            sections.append(f"Side-by-Side Comparison Table & Relative Summary ({', '.join(tickers)})")
            return sections

    # 3. Comma-separated multi-topic explanations: "Explain/Describe/Compare A, B, C, and D"
    m_list = re.match(
        r"^(?:please\s+)?(?:explain|describe|compare|detail|break\s+down|summarize)\s+(?:the\s+(?:differences\s+between|following\s+topics|concepts\s+of)\s*:?\s*)?(.+)$",
        user_input.strip(),
        flags=re.I,
    )
    if m_list and "," in m_list.group(1):
        tail = m_list.group(1).strip().rstrip(".?!")
        raw_items = re.split(r"\s*,\s*(?:and\s+)?|\s+and\s+", tail)
        clean_items = [it.strip() for it in raw_items if 4 <= len(it.strip()) <= 75 and len(it.strip().split()) <= 10]
        if 3 <= len(clean_items) <= 5:
            return clean_items

    return None


def plan_parallel_sections(user_input: str) -> list[str] | None:
    """
    Feature A — Automatic Section-Parallel / Map-Reduce Planner:
    Automatically returns a list of 3-6 independent section titles when a prompt is a multi-part,
    multi-entity, or comprehensive deep-dive request; returns `None` for normal single-stream prompts.
    """
    auto_secs = detect_auto_parallel_sections(user_input)
    if auto_secs:
        return auto_secs

    u_lower = user_input.lower().strip()
    is_deep_dive = bool(
        re.search(
            r"\b(comprehensive\s+(?:guide|overview|breakdown|analysis)|in-depth\s+(?:guide|analysis|breakdown|explanation)|detailed\s+(?:guide|breakdown|architectural\s+overview)|multi-section|step-by-step\s+guide\s+to\s+building)\b",
            u_lower,
        )
    )
    if not is_deep_dive:
        return None

    try:
        base_messages = [
            {
                "role": "system",
                "content": (
                    "Decompose the user's request into exactly 4 distinct, non-overlapping section titles "
                    "that can be written in parallel to form a complete, well-structured response.\n"
                    "Output ONLY 4 lines, one concise section title per line (no numbering, no bullets, no preamble)."
                ),
            },
            {"role": "user", "content": user_input},
        ]
        req_messages, extra_params = prepare_request_messages_and_extra(
            base_messages, use_thinking=False, is_utility_call=True
        )
        create_kwargs = {
            "model": MODEL_NAME,
            "messages": req_messages,
            "temperature": 0.0,
            "max_tokens": 100,
        }
        if extra_params:
            create_kwargs["extra_body"] = extra_params

        resp = client.chat.completions.create(**create_kwargs)
        raw = strip_thinking_and_dedup(resp.choices[0].message.content or "")
        lines = []
        for ln in raw.splitlines():
            cleaned_ln = re.sub(r"^[\s\-\*\d\.\)\:]+", "", ln).strip().strip("*\"'`")
            if len(cleaned_ln) >= 4:
                lines.append(cleaned_ln[:90])
        if 3 <= len(lines) <= 5:
            return lines[:5]
    except Exception:
        pass

    return [
        "Core Overview & Fundamental Principles",
        "Architecture & Underlying Mechanics",
        "Quantitative Performance & Practical Trade-offs",
        "Summary Synthesis & Key Takeaways",
    ]


class ProgressiveMarkdownStreamer:
    """
    SSH- & Scrollback-Safe Progressive Markdown Streamer:
    - Why standard `Live(Markdown(full_response))` jitters and breaks SSH scrollback:
      Redrawing a multi-line Markdown document 15x/sec sends ANSI cursor-up sequences (`\\x1b[<N>A`).
      Over SSH, erasing and re-printing dozens of lines causes visible flicker/jitter, and once the
      response exceeds 1 terminal screen height (~40 lines), cursor-up clamps at row 0 of the viewport,
      pushing duplicate frames into the terminal scrollback buffer on every token chunk.
    - How `ProgressiveMarkdownStreamer` solves both:
      1. The mutable `Live` region is strictly a single non-wrapping 1-line status ticker (`no_wrap=True`,
         `transient=True`) at the very bottom of the screen, so cursor-up never exceeds 1 line and ZERO
         duplicate frames ever enter the SSH scrollback buffer.
      2. As soon as a self-contained Markdown block (heading, paragraph, completed table, completed list,
         or closed code fence) finishes at a `\\n\\n` boundary, it is permanently printed ONCE above the
         status ticker via `live.console.print(Markdown(block))` and never redrawn again.
    """

    def __init__(self, target_console: Console):
        self.console = target_console
        self.committed_char_len = 0
        self.token_count = 0

    @staticmethod
    def _is_table_line(line: str) -> bool:
        s = line.strip()
        return s.startswith("|") or (s.count("|") >= 2 and not s.startswith(("http", "//")))

    @staticmethod
    def _is_list_line(line: str) -> bool:
        return bool(
            re.match(
                r"^(\s*[-*•+]\s+|\s*\d+[\.\)]\s+|\s+[^\s])",
                line,
            )
        )

    def _find_safe_commit_boundary(self, uncommitted: str) -> int:
        """
        Returns the character index `idx` in `uncommitted` up to which complete Markdown blocks
        can be permanently printed without splitting an in-progress code fence, table, or list.
        Returns 0 if no complete block boundary is ready yet.
        """
        if "\n\n" not in uncommitted:
            return 0

        search_pos = 0
        best_cut = 0

        while True:
            sep_idx = uncommitted.find("\n\n", search_pos)
            if sep_idx == -1:
                break
            after_sep = sep_idx + 2
            # Require at least 2 non-newline characters after `\n\n` so we can inspect what starts next
            remainder = uncommitted[after_sep:]
            remainder_stripped = remainder.lstrip("\n")
            if len(remainder_stripped) < 2 or "\n" not in remainder_stripped:
                break

            candidate = uncommitted[:sep_idx]
            next_first_line = remainder_stripped.split("\n", 1)[0]

            # 1. Never split inside an unclosed ``` code fence
            if candidate.count("```") % 2 == 1:
                search_pos = after_sep
                continue

            # 2. Never split inside an ongoing Markdown table
            cand_lines = [ln for ln in candidate.splitlines() if ln.strip()]
            last_line = cand_lines[-1] if cand_lines else ""
            if self._is_table_line(last_line) and self._is_table_line(next_first_line):
                search_pos = after_sep
                continue

            # 3. Never split inside an ongoing bullet/numbered list unless the next block is a heading/rule
            if (
                self._is_list_line(last_line)
                and self._is_list_line(next_first_line)
                and not next_first_line.lstrip().startswith(("#", "---"))
            ):
                search_pos = after_sep
                continue

            # 4. Never split a standalone heading from its body paragraph
            if last_line.lstrip().startswith("#"):
                search_pos = after_sep
                continue

            best_cut = after_sep
            search_pos = after_sep

        return best_cut

    def make_status_line(self, tail_text: str, prefix_label: str = "⚡ Streaming") -> Text:
        width = max(40, self.console.size.width - 4)
        preview = re.sub(r"\s+", " ", tail_text).strip()
        if len(preview) > width - 28:
            preview = "…" + preview[-(width - 29) :]
        line = Text(no_wrap=True, overflow="ellipsis")
        line.append(f"{prefix_label} ({self.token_count} tok): ", style="bold cyan")
        line.append(preview if preview else "...", style="dim")
        return line

    def step(self, live: Live, display_text: str, prefix_label: str = "⚡ Streaming"):
        self.token_count += 1
        if len(display_text) < self.committed_char_len:
            self.committed_char_len = 0

        uncommitted = display_text[self.committed_char_len :]
        cut = self._find_safe_commit_boundary(uncommitted)
        if cut > 0:
            block_to_commit = uncommitted[:cut].strip()
            if block_to_commit:
                live.console.print(Markdown(block_to_commit))
                live.console.print()
            self.committed_char_len += cut
            uncommitted = display_text[self.committed_char_len :]

        live.update(self.make_status_line(uncommitted, prefix_label=prefix_label))

    def finish(self, display_text: str):
        if len(display_text) < self.committed_char_len:
            self.committed_char_len = 0
        remainder = display_text[self.committed_char_len :].strip()
        if remainder:
            self.console.print(Markdown(remainder))


def stream_parallel_sections(
    base_prefix_messages: list[dict],
    shared_active_user_content: str,
    sections: list[str],
    use_thinking: bool,
    verbose_flag: bool = False,
) -> tuple[str, float, float, int, float, float]:
    """
    Feature A — Section-Parallel / Map-Reduce Executor (Scrollback-Safe):
    Launches N simultaneous streaming requests (1 per section). While all N streams decode in parallel,
    displays a single-line non-wrapping status bar (`transient=True`) so zero jitter or scrollback
    corruption occurs over SSH, then prints each completed section permanently in order (1..N) as soon
    as it is ready!
    Returns (combined_response, ttft_ms, gen_duration, total_tokens, effective_tps, avg_stream_tps).
    """
    n_streams = len(sections)
    section_buffers = ["" for _ in range(n_streams)]
    section_tokens = [0 for _ in range(n_streams)]
    section_first_t: list[float | None] = [None for _ in range(n_streams)]
    section_end_t: list[float | None] = [None for _ in range(n_streams)]
    section_done = [False for _ in range(n_streams)]
    section_printed = [False for _ in range(n_streams)]
    lock = threading.Lock()
    start_time = time.perf_counter()

    def _run_section_stream(idx: int, sec_title: str):
        other_titles = ", ".join(f"'{s}'" for j, s in enumerate(sections) if j != idx)
        tail_assignment = (
            f"\n\n### PARALLEL STREAM ASSIGNMENT (Section {idx + 1} of {n_streams}):\n"
            f"You are Stream {idx + 1} of {n_streams} generating a multi-section response in parallel.\n"
            f"Write ONLY Section {idx + 1}: **{sec_title}**.\n"
            f"- Start directly with the Markdown heading `### {idx + 1}. {sec_title}`.\n"
            f"- Focus exclusively on `{sec_title}` (~120–220 words, using Markdown tables or bullets where helpful).\n"
            f"- Do NOT cover the other parallel sections ({other_titles}) and do NOT include a generic intro or conclusion."
        )
        sec_payload = list(base_prefix_messages) + [
            {"role": "user", "content": shared_active_user_content + tail_assignment}
        ]
        active_sec_msgs, extra_sec_params = prepare_request_messages_and_extra(
            sec_payload, use_thinking=use_thinking, is_utility_call=False
        )
        if idx == 0 and verbose_flag:
            console.print(
                Panel(
                    json.dumps(active_sec_msgs, indent=2),
                    title=f"[bold yellow]EXACT vLLM Parallel Payload (Stream 1 of {n_streams})[/bold yellow]",
                    border_style="yellow",
                    expand=False,
                )
            )
        stream_kwargs = {
            "model": MODEL_NAME,
            "messages": active_sec_msgs,
            "temperature": 0.6 if use_thinking else 0.7,
            "max_tokens": min(700, MODEL_PROFILE["max_gen_tokens"]),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if extra_sec_params:
            stream_kwargs["extra_body"] = extra_sec_params

        try:
            stream = client.chat.completions.create(**stream_kwargs)
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        now_t = time.perf_counter()
                        with lock:
                            if section_first_t[idx] is None:
                                section_first_t[idx] = now_t
                            section_buffers[idx] += delta
                if hasattr(chunk, "usage") and chunk.usage:
                    with lock:
                        section_tokens[idx] = chunk.usage.completion_tokens
        except Exception as e:
            with lock:
                section_buffers[idx] += f"\n*(Stream {idx + 1} error: {e})*"
        finally:
            with lock:
                section_end_t[idx] = time.perf_counter()
                if section_tokens[idx] == 0 and section_buffers[idx]:
                    section_tokens[idx] = max(1, len(section_buffers[idx]) // 4)
                section_done[idx] = True

    def _format_single_section(idx: int) -> str:
        title = sections[idx]
        cleaned_sec = strip_thinking_and_dedup(section_buffers[idx])
        if cleaned_sec:
            if not cleaned_sec.lstrip().startswith("#"):
                cleaned_sec = f"### {idx + 1}. {title}\n{cleaned_sec}"
            return cleaned_sec
        return f"### {idx + 1}. {title}\n*(No content)*"

    def _make_parallel_status_line() -> Text:
        with lock:
            badges = []
            total_chars = 0
            for i in range(n_streams):
                n_tok_est = section_tokens[i] or (len(section_buffers[i]) // 4)
                total_chars += len(section_buffers[i])
                state = "✓" if section_done[i] else f"{n_tok_est}t"
                badges.append(f"S{i + 1}:{state}")
            status_str = " ".join(badges)
        line = Text(no_wrap=True, overflow="ellipsis")
        line.append(f"⚡ Parallel Map-Reduce ({n_streams} streams | ~{total_chars // 4} tok): ", style="bold cyan")
        line.append(status_str, style="dim")
        return line

    threads = [
        threading.Thread(target=_run_section_stream, args=(i, sec_title), daemon=True)
        for i, sec_title in enumerate(sections)
    ]
    for t in threads:
        t.start()

    next_to_print = 0
    with Live(
        _make_parallel_status_line(),
        console=console,
        refresh_per_second=8,
        transient=True,
    ) as live:
        while not all(section_done):
            # Permanently print completed sections in sequential order (0, 1, 2, ...) as soon as each finishes!
            while next_to_print < n_streams and section_done[next_to_print]:
                with lock:
                    sec_md = _format_single_section(next_to_print)
                    section_printed[next_to_print] = True
                if next_to_print > 0:
                    live.console.print(Markdown("---"))
                live.console.print(Markdown(sec_md))
                live.console.print()
                next_to_print += 1
            live.update(_make_parallel_status_line())
            time.sleep(0.08)

        while next_to_print < n_streams:
            with lock:
                sec_md = _format_single_section(next_to_print)
                section_printed[next_to_print] = True
            if next_to_print > 0:
                live.console.print(Markdown("---"))
            live.console.print(Markdown(sec_md))
            live.console.print()
            next_to_print += 1

    for t in threads:
        t.join(timeout=1.0)

    end_time = time.perf_counter()
    valid_firsts = [t for t in section_first_t if t is not None]
    first_token_t = min(valid_firsts) if valid_firsts else end_time
    ttft_ms = (first_token_t - start_time) * 1000.0
    gen_duration = max(0.001, end_time - first_token_t)
    total_tokens = sum(section_tokens)
    effective_tps = total_tokens / gen_duration

    per_stream_rates = []
    for i in range(n_streams):
        if section_first_t[i] and section_end_t[i] and section_end_t[i] > section_first_t[i]:
            per_stream_rates.append(section_tokens[i] / (section_end_t[i] - section_first_t[i]))
    avg_stream_tps = sum(per_stream_rates) / len(per_stream_rates) if per_stream_rates else (effective_tps / n_streams)

    final_parts = [_format_single_section(i) for i in range(n_streams)]
    combined_response = "\n\n---\n\n".join(final_parts)
    return combined_response, ttft_ms, gen_duration, total_tokens, effective_tps, avg_stream_tps


# --- Session Startup ---
console.print(
    Panel(
        f"[dim]Model:[/dim] [bold white]{MODEL_NAME}[/bold white]\n"
        f"[dim]Endpoint:[/dim] [cyan]{args.url}[/cyan] | "
        f"[dim]Detected Profile:[/dim] [bold green]{MODEL_PROFILE['description']}[/bold green] | "
        f"[dim]Context Limit:[/dim] [cyan]{MODEL_PROFILE['max_model_len']:,} tokens[/cyan]",
        title="[bold cyan]cjllm — Model-Agnostic vLLM Search & Inference Harness[/bold cyan]",
        subtitle="[dim]CJ vLLM Search Harness (@palisurf) • github.com/palisurf/cjllm-harness[/dim]",
        border_style="cyan",
        expand=False,
    )
)
start_file = console.input(
    f"Enter cache file to load [[dim]Press Enter for default: '{DEFAULT_CACHE_FILE}'[/dim]]: "
).strip()

if not start_file:
    start_file = DEFAULT_CACHE_FILE

conversation_history, summary_buffer = re_hydrate_cache(start_file)

CURRENT_DATE_STR, CURRENT_LOCAL_CLOCK_STR, CURRENT_UTC_CLOCK_STR = get_dual_clock_str()
LOCAL_TZ_LABEL = datetime.now(LOCAL_TZ).strftime("%Z (UTC%z)")
BASE_SYSTEM_PROMPT = (
    f"You are an intelligent, concise AI assistant. Today's local date is {CURRENT_DATE_STR} "
    f"(Local Timezone: {LOCAL_TZ_LABEL}). "
    "Prioritize answering the user's latest message directly and accurately, "
    "while using prior conversation turns to resolve follow-up references and maintain continuity. "
    "Treat the Prior Conversation Summary Buffer as background context. "
    "When a [SKILL: LIVE SYSTEM CLOCK & CALENDAR] block or live web search data is provided in the active turn, "
    "always report the exact live timestamps, dates, figures, and tables from that block over pre-training knowledge or older turns. "
    "NEVER invent, estimate, or generate placeholder tables or figures when real-time data is requested; "
    "report only the exact measurements present in the retrieved data."
)

if conversation_history is None:
    console.print("[dim][Cache] Initializing fresh context session.[/dim]")
    conversation_history = [
        {
            "role": "system",
            "content": BASE_SYSTEM_PROMPT,
        }
    ]
    summary_buffer = ""
else:
    # Ensure base system prompt is up to date
    if conversation_history and conversation_history[0].get("role") == "system":
        conversation_history[0]["content"] = BASE_SYSTEM_PROMPT

verbose_mode = False
think_mode = bool(args.think)

console.print("\n[bold]Commands:[/bold]")
console.print(
    f"  [cyan]Memory Window[/cyan]        : High-water {MAX_PAIRS} pairs (~{MAX_VERBATIM_TOKENS:,} tokens), batch-evicting {EVICT_BATCH_PAIRS} pairs"
)
console.print("  [cyan]/clear[/cyan]               : Flush conversation history & summary buffer")
console.print("  [cyan]/think[/cyan]               : Toggle Chain-of-Thought reasoning mode (default: OFF)")
console.print("  [cyan]/think <query>[/cyan]       : Run a single query with Chain-of-Thought reasoning enabled")
console.print("  [cyan]/v[/cyan]                   : Toggle verbose mode (echoes exact vLLM prompt payload)")
console.print("  [cyan]/v <query>[/cyan]           : Run a single query with exact vLLM prompt payload echoing forced")
console.print("  [cyan]/save <filename>[/cyan]     : De-hydrate current context log")
console.print("  [cyan]/load <filename>[/cyan]     : Re-hydrate context log from file")
console.print("  [cyan]chat_end[/cyan]             : Save to default cache and exit\n")

# --- Main Interaction Loop ---
while True:
    try:
        user_input = console.input("[bold blue]User:[/bold blue] ").strip()
        if not user_input:
            continue

        # Terminal Termination Handling
        if user_input.lower() in ["chat_end", "exit", "quit", "/exit", "/quit"]:
            console.print(f"\n[bold yellow][Ending Session][/bold yellow] De-hydrating cache to '{DEFAULT_CACHE_FILE}'...")
            de_hydrate_cache(conversation_history, summary_buffer, DEFAULT_CACHE_FILE)
            save_cmd_history()
            break

        # Context Flush Handling
        if user_input.lower() in ["/clear", "/reset"]:
            conversation_history = [{"role": "system", "content": BASE_SYSTEM_PROMPT}]
            summary_buffer = ""
            console.print("[bold yellow][Context Reset][/bold yellow] Flushed all history and summary buffer.\n")
            continue

        # Handle standalone `/think` mode toggle
        if user_input.lower() == "/think":
            think_mode = not think_mode
            status = "[bold green]ENABLED[/bold green]" if think_mode else "[bold red]DISABLED[/bold red]"
            console.print(f"[bold cyan][Think Mode][/bold cyan] Chain-of-Thought reasoning is now {status}.\n")
            continue

        # Handle standalone `/v` mode toggle
        if user_input.lower() == "/v":
            verbose_mode = not verbose_mode
            status = "[bold green]ENABLED[/bold green]" if verbose_mode else "[bold red]DISABLED[/bold red]"
            console.print(f"[bold cyan][Verbose Mode][/bold cyan] Prompt payload echoing is now {status}.\n")
            continue

        # Check for single-turn `/think <query>` or `/v <query>` prefixes
        force_think_this_turn = False
        force_verbose_this_turn = False

        while True:
            if user_input.startswith("/think "):
                force_think_this_turn = True
                user_input = user_input[7:].strip()
            elif user_input.startswith("/v "):
                force_verbose_this_turn = True
                user_input = user_input[3:].strip()
            else:
                break

        if not user_input:
            continue

        # Manual In-Session De-hydration
        if user_input.startswith("/save"):
            parts = user_input.split(maxsplit=1)
            target_file = parts[1].strip() if len(parts) > 1 else DEFAULT_CACHE_FILE
            de_hydrate_cache(conversation_history, summary_buffer, target_file)
            continue

        # Manual In-Session Re-hydration
        if user_input.startswith("/load"):
            parts = user_input.split(maxsplit=1)
            if len(parts) > 1:
                loaded_hist, loaded_sum = re_hydrate_cache(parts[1].strip())
                if loaded_hist is not None:
                    conversation_history = loaded_hist
                    summary_buffer = loaded_sum
            else:
                console.print("[bold red][Error][/bold red] Specify a filename: `/load <filename>`")
            continue

        use_thinking = think_mode or force_think_this_turn
        is_quant_query = needs_quantitative_reasoning(user_input)
        effective_thinking = use_thinking or is_quant_query
        quant_instruction = (
            "\n- For any mathematical calculations, share allocations, elapsed time ('minutes ago' by subtracting in the same timezone), or physics formulas requested, "
            "write out each intermediate step and equation explicitly before stating the final result so every number is exact."
            if is_quant_query
            else ""
        )

        # 1. Merge Base System Prompt & Summary Buffer into a Single System Message
        if summary_buffer:
            combined_system_content = (
                f"{BASE_SYSTEM_PROMPT}\n\n"
                f"Prior Conversation Summary Buffer (Background Context):\n{summary_buffer}"
            )
        else:
            combined_system_content = BASE_SYSTEM_PROMPT

        base_prefix_messages = [{"role": "system", "content": combined_system_content}]
        base_prefix_messages.extend(conversation_history[1:])

        # 2. Built-in Current Time, Date & World Clock Skill
        use_clock_skill = is_clock_skill_query(user_input)
        clock_skill_block = ""
        if use_clock_skill:
            clock_status, clock_skill_block = build_live_clock_skill_context(user_input)
            console.print(f"[dim][Skill: Current Time] [bold cyan]{clock_status}[/bold cyan][/dim]")

        # 3. Check Automatic Section-Parallel / Map-Reduce Decomposition (Feature A)
        parallel_sections = plan_parallel_sections(user_input)

        # Prepare direct-turn content (used for direct response or speculative racing)
        if use_clock_skill:
            direct_user_content = (
                f"{clock_skill_block}\n\n"
                "### INSTRUCTIONS:\n"
                "Answer the Active User Request below using the exact live current time, date, day of the week, and timezone values "
                f"from the [SKILL: LIVE SYSTEM CLOCK & CALENDAR] block above. Ignore any older timestamps from earlier conversation turns.{quant_instruction}\n\n"
                f"### ACTIVE USER REQUEST:\n{user_input}"
            )
        else:
            direct_user_content = (
                f"{user_input}\n\n[Instruction: Write out each intermediate calculation step explicitly before stating the final result.]"
                if is_quant_query and not use_thinking
                else user_input
            )

        spec_queue = None
        spec_meta = None

        if parallel_sections:
            # Automatic Section-Parallel turn: determine if web search is needed first
            if use_clock_skill and not is_external_data_query(user_input):
                search_query = None
            else:
                console.print("[dim]Analyzing search intent...[/dim]", end="\r")
                search_query = determine_search_query(user_input, conversation_history)
        else:
            # Standard turn: Feature B — Speculative Router + Direct-Answer Racing
            spec_base_payload = list(base_prefix_messages) + [{"role": "user", "content": direct_user_content}]
            spec_active_payload, spec_extra_body = prepare_request_messages_and_extra(
                spec_base_payload, use_thinking=effective_thinking, is_utility_call=False
            )
            console.print("[dim]Analyzing search intent (speculative direct stream active)...[/dim]", end="\r")
            search_query, spec_queue, spec_meta = race_router_and_speculative_stream(
                user_input,
                conversation_history,
                spec_active_payload,
                spec_extra_body,
                effective_thinking,
            )

        if search_query:
            console.print(f"[dim]Searching web (max 5) for: '[bold cyan]{search_query}[/bold cyan]'...[/dim]")
            web_payload = fetch_web_context(search_query, max_results=5)
            skill_prefix = f"{clock_skill_block}\n\n" if clock_skill_block else ""
            active_user_content = (
                f"{skill_prefix}{web_payload}\n"
                "### INSTRUCTIONS:\n"
                f"Today's date is {CURRENT_DATE_STR}. Your internal pre-training knowledge ended in 2024.\n"
                "- Answer the Active User Request below using the live web search data and extracted tables above.\n"
                "- CRITICAL: Chronological tables (such as lists of Presidents, leaders, or champions) list the oldest entries at the top and the MOST RECENT / INCUMBENT entries (2025–2026) at the bottom. Always check the bottom rows (e.g., #47 / 'Incumbent' / 2025–present) and override both your 2024 pre-training memory AND any stale earlier conversation turns with the 2025–2026 facts in the table!\n"
                "- When asked for the 'last N' or 'most recent N' presidents, leaders, or winners, list them in reverse chronological order starting from the current 2025–2026 incumbent at the bottom of the table (e.g., for last 5 US Presidents: start at #47 Donald Trump [2025–Incumbent], #46 Joe Biden [2021–2025], #45 Donald Trump [2017–2021], #44 Barack Obama [2009–2017], #43 George W. Bush [2001–2009], and #42 Bill Clinton [1993–2001] so both the last 5 presidencies and last 5 distinct individuals are clear)."
                f"{quant_instruction}\n\n"
                f"### ACTIVE USER REQUEST:\n{user_input}"
            )
        else:
            if spec_queue is not None:
                console.print("[dim]Direct response (speculative stream hit — router latency hidden).[/dim]")
            elif not use_clock_skill:
                console.print("[dim]Direct response (no web search needed).[/dim]")
            active_user_content = direct_user_content

        # --- EXECUTE GENERATION (AUTO SECTION-PARALLEL OR SINGLE STREAM) ---
        if parallel_sections:
            console.print(
                f"[dim][Auto-Parallel Map-Reduce] Launching [bold cyan]{len(parallel_sections)} simultaneous streams[/bold cyan]: "
                f"{' | '.join(parallel_sections)}[/dim]"
            )
            console.print("\n[bold green]Assistant:[/bold green]")
            (
                full_response,
                ttft_ms,
                gen_duration,
                completion_tokens,
                tps,
                avg_stream_tps,
            ) = stream_parallel_sections(
                base_prefix_messages,
                active_user_content,
                parallel_sections,
                use_thinking=effective_thinking,
                verbose_flag=(verbose_mode or force_verbose_this_turn),
            )
            speedup_x = tps / avg_stream_tps if avg_stream_tps > 0 else float(len(parallel_sections))
            metrics_summary = (
                f"[dim]Parallel Streams: [bold cyan]{len(parallel_sections)}[/bold cyan] | "
                f"TTFT: [bold]{ttft_ms:.1f}ms[/bold] | "
                f"Gen Time: [bold]{gen_duration:.2f}s[/bold] | "
                f"Tokens: [bold]{completion_tokens}[/bold] | "
                f"Effective Speed: [bold cyan]{tps:.1f} t/s[/bold cyan] "
                f"([bold green]{speedup_x:.1f}x[/bold green] vs {avg_stream_tps:.1f} t/s per stream)[/dim]"
            )
            console.print(Panel(metrics_summary, border_style="dim", expand=False))
            console.print()
        else:
            base_payload = list(base_prefix_messages) + [{"role": "user", "content": active_user_content}]
            active_payload, extra_body_params = prepare_request_messages_and_extra(
                base_payload, use_thinking=effective_thinking, is_utility_call=False
            )

            # Echo EXACT vLLM Payload Array if Verbose Mode is Active
            if verbose_mode or force_verbose_this_turn:
                full_payload_json = json.dumps(active_payload, indent=2)
                console.print(
                    Panel(
                        full_payload_json,
                        title="[bold yellow]EXACT vLLM Payload Sent to Model[/bold yellow]",
                        border_style="yellow",
                        expand=False,
                    )
                )

            console.print("\n[bold green]Assistant:[/bold green]")

            full_response = ""
            completion_tokens = 0

            prog_streamer = ProgressiveMarkdownStreamer(console)

            if spec_queue is not None and spec_meta is not None:
                # Consume the already-running speculative direct stream
                start_time = spec_meta["start_time"]
                first_token_time = spec_meta["first_token_time"]
                with Live(
                    prog_streamer.make_status_line(""),
                    console=console,
                    refresh_per_second=8,
                    transient=True,
                ) as live:
                    while True:
                        msg_type, item = spec_queue.get()
                        if msg_type == "done":
                            break
                        if msg_type == "error":
                            raise item
                        chunk = item
                        if chunk.choices and len(chunk.choices) > 0:
                            content = chunk.choices[0].delta.content or ""
                            if content:
                                if first_token_time is None:
                                    first_token_time = spec_meta["first_token_time"] or time.perf_counter()
                                full_response += content
                                if use_thinking:
                                    prog_streamer.step(live, format_thinking_markdown(full_response))
                                else:
                                    cleaned_live = strip_thinking_and_dedup(full_response)
                                    if cleaned_live:
                                        prog_streamer.step(live, cleaned_live)
                                    elif is_quant_query:
                                        live.update(
                                            prog_streamer.make_status_line(
                                                "Calculating intermediate equations...",
                                                prefix_label="⚡ Reasoning",
                                            )
                                        )
                        if hasattr(chunk, "usage") and chunk.usage:
                            completion_tokens = chunk.usage.completion_tokens
                final_disp = (
                    format_thinking_markdown(full_response)
                    if use_thinking
                    else strip_thinking_and_dedup(full_response)
                )
                prog_streamer.finish(final_disp)
                end_time = time.perf_counter()
            else:
                start_time = time.perf_counter()
                first_token_time = None
                stream_kwargs = {
                    "model": MODEL_NAME,
                    "messages": active_payload,
                    "temperature": 0.6 if effective_thinking else 0.7,
                    "max_tokens": MODEL_PROFILE["max_gen_tokens"],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                if extra_body_params:
                    stream_kwargs["extra_body"] = extra_body_params

                stream = client.chat.completions.create(**stream_kwargs)

                with Live(
                    prog_streamer.make_status_line(""),
                    console=console,
                    refresh_per_second=8,
                    transient=True,
                ) as live:
                    for chunk in stream:
                        if chunk.choices and len(chunk.choices) > 0:
                            content = chunk.choices[0].delta.content or ""
                            if content:
                                if first_token_time is None:
                                    first_token_time = time.perf_counter()
                                full_response += content
                                if use_thinking:
                                    prog_streamer.step(live, format_thinking_markdown(full_response))
                                else:
                                    cleaned_live = strip_thinking_and_dedup(full_response)
                                    if cleaned_live:
                                        prog_streamer.step(live, cleaned_live)
                                    elif is_quant_query:
                                        live.update(
                                            prog_streamer.make_status_line(
                                                "Calculating intermediate equations...",
                                                prefix_label="⚡ Reasoning",
                                            )
                                        )

                        if hasattr(chunk, "usage") and chunk.usage:
                            completion_tokens = chunk.usage.completion_tokens

                final_disp = (
                    format_thinking_markdown(full_response)
                    if use_thinking
                    else strip_thinking_and_dedup(full_response)
                )
                prog_streamer.finish(final_disp)
                end_time = time.perf_counter()

            if completion_tokens == 0 and full_response:
                completion_tokens = max(1, int(len(full_response) / 4))

            # Metric Calculations
            ttft_ms = ((first_token_time or end_time) - start_time) * 1000
            gen_duration = end_time - (first_token_time or start_time)
            tps = completion_tokens / gen_duration if gen_duration > 0 else 0.0

            metrics_summary = (
                f"[dim]TTFT: [bold]{ttft_ms:.1f}ms[/bold] | "
                f"Gen Time: [bold]{gen_duration:.2f}s[/bold] | "
                f"Tokens: [bold]{completion_tokens}[/bold] | "
                f"Speed: [bold cyan]{tps:.1f} t/s[/bold cyan][/dim]"
            )
            console.print(Panel(metrics_summary, border_style="dim", expand=False))
            console.print()

        # --- RECORD CLEAN HISTORY TURN (WITH CORRUPTION & COT FILTERING) ---
        clean_resp = strip_thinking_and_dedup(full_response)

        # Discard responses under 10 chars or starting with orphan punctuation to prevent template corruption
        if len(clean_resp) > 10 and not clean_resp.startswith((",", ".", ")", "]")):
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": clean_resp})
        else:
            console.print("[bold red][Warning][/bold red] Discarded low-quality or corrupted assistant turn from history.")

        # --- ROLLING MEMORY COMPRESSION BUFFER (HIGH-WATER / LOW-WATER BATCH EVICTION) ---
        current_pairs = (len(conversation_history) - 1) // 2
        current_verbatim_tokens = estimate_history_tokens(conversation_history)

        if current_pairs > MAX_PAIRS or (current_verbatim_tokens > MAX_VERBATIM_TOKENS and current_pairs > 4):
            target_keep_pairs = max(4, MAX_PAIRS - EVICT_BATCH_PAIRS)
            evicted_batch: list[tuple[str, str]] = []
            while current_pairs > target_keep_pairs or (
                estimate_history_tokens(conversation_history) > (MAX_VERBATIM_TOKENS - 3000) and current_pairs > 4
            ):
                old_user_msg = conversation_history.pop(1)
                old_assistant_msg = conversation_history.pop(1)
                evicted_batch.append(
                    (old_user_msg.get("content", ""), old_assistant_msg.get("content", ""))
                )
                current_pairs = (len(conversation_history) - 1) // 2
                if len(evicted_batch) >= EVICT_BATCH_PAIRS and current_pairs <= MAX_PAIRS:
                    break

            if evicted_batch:
                summary_buffer = compress_memory_batch(summary_buffer, evicted_batch)

        # Silently persist cache and prompt history after each turn
        de_hydrate_cache(conversation_history, summary_buffer, DEFAULT_CACHE_FILE, silent=True)
        save_cmd_history()

    except KeyboardInterrupt:
        console.print(f"\n[yellow][Interrupted][/yellow] Auto-saving cache to '{DEFAULT_CACHE_FILE}'...")
        de_hydrate_cache(conversation_history, summary_buffer, DEFAULT_CACHE_FILE)
        save_cmd_history()
        break
    except Exception as e:
        console.print(f"\n[bold red][Error][/bold red] Communication failure: {e}")
        break
