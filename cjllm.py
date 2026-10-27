"""
cjllm.py — CJ vLLM Search & Inference Harness (cjllm-search-harness)

Author: Christopher Jones (@palisurf)
Repository: https://github.com/palisurf/cjllm-harness

A single-file, model-agnostic vLLM inference and search-augmented generation harness
featuring automatic reasoning/tokenizer profile detection, multi-tier live web/quote/weather/table
extraction, rolling summary-buffer memory compression, and live TTFT/TPS telemetry.
"""

import argparse
from datetime import datetime, timezone
import html
import json
import os
import re
import readline
import time
import warnings
from urllib.parse import quote_plus
from openai import OpenAI
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

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
    default=10,
    help="Number of verbatim user/assistant turn pairs to retain before rolling compression (default: 10)",
)
parser.add_argument(
    "--cache",
    default="chat_cache.json",
    help="Default session cache file (default: chat_cache.json)",
)
parser.add_argument(
    "--think",
    action="store_true",
    help="Start session with Chain-of-Thought reasoning mode enabled",
)
args, _ = parser.parse_known_args()

# Initialize OpenAI client pointing to local inference engine
client = OpenAI(
    base_url=args.url,
    api_key=args.api_key,
)

DEFAULT_CACHE_FILE = args.cache
COMMAND_HISTORY_FILE = ".chat_prompt_history"
MAX_PAIRS = args.max_pairs

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


def fetch_top_page_table_or_excerpt(text_results: list[dict], query: str = "", max_chars: int = 2800) -> str:
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
            for tag in ("script", "style", "svg", "noscript", "header", "footer", "nav"):
                raw = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", " ", raw, flags=re.S | re.I)

            # Pass 1: Extract structured HTML <table> rows across up to 6 tables per page
            tables = re.findall(r"<table[^>]*>(.*?)</table>", raw, flags=re.S | re.I)
            if tables:
                rows_out = []
                for tbl in tables[:6]:
                    tbl_rows = 0
                    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", tbl, flags=re.S | re.I):
                        cells = [
                            clean_text(re.sub(r"<[^>]+>", " ", cell))
                            for cell in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, flags=re.S | re.I)
                        ]
                        cells = [c for c in cells if c]
                        if len(cells) >= 2:
                            row_line = " | ".join(cells)
                            if any(
                                skip_phrase in row_line
                                for skip_phrase in ("Unit of Measure:", "Change units", "Click on the graph")
                            ):
                                continue
                            rows_out.append(row_line)
                            tbl_rows += 1
                        if tbl_rows >= 10 or len(rows_out) >= 40:
                            break
                    if len(rows_out) >= 40:
                        break
                if len(rows_out) >= 3:
                    table_text = "\n".join(rows_out)[: max_chars // 2 if extracted_blocks else max_chars]
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
    Returns (today_date_str, local_clock_str, utc_clock_str) so the model and search router
    always know both the user's local timezone clock and the current UTC/GMT clock (critical
    for NOAA NDBC buoy timestamps and financial/market close comparisons).
    """
    now_local = datetime.now().astimezone()
    now_utc = datetime.now(timezone.utc)
    today_str = now_local.strftime("%A, %B %d, %Y")
    local_clock_str = now_local.strftime("%A, %B %d, %Y %I:%M %p %Z")
    utc_clock_str = now_utc.strftime("%Y-%m-%d %H:%M UTC (GMT)")
    return today_str, local_clock_str, utc_clock_str


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


def determine_search_query(user_input: str, history: list) -> str | None:
    """
    Fast, context-aware classification pass:
    - Deterministically skips pure coding/math/physics prompts that don't request live external lookup.
    - Injects today's system date so 'current'/'today'/'latest' queries use the real year (not training cutoff).
    - Uses up to the last 3 conversation turn pairs (600 chars/turn) + extracted station/ticker IDs
      to resolve multi-hop pronouns ('that buoy', 'the second one', 'their', 'it').
    - Automatically adapts to the active MODEL_PROFILE (skipping CoT when supported).
    """
    u_lower = user_input.lower().strip()

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
                    "- Use Recent Conversation Context to resolve pronouns, implicit topic references (e.g., 'for today only instead', 'that buoy', 'the second one'), or when previous turns failed to provide the requested data.\n"
                    "- When Recent Conversation Context contains a specific 5-digit NOAA buoy station ID (e.g., 46221, 46054, 46218) that matches the buoy referenced by the user, ALWAYS include that exact 5-digit station ID (e.g. 'NOAA buoy 46054 wave height water temp') in your search query.\n"
                    f"- When the user asks for 'current', 'latest', or 'today', use the current date/year ({today_str}) in your search query (NEVER use 2023 or 2024).\n"
                    "- If the user asks to pull data, search, update, verify, or narrow a previous real-time data request (such as 'for today only instead'), ALWAYS output a search query combining the topic from Recent Conversation Context with the Latest User Prompt.\n"
                    "- If search IS NOT needed (math, coding, reasoning, established general knowledge, or follow-up analysis of facts already stated), reply ONLY with 'NONE'.\n"
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
            return None

        # Guard against leaked reasoning sentences being sent to DuckDuckGo
        lower_q = res_text.lower()
        if len(res_text.split()) > 12 or any(
            phrase in lower_q
            for phrase in ("to determine", "i'll analyze", "i will analyze", "the user is asking", "user's request")
        ):
            cleaned_fallback = re.sub(
                r"\b(show me|the|with|for|create|a|table|this|please|format|in|go|and|pull|information|from|publically|available|data|on)\b",
                " ",
                user_input,
                flags=re.I,
            )
            cleaned_fallback = re.sub(r"\s+", " ", cleaned_fallback).strip()
            res_text = cleaned_fallback[:80] if cleaned_fallback else user_input[:80]

        res_text = res_text.replace('"', "").replace("'", "")

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


def compress_memory(existing_summary: str, old_user: str, old_assistant: str) -> str:
    """
    Uses a non-thinking completion pass to fold overflowing turn pairs into the summary buffer
    without leaking Chain-of-Thought into the system prompt.
    """
    prompt = (
        f"Existing Conversation Summary:\n{existing_summary if existing_summary else 'None'}\n\n"
        f"Turn to Integrate:\nUser: {old_user}\nAssistant Answer: {old_assistant}\n\n"
        "Provide an updated, concise summary (2-4 sentences max) capturing key entities, user requirements, "
        "and core conclusions. Do NOT include meta-commentary, disclaimers, or web search boilerplate."
    )
    try:
        console.print("[dim]Updating rolling memory summary buffer...[/dim]")
        base_messages = [
            {"role": "system", "content": "You are a concise conversation memory summarizer."},
            {"role": "user", "content": prompt},
        ]
        req_messages, extra_params = prepare_request_messages_and_extra(
            base_messages, use_thinking=False, is_utility_call=True
        )
        create_kwargs = {
            "model": MODEL_NAME,
            "messages": req_messages,
            "temperature": 0.0,
            "max_tokens": 400,
        }
        if extra_params:
            create_kwargs["extra_body"] = extra_params

        response = client.chat.completions.create(**create_kwargs)
        cleaned = strip_thinking_and_dedup(response.choices[0].message.content or "")
        if cleaned:
            return cleaned
    except Exception:
        pass

    entry = f"User asked: '{old_user}' | Assistant answered: '{old_assistant[:120]}...'"
    return f"{existing_summary}\n- {entry}" if existing_summary else f"- {entry}"


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
BASE_SYSTEM_PROMPT = (
    f"You are an intelligent, concise AI assistant. Today's date is {CURRENT_DATE_STR} "
    f"(Local Time: {CURRENT_LOCAL_CLOCK_STR} | UTC/GMT Time: {CURRENT_UTC_CLOCK_STR}). "
    "Prioritize answering the user's latest message directly and accurately, "
    "while using prior conversation turns to resolve follow-up references and maintain continuity. "
    "Treat the Prior Conversation Summary Buffer as background context. "
    "When web search data is provided, always prefer the live retrieved figures and tables over pre-training knowledge. "
    "NEVER invent, estimate, or generate placeholder tables or figures when real-time data is requested; "
    "report only the exact measurements present in the retrieved web data."
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
console.print(f"  [cyan]Max Pairs Configured[/cyan] : Retaining {MAX_PAIRS} verbatim pairs before rolling buffer compression")
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
        if user_input.lower() in ["chat_end", "exit", "quit"]:
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

        # Context-Aware Hybrid Search Classification Step
        console.print("[dim]Analyzing search intent...[/dim]", end="\r")
        search_query = determine_search_query(user_input, conversation_history)

        if search_query:
            console.print(f"[dim]Searching web (max 5) for: '[bold cyan]{search_query}[/bold cyan]'...[/dim]")
            web_payload = fetch_web_context(search_query, max_results=5)
            active_user_content = (
                f"{web_payload}\n"
                "### INSTRUCTIONS:\n"
                "Answer the Active User Request below using the web search data above where relevant, "
                "while maintaining continuity with our recent conversation history.\n\n"
                f"### ACTIVE USER REQUEST:\n{user_input}"
            )
        else:
            console.print("[dim]Direct response (no web search needed).[/dim]")
            active_user_content = user_input

        # --- CONSTRUCT VLLM PAYLOAD ARRAY ---
        # 1. Merge Base System Prompt & Summary Buffer into a Single System Message
        if summary_buffer:
            combined_system_content = (
                f"{BASE_SYSTEM_PROMPT}\n\n"
                f"Prior Conversation Summary Buffer (Background Context):\n{summary_buffer}"
            )
        else:
            combined_system_content = BASE_SYSTEM_PROMPT

        base_payload = [{"role": "system", "content": combined_system_content}]

        # 2. Verbatim Recent Pairs
        base_payload.extend(conversation_history[1:])

        # 3. Active User Turn
        base_payload.append({"role": "user", "content": active_user_content})

        # 4. Adapt payload & extra_body parameters to the auto-detected MODEL_PROFILE
        active_payload, extra_body_params = prepare_request_messages_and_extra(
            base_payload, use_thinking=use_thinking, is_utility_call=False
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

        # Performance Metric Counters
        start_time = time.perf_counter()
        first_token_time = None
        full_response = ""
        completion_tokens = 0

        stream_kwargs = {
            "model": MODEL_NAME,
            "messages": active_payload,
            "temperature": 0.6 if use_thinking else 0.7,
            "max_tokens": MODEL_PROFILE["max_gen_tokens"],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if extra_body_params:
            stream_kwargs["extra_body"] = extra_body_params

        stream = client.chat.completions.create(**stream_kwargs)

        with Live(Markdown(""), console=console, refresh_per_second=15, vertical_overflow="visible") as live:
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    content = chunk.choices[0].delta.content or ""
                    if content:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        full_response += content
                        if use_thinking:
                            live.update(Markdown(format_thinking_markdown(full_response)))
                        else:
                            live.update(Markdown(strip_thinking_and_dedup(full_response)))

                if hasattr(chunk, "usage") and chunk.usage:
                    completion_tokens = chunk.usage.completion_tokens

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

        # --- ROLLING MEMORY COMPRESSION BUFFER ---
        current_pairs = (len(conversation_history) - 1) // 2

        while current_pairs > MAX_PAIRS:
            old_user_msg = conversation_history.pop(1)
            old_assistant_msg = conversation_history.pop(1)

            summary_buffer = compress_memory(
                summary_buffer,
                old_user_msg.get("content", ""),
                old_assistant_msg.get("content", ""),
            )

            current_pairs = (len(conversation_history) - 1) // 2

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
