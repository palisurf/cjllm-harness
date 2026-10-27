# CJ vLLM Search Harness (`cjllm-search-harness`)

A lightweight, single-file, zero-framework Python CLI inference and search-augmented generation harness ([`cjllm.py`](./cjllm.py)) built for **OpenAI-compatible `vLLM` endpoints** running on local AI workstations and accelerator appliances (including **Tenstorrent QuietBox 2 (`p300x2` Blackhole)** and **NVIDIA DGX Spark**).

**Author:** Christopher Jones ([@palisurf](https://github.com/palisurf))  
**Repository:** [https://github.com/palisurf/cjllm-harness](https://github.com/palisurf/cjllm-harness)

---

## 1. Introduction & Motivation

Running frontier open-weight LLMs locally on dedicated hardware (`vLLM` on Tenstorrent Blackhole/Wormhole accelerators or NVIDIA DGX/CUDA systems) provides raw token throughput and total data privacy, but exposes a gap between **serving a model** and **having a reliable daily driver**:

1. **Raw `vLLM` Endpoints Are Stateless and Cut Off in Time:** A bare `/v1/chat/completions` endpoint has no awareness of the current date/weekday, no access to live market quotes, weather, or ocean buoy telemetry, and no rolling memory management once a conversation outgrows the sliding window.
2. **Tokenizer & Reasoning-Template Friction:** Serving distilled reasoning checkpoints under hardware-optimized server profiles (such as running `DeepSeek-R1-Distill-Qwen-32B` weights under a `Qwen/Qwen3-32B` `tt-inference-server` specification) introduces special-token ID mismatches (`<|box_start|>` / `<|box_end|>` vs. `<think>` / `</think>`). Naive OpenAI clients suffer from leaked Chain-of-Thought monologues, doubled responses, or wasted reasoning latency on simple factual turns.
3. **Shallow Search in Existing Local Frontends:** Most local LLM wrappers either require deploying heavy multi-container web stacks (Node.js, Postgres, Redis, vector DBs) or rely on basic DuckDuckGo/Serper meta-description snippets (~150 characters). Those snippets strip out the actual numbers from live stock indices (`Dow`, `S&P 500`, `Nasdaq`), multi-day weather tables, surf swell forecasts, and NOAA NDBC buoy station tables—causing local models to either refuse or hallucinate figures.

### Intent of `cjllm-search-harness`
`cjllm-search-harness` was created to solve all three problems in **a single, self-contained Python script ([`cjllm.py`](./cjllm.py))** with zero external server dependencies, zero paid search API keys, and identical behavior across **Tenstorrent QuietBox 2** and **NVIDIA DGX Spark**:

* **Drop-in Terminal Ergonomics:** Runs immediately over SSH or locally (`./run_chat`) with full GNU `readline` history, live Markdown streaming, and instant slash commands (`/think`, `/v`, `/save`, `/load`, `/clear`).
* **Runtime Model & Tokenizer Auto-Adaptation:** Probes `/v1/models` and token behavior at startup (~100ms) to automatically configure stop tokens and reasoning prefills for `DeepSeek-R1-Distill` (`deepseek_qwen3_bridge`), native reasoning models (`native_think`), or standard instruct models (`standard_instruct`).
* **Deep, Keyless Web & Table Grounding:** Combines a fast LLM search-intent router with a 4-tier scraper that pulls live Yahoo Finance multi-index quote cards, 7-day weather forecasts, up to 6 structured HTML `<table>` blocks per page, and direct NOAA NDBC station telemetry (`ndbc.noaa.gov`) alongside organic and news search results.
* **Built-In Hardware Telemetry:** Every response reports real-time **Time-To-First-Token (`TTFT ms`)**, **Generation Time (`s`)**, **Output Tokens**, and **Decode Throughput (`t/s`)**, making it both a daily assistant and an interactive hardware benchmark harness.

---

## 2. Comparison Against Other Local LLM Harnesses

| Capability | `cjllm-search-harness` (`cjllm.py`) | Raw `vLLM` / Basic OpenAI CLI | Ollama CLI (`ollama run`) | Open WebUI | Perplexica / Local Search UIs |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Deployment Footprint** | **Single Python file** (`pip install -r requirements.txt`) | Single script | Go daemon + CLI | Heavy Docker/Node/DB stack | Multi-container Docker + SearXNG |
| **Model & Tokenizer Auto-Detection** | **Yes** (`deepseek_qwen3_bridge`, `native_think`, `standard_instruct`) | No (manual stop tokens & templates) | Bundled Modelfile only | Manual per-model configuration | Manual configuration |
| **Fast `OFF` Mode for Reasoning Models** | **Yes** (Closed micro-thought prefill skips CoT & cuts latency ~50%; toggle `/think` anytime) | No (always runs full CoT monologue) | Limited (`--think` on supported tags only) | Displays CoT, rarely suppresses generation | No |
| **Live Search Without Paid API Keys** | **Yes** (Keyless `ddgs` + `primp` browser-impersonated scraping) | None | None built-in | Basic snippet search or paid APIs | Requires local SearXNG container |
| **Live Financial & Multi-Index Quotes** | **Yes** (Auto-expands `Dow`, `S&P 500`, `Nasdaq` & ticker quote cards) | None | None | Meta-description blurbs only | Meta-description blurbs only |
| **Structured HTML `<table>` & NOAA Buoy Extraction** | **Yes** (Extracts up to 6 HTML tables/page + auto-resolves 5-digit NOAA NDBC buoy IDs) | None | None | Full-page text dump (loses table structure) | Text chunks only |
| **Context-Aware Search Router** | **Yes** (Resolves pronouns & follow-up narrowing like *"for today only instead"*) | None | None | Basic query rewrite | Yes (higher latency multi-agent chain) |
| **Memory Management** | **Sliding verbatim window (`MAX_PAIRS`) + LLM rolling summary buffer + KV prefix-cache friendly** | Manual / unbounded | Simple context truncation | Full history / vector RAG | Per-thread history |
| **Crash-Safe Session Persistence** | **Yes** (Auto-saves `chat_cache.json` & `.chat_prompt_history` after every turn) | None | None | Database-backed | Database-backed |
| **Per-Turn Hardware Telemetry** | **Yes** (`TTFT ms`, `Gen Time s`, `Tokens`, `Speed t/s`, `/v` payload inspector) | None | `--verbose` summary at exit | Basic token speed | None |

---

## 3. Current Capabilities & Architecture

```
User Input
   │
   ├──► Slash Command Handler (/think, /v, /clear, /save, /load, chat_end)
   │
   ▼
1. Search Intent Classifier (`determine_search_query`)
   • Injects current system weekday & date (prevents training-cutoff year hallucinations)
   • Passes recent conversation turns to resolve pronouns ("they", "it") and follow-ups ("today only")
   • Uses non-thinking fast prefill (30-token budget → "NONE" or concise 2–12 word search query)
   • Strips preamble labels and rejects leaked reasoning sentences before hitting search engines
   │
   ├──► [If Search Needed] Multi-Tier Web Context Builder (`fetch_web_context`)
   │       ├─ Tier 1: Yahoo Search Rich Quote, Multi-Index (`^DJI`, `^GSPC`, `^IXIC`), Weather & Fact Cards
   │       ├─ Tier 2: Multi-Table HTML `<table>` Extractor (up to 6 tables/page + NOAA NDBC station resolver)
   │       ├─ Tier 3: Cleaned Organic Web Snippets (`DDGS().text()`)
   │       └─ Tier 4: Recent News Articles (`DDGS().news()`)
   │
   ▼
2. Payload Construction & Model-Profile Adaptation (`prepare_request_messages_and_extra`)
   • Single merged system prompt (`BASE_SYSTEM_PROMPT` + `summary_buffer`)
   • Verbatim sliding window (`MAX_PAIRS = 10` recent user/assistant turns)
   • Active user turn (with injected live web data and anti-hallucination guardrails)
   • Automatic tokenizer prefill & stop-token adaptation based on `MODEL_PROFILE`
   │
   ▼
3. Streaming Generation & Live Telemetry
   • Streams Markdown via `rich.live.Live`
   • Default `think_mode = OFF`: suppresses CoT via completed micro-thought prefill + live stream sanitizer
   • Optional `/think` (`ON`): streams Chain-of-Thought inside a formatted quote block separated from the answer
   • Computes and displays `TTFT (ms)`, `Gen Time (s)`, `Tokens`, and `Speed (t/s)`
   │
   ▼
4. Sanitization, Rolling Memory Compression & Auto-Save
   • Strips CoT blocks and residual special tokens (`strip_thinking_and_dedup`)
   • Quality gate discards degenerate/corrupted responses before they pollute history
   • Evicts oldest turn pairs beyond `MAX_PAIRS` into `summary_buffer` (`compress_memory`)
   • Auto-saves `chat_cache.json` and `.chat_prompt_history` after every completed turn
```

### Detailed Feature Breakdown

#### A. DeepSeek-R1-Distill / Qwen3 Tokenizer Bridge & Leak-Free Non-Thinking Default
When serving `DeepSeek-R1-Distill-Qwen-32B` weights under a `Qwen/Qwen3-32B` model specification, the server loads `Qwen/Qwen3-32B`'s tokenizer from HuggingFace while executing DeepSeek's distilled weights:
* Token ID `151648` (`<think>` in DeepSeek) maps to `"<|box_start|>"` in the Qwen3 tokenizer.
* Token ID `151649` (`</think>` in DeepSeek) maps to `"<|box_end|>"` in the Qwen3 tokenizer.
* Token ID `151643` (`<｜end▁of▁sentence｜>` in DeepSeek) maps to `"<|endoftext|>"` in the Qwen3 tokenizer.

**How [`cjllm.py`](./cjllm.py) handles this automatically:**
* **Fast Default Mode (`think_mode = False`):** Prefills the assistant turn with a completed 1-sentence micro-thought (`"<|box_start|>I will answer the user request directly and concisely.<|box_end|>\n"`, `continue_final_message=True`, `add_generation_prompt=False`) and sets `stop_token_ids=[151649, 151643]`, while applying `strip_thinking_and_dedup` during live streaming. Satisfying the model's CoT prior with a completed thought prevents unmarked reasoning leaks and cuts response latency in half.
* **Reasoning Mode (`/think`):** Prefills `"<|box_start|>\n"` with `skip_special_tokens=False`, streams the live Chain-of-Thought inside a `💭 Thinking (Chain-of-Thought)` Markdown block, splits the final answer at `"<|box_end|>"`, and strips the CoT before saving to `conversation_history`.

#### B. 100% Model-Agnostic Startup Auto-Detection (`detect_model_profile`)
At startup, the harness queries `/v1/models` to auto-discover the active model ID and `max_model_len`, then runs a fast (~100ms, 3-token) probe to classify the server into one of three profiles:
1. **`deepseek_qwen3_bridge`:** Engaged when the model closes `<|box_start|>` with `<|box_end|>` (`DeepSeek-R1-Distill-Qwen-32B` on `Qwen/Qwen3-32B`).
2. **`native_think`:** Engaged when the model uses literal `<think>` / `</think>` tokens (native `DeepSeek-R1`, `Qwen3`, `QwQ`).
3. **`standard_instruct`:** Engaged for non-reasoning instruct models (`Llama-3.1-70B-Instruct`, `Llama-3.1-8B-Instruct`, `Mistral`, `Gemma`). Sends standard chat completion payloads while still supporting `/think` via prompt-guided `<think>...</think>` tags.

#### C. Context-Aware & Date-Grounded Search Intent Router (`determine_search_query`)
* **Weekday + System Clock Injection:** Injects `Today's date is <Weekday, Month Day, Year>` into the search classifier, web context header, and `BASE_SYSTEM_PROMPT` so relative time queries (*"today"*, *"this week"*, *"latest"*) always anchor to the real system clock.
* **Follow-Up & Pronoun Resolution:** Uses recent user/assistant turns to turn conversational follow-ups (e.g., *"Ok for today only instead"* after a surf forecast query, or *"What are their github repos?"*) into self-contained search queries.
* **Reasoning-Leak Guardrails:** Strips any `"Search Query:"` labels and rejects >12-word reasoning sentences so DuckDuckGo and Yahoo only receive clean keyword queries.

#### D. 4-Tier Live Web, Market, Weather, Surf & Buoy Extractor (`fetch_web_context`)
1. **Live Market, Multi-Index, Weather & Fact Cards (`fetch_yahoo_rich_cards`):**
   * **Stock & Multi-Index Quotes:** Extracts live prices, point/percentage changes, market cap, P/E, and 52-week ranges. Queries mentioning multiple market indices (*"What happened on the DOW, S&P, and Nasdaq today?"*) automatically expand to fetch all three live index cards (`^DJI`, `^GSPC`, `^IXIC`) in parallel.
   * **Current Weather & 7-Day Forecast:** Extracts current conditions and daily high/low temperatures with condition labels.
   * **Structured Entity Fact Tables:** Extracts embedded financial and company metrics.
2. **Multi-Table HTML `<table>` & NOAA Buoy Scraper (`fetch_top_page_table_or_excerpt`):**
   * Prioritizes tabular domains (`ndbc.noaa.gov`, `swellinfo.com`, `deepswell.com`, `surf-forecast.com`, `companiesmarketcap.com`, `stockanalysis.com`, `wikipedia.org`, `slickcharts.com`, `finance.yahoo.com`).
   * Automatically detects 5-digit **NOAA NDBC buoy station IDs** (e.g., `46221` Santa Monica Bay, `46025` Santa Monica Basin) and directly fetches `https://www.ndbc.noaa.gov/station_page.php?station=<ID>`, extracting up to **6 HTML `<table>` blocks per page** (covering real-time wave summary, swell components, wind wave period, and 15-minute meteorological readings).
3. **Filtered Organic Snippets (`DDGS().text()`):** Up to 5 cleaned organic web snippets.
4. **Recent News Snippets (`DDGS().news()`):** Up to 3 dated news items for breaking events and market catalysts.

#### E. Rolling Summary-Buffer Memory, Prefix Caching & Crash-Safe Auto-Save
* Retains up to `MAX_PAIRS` (default `10`) verbatim user/assistant turn pairs for highKV prefix-cache hit rates across both **NVIDIA DGX Spark** and **Tenstorrent QuietBox 2** (see [`TT_PREFIX_CACHE_FIX.md`](./TT_PREFIX_CACHE_FIX.md) for details on the `tt_transformers` `PR #55630` page-table fix on `P300X2`).
* Folds evicted turn pairs into `summary_buffer` via a fast non-thinking summarization pass (`compress_memory`).
* Automatically persists `chat_cache.json` and `.chat_prompt_history` after every completed turn so no context is lost if an SSH session drops.

---

## 4. Usage & Commands

### Installation & Dependencies
```bash
pip install -r requirements.txt
```

### Starting the Harness
```bash
# Run via convenience script or directly with automatic model & tokenizer detection
./run_chat
# or:
python3 cjllm.py
```

### Optional CLI Arguments
```bash
python3 cjllm.py \
  --url http://localhost:8000/v1 \
  --model Qwen/Qwen3-32B \
  --max-pairs 10 \
  --cache chat_cache.json \
  --think
```

### Interactive In-Session Commands
| Command | Description |
| :--- | :--- |
| `/think` | Toggle Chain-of-Thought reasoning mode ON/OFF (default: `OFF` for fast direct answers) |
| `/think <query>` | Run a single query with Chain-of-Thought reasoning enabled |
| `/v` | Toggle verbose mode ON/OFF (displays the exact JSON payload sent to `vLLM`) |
| `/v <query>` | Run a single query with verbose JSON payload display forced |
| `/clear` or `/reset` | Flush conversation history and rolling summary buffer |
| `/save [filename]` | Save current conversation history and summary buffer to JSON |
| `/load <filename>` | Load conversation history and summary buffer from JSON |
| `chat_end` / `exit` / `quit` | Save session to `chat_cache.json` and exit |

---

## 5. Project File Inventory (`cjllm-search-harness`)

| File | Purpose |
| :--- | :--- |
| [`cjllm.py`](./cjllm.py) | Primary model-agnostic `vLLM` inference harness with tokenizer bridge, live quote/weather/table/buoy scraper, and `/think` mode |
| [`run_chat`](./run_chat) | Convenience launcher for `cjllm.py` (passes through any CLI arguments) |
| [`requirements.txt`](./requirements.txt) | Python package dependencies (`openai`, `rich`, `ddgs`, `primp`) |
| [`TT_PREFIX_CACHE_FIX.md`](./TT_PREFIX_CACHE_FIX.md) | Technical note on the `tt_transformers` (`PR #55630`) untraced prefix-cache page-table patch for Tenstorrent QuietBox 2 (`P300X2`) |
| [`.gitignore`](./.gitignore) | Git ignore rules (`__pycache__/`, `*.swp`, `.chat_prompt_history`, `chat_cache.json`) |
| [`chat_cache.json`](./chat_cache.json) | Default persisted conversation history and rolling summary buffer |
| [`README.md`](./README.md) | Motivation, harness comparison, architecture, feature reference, and usage documentation |
