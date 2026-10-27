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
| **Memory Management** | **30-pair (`~20K` token) verbatim window + 8-pair batch eviction into numerical summary buffer (KV prefix-cache friendly)** | Manual / unbounded | Simple context truncation | Full history / vector RAG | Per-thread history |
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
   • Verbatim sliding window (`MAX_PAIRS = 30` pairs / `~20,000` tokens high-water mark, batch-evicting `8` pairs)
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

#### C. Built-In Current Time, Date & World Clock Skill (`build_live_clock_skill_context`) + Context-Aware Search Router (`determine_search_query`)
* **Built-In Current Time, Calendar & World Clock Skill (`is_clock_skill_query` / `build_live_clock_skill_context`):**
  * Automatically resolves the user's local IANA timezone (`--tz` or `CJLLM_TZ`, defaulting to `America/Los_Angeles` `PDT`/`PST` when running on headless `Etc/UTC` Linux servers/containers so evening queries after 5:00 PM Pacific never roll forward into tomorrow's UTC date).
  * Intercepts queries asking for the current time, date, day of the week (*"What time is it now"*, *"What day of the week is tomorrow"*), or world timezones (*"What time is it in Tokyo and London right now?"*) and injects an authoritative `[SKILL: LIVE SYSTEM CLOCK & CALENDAR]` block into the active turn (`messages[-1]`) down to the second—without mutating `messages[0]` (preserving 100% KV prefix-cache hit rates) and overriding any older timestamps in conversation history.
* **Dual Local & UTC/GMT Clock Grounding (`get_dual_clock_str`):** Injects both the local system clock (`PDT/PST`) and the current `UTC/GMT` clock (in both 12-hour and 24-hour formats) into the search classifier and web context header so UTC-stamped marine/aviation reports (like NOAA buoys crossing UTC midnight) are always converted and subtracted accurately.
* **3-Turn Multi-Hop & Entity Resolution:** Inspects up to 3 conversation turn pairs (`600` chars/turn) and automatically indexes 5-digit NOAA buoy station IDs and location clauses from recent history so multi-hop follow-ups (*"Pull the readings for that buoy"*, *"the second one"*, *"for today only instead"*) preserve the exact station ID or ticker.
* **Pure Code/Math False-Positive Guard & Reasoning-Leak Filter:** Deterministically skips web searches on pure coding, circuit physics, and calculus prompts containing words like *"current"* or *"today"*, while stripping `"Search Query:"` labels and rejecting >12-word reasoning sentences.

#### D. 4-Tier Live Web, Market, Weather, Surf & Buoy Extractor (`fetch_web_context`)
1. **Live Market, Multi-Ticker, Macro-Asset, Multi-City Weather & Fact Cards (`fetch_yahoo_rich_cards`):**
   * **Stock, Multi-Index & Macro Quotes:** Automatically expands queries mentioning multiple US indices (`^DJI`, `^GSPC`, `^IXIC`, `^RUT`), macro assets (`Bitcoin`, `Ethereum`, `Gold`, `10-Year Treasury Yield`), or multiple individual stock tickers (*"Compare NVDA, AAPL, and AMD today"*) into parallel live Yahoo Finance quote card lookups.
   * **Multi-City Current Weather & 7-Day Forecast:** Automatically splits multi-location weather comparisons (*"Compare weather in Malibu, CA vs. Honolulu, HI"*) so each city gets its own live current conditions and 7-day forecast card.
   * **Structured Entity Fact Tables:** Extracts embedded financial and company metrics.
2. **Two-Pass Multi-Table HTML `<table>` & NOAA Buoy Scraper (`fetch_top_page_table_or_excerpt`):**
   * **Pass 1 (Tabular Priority):** Scans up to 4 candidate URLs across prioritized tabular domains (`ndbc.noaa.gov`, `weather.gov`, `swellinfo.com`, `deepswell.com`, `surf-forecast.com`, `tideschart.com`, `companiesmarketcap.com`, `stockanalysis.com`, `wikipedia.org`, `slickcharts.com`, `finance.yahoo.com`) and extracts up to **6 HTML `<table>` blocks per page**, ensuring a non-tabular rank-1 article never shadows a real table at rank 2–4.
   * **Automatic NOAA Buoy Resolution:** Detects 5-digit **NOAA NDBC buoy station IDs** (`46221`, `46054`, `46025`, etc.) in either the search query or snippets and directly fetches `https://www.ndbc.noaa.gov/station_page.php?station=<ID>`.
   * **Pass 2 (Plain-Text Fallback):** Falls back to a cleaned webpage excerpt only when no candidate URL contains a valid HTML `<table>`.
3. **Filtered Organic Snippets (`DDGS().text()`):** Up to 5 cleaned organic web snippets.
4. **Recent News Snippets (`DDGS().news()`):** Up to 3 dated news items for breaking events and market catalysts.

#### E. Quantitative Auto-Think Escalation, High-Precision Memory & Crash-Safe Auto-Save
* **Quantitative Reasoning Escalation (`needs_quantitative_reasoning`):** When `think_mode = OFF` (default), prompts requiring multi-step math, share allocations, timezone deltas, or physics formulas (such as deep-water wavelength $L = \frac{gT^2}{2\pi}$) automatically enable internal `<|box_start|>` reasoning and instruct the model to write out intermediate equations, maintaining full mathematical accuracy.
* **High-Water (`MAX_PAIRS = 30` / `~20,000` Tokens) + 8-Pair Batch-Eviction Memory Architecture (`compress_memory_batch`):**
  * **Why 1-by-1 Sliding Windows Thrash the KV Prefix Cache:** In a naive sliding window (e.g., evicting 1 turn pair on every turn after Turn 10), folding the evicted turn into `summary_buffer` mutates `messages[0]` (the system prompt) on **every single turn** once the limit is reached. Because `vLLM` prefix caching hashes token blocks sequentially from token `0`, mutating `messages[0]` invalidates **100% of the cached KV blocks** on every subsequent turn.
  * **The 30-Pair / 8-Pair Hysteresis Solution:** [`cjllm.py`](./cjllm.py) retains up to **`MAX_PAIRS = 30` verbatim turn pairs** (or `MAX_VERBATIM_TOKENS = 20000` tokens) before triggering compression. When the conversation crosses that high-water mark, [`compress_memory_batch`](./cjllm.py) evicts the **oldest `8` turn pairs at once in a single LLM call** (dropping the verbatim window from `30` down to `22` pairs). For the next 8 turns (`22 → 30`), `messages[0]` remains 100% byte-identical, preserving full KV prefix-cache hit rates across both **NVIDIA DGX Spark** and **Tenstorrent QuietBox 2** (see [`TT_PREFIX_CACHE_FIX.md`](./TT_PREFIX_CACHE_FIX.md)).
  * **High-Precision Numerical Archivist Prompt:** Uses a 700-token structured archivist prompt that preserves every exact price, percentage, wave height, period, temperature, ticker, and buoy station ID verbatim when folding evicted turns into `summary_buffer`.
* **Crash-Safe Persistence:** Automatically saves `chat_cache.json` and `.chat_prompt_history` after every completed turn.

#### F. "Lost in the Middle" Multi-Needle Verification (`30` Turn Pairs / `~18,567` Tokens)
To verify that extending the verbatim window to `MAX_PAIRS = 30` (`~18.6K` tokens) does not introduce "Lost in the Middle" attention dilution on `DeepSeek-R1-Distill-Qwen-32B` (`P300X2`), the harness was benchmarked against a 30-turn-pair (`60`-message, `~18,567`-token) conversation packed with **24 dense technical distractor turns** (containing dozens of competing numbers, byte counts, and dollar figures) and **12 specific numerical needles** planted across 6 conversation depths (`7%`, `30%`, `50% Exact Middle`, `53% Exact Middle`, `77%`, and `97%`):

| Conversation Depth | Turn Pair | Planted Needle | Model Recall Output at `~18,567` Tokens | Result |
| :--- | :--- | :--- | :--- | :--- |
| **7% (Early)** | Pair 2 | NOAA Buoy `46054` gust **`27.4 knots`**, water temp **`58.3 °F`** | `Peak wind gust of 27.4 knots, water temperature of 58.3°F` | **PASS (2/2)** |
| **30% (Early-Mid)** | Pair 9 | 4-worker LUT QC-LDPC FER **`1.42e-5`**, $E_b/N_0$ **`3.75 dB`** | `FER of 1.42e-5, Eb/N0 of 3.75 dB` | **PASS (2/2)** |
| **50% (Exact Middle)** | Pair 15 | Broadcom (`AVGO`) **`137 shares`** @ **`$1,084.60`** | `Limit order for 137 shares at $1,084.60 per share` | **PASS (2/2)** |
| **53% (Exact Middle)** | Pair 16 | Malibu Surfrider low tide **`-0.8 ft`** @ **`3:42 PM PDT`** | `Afternoon low tide of -0.8 ft at 3:42 PM PDT` | **PASS (2/2)** |
| **77% (Late-Mid)** | Pair 23 | `P300X2` `l1_small_size` **`24,576`** B, `trace_region_size` **`402,653,184`** B | `l1_small_size of 24,576 bytes, trace_region_size of 402,653,184 bytes` | **PASS (2/2)** |
| **97% (Recent)** | Pair 29 | Generac (`GNRC`) close **`$162.85`**, P/E **`23.40`** | `Closing price of $162.85, P/E ratio of 23.40` | **PASS (2/2)** |
| **Post-Turn-31 Overflow** | Evicted Pairs 1–8 | Batch-evict oldest 8 pairs (`31 → 23` pairs) into `summary_buffer` | Retained `NOAA Buoy 46054: Peak wind gust of 27.4 knots (NW), sea surface temp 58.3°F` verbatim | **PASS** |

* **Total Multi-Needle Recall Score:** **`12 / 12 (100.0%)`** across all conversation depths (`7%` through `97%`) with zero attention dilution.

#### G. Automatic Multi-Slot Concurrency Acceleration (Speculative Router Racing & Section-Parallel Map-Reduce)
Because LLM decode on both **Tenstorrent QuietBox 2 (`P300X2`, 32 slots)** and **NVIDIA DGX Spark (`GB10`, 128+ slots)** is memory-bandwidth bound at `batch = 1`, `vLLM` continuous batching can decode multiple concurrent streams in a single forward pass by reading the 32B model weights from memory once per step. [`cjllm.py`](./cjllm.py) exploits those parallel slots **100% automatically with zero slash commands**:

1. **Automatic Speculative Router + Direct-Answer Racing (`race_router_and_speculative_stream`):**
   * On every standard prompt, [`cjllm.py`](./cjllm.py) fires **Slot 1 (`determine_search_query`)** and **Slot 2 (Speculative Direct Answer Stream)** at the exact same millisecond (`t = 0`).
   * Whenever Slot 1 returns `None` (no web search needed), Slot 2 has **already completed prefill and begun streaming tokens** while the router was evaluating — hiding 100% of the search-router latency (`0.0 ms` post-router wait).
   * Whenever Slot 1 returns a search query, Slot 2 is aborted/closed in `< 1 ms`, freeing the server slot immediately before web retrieval begins.
2. **Automatic Section-Parallel / Map-Reduce Generation (`plan_parallel_sections` + `stream_parallel_sections`):**
   * Automatically detects when a prompt naturally decomposes into 3–5 independent parallel sections (such as `1) ... 2) ... 3) ...` numbered sub-tasks, comparing $\ge 3$ stock tickers like *"Compare NVDA, AAPL, and AMD today"*, comma-separated multi-topic explanations like *"Explain NoC routing, Paged KV caching, RoPE embeddings, and SwiGLU activations"*, or comprehensive deep-dive requests).
   * Launches **all $N$ section streams simultaneously** with a 99%-identical prompt prefix (maximizing `vLLM` KV prefix caching across all $N$ slots) and renders all $N$ sections growing in parallel in the terminal, achieving **`3.5×` to `4.2×` higher single-user throughput (`~58–75+ tok/s`)**.

##### QuietBox 2 (`P300X2`, 4× Blackhole) 32-Stream Continuous-Batching Benchmark (`DeepSeek-R1-Distill-Qwen-32B`)
| Simultaneous Streams | Total Output Tokens | Total Wall-Clock Time | TTFT (`min / avg / max`) | Per-Stream Decode Speed | Aggregate Decode Throughput | Net Wall-Clock Throughput | Scaling vs. 1 Stream |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 stream** | `128` | `7.04 s` | `87 / 87 / 87 ms` | **`18.4 t/s`** | **`18.4 t/s`** | **`18.2 t/s`** | **`1.00×`** |
| **4 streams** | `512` | `8.12 s` | `91 / 252 / 306 ms` | **`16.3 t/s`** | **`65.1 t/s`** | **`63.1 t/s`** | **`3.47×`** |
| **8 streams** | `1,024` | `9.43 s` | `95 / 523 / 587 ms` | **`14.4 t/s`** | **`115.0 t/s`** | **`108.6 t/s`** | **`5.97×`** |
| **16 streams** | `2,048` | `12.87 s` | `519 / 873 / 1,154 ms` | **`10.7 t/s`** | **`170.9 t/s`** | **`159.1 t/s`** | **`8.75×`** |
| **32 streams** | `4,096` | `20.21 s` | `117 / 2,209 / 2,284 ms` | **`7.1 t/s`** | **`227.8 t/s`** | **`202.7 t/s`** | **`11.15×`** |

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
  --max-pairs 30 \
  --max-verbatim-tokens 20000 \
  --evict-batch 8 \
  --tz America/Los_Angeles \
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
