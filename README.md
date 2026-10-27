# CJ vLLM Search Harness (`cjllm-search-harness`)

A lightweight, single-file, zero-framework Python CLI inference and search-augmented generation harness ([`cjllm.py`](./cjllm.py)) built for **OpenAI-compatible `vLLM` endpoints** running on local AI workstations and accelerator appliances (including **Tenstorrent QuietBox 2 (`p300x2` Blackhole)** and **NVIDIA DGX Spark**).

**Author:** Christopher Jones ([@palisurf](https://github.com/palisurf))  
**Repository:** [https://github.com/palisurf/cjllm-harness](https://github.com/palisurf/cjllm-harness)

---

## How `cjllm.py` Compares to the Closest Single-File & Minimal CLI Harnesses

Most open-source LLM tools split into either **minimal terminal CLIs (without deep search or `vLLM` hardware tuning)** or **heavy multi-container web stacks (Open WebUI, Perplexica)**. [`cjllm.py`](./cjllm.py) sits in a niche of its own—combining the **single-file hackability** of Andrej Karpathy's `minbpe`/`llm.c` scripts, the **search-grounded synthesis** of Perplexica, and **hardware-aware `vLLM` continuous-batching / KV-cache optimization** tuned for appliances like the Tenstorrent QuietBox 2 and NVIDIA DGX Spark:

| Capability | [`cjllm.py`](./cjllm.py) | `aichat` | Simon Willison's `llm` | `tgpt` | `vllm chat` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Single-File Python Script** | **Yes** (`~2,280` LoC) | Compiled Rust crate | Multi-module Python pkg | Go binary | **Yes** (`~150` LoC) |
| **Keyless Search-by-Default** | **Yes** (`ddgs` + `primp`) | Paid API / MCP only | Plugin required | `-w` flag (snippets only) | No |
| **Live Finance, Weather, Buoy & HTML `<table>` Scraper** | **Yes** (4-tier + head/tail rows) | No | No | No | No |
| **Auto Parallel Slot Map-Reduce (`2–5` Streams)** | **Yes** (`2.8×–4.2×` speedup) | No | No | No | No |
| **KV Prefix-Cache-Aware Batch Memory (`30` pairs / `8` evict)** | **Yes** | Basic truncation | Full log | Basic truncation | Unbounded |
| **Tokenizer Bridge (`<\|box_start\|>` / `/think` toggle)** | **Yes** (auto-probed) | Manual config | Manual | No | No |
| **50-Line Hybrid Transient SSH Scrollback Streamer** | **Yes** | Standard ANSI redraw | Raw stdout | Standard redraw | Raw stdout |

* **vs. `tgpt` / `chatblade` (Closest in Keyless Web Search):** `tgpt` and `chatblade` offer drop-in terminal chat with `-w` DuckDuckGo search, but only inject short ~150-character search snippets—lacking Yahoo Finance multi-ticker cards, 7-day weather forecasts, head+tail HTML `<table>` extraction, NOAA buoy resolution, parallel `vLLM` slot fan-out, and KV prefix-cache memory management.
* **vs. `aichat` & Simon Willison's `llm` (Closest in Terminal REPL Ergonomics):** Both provide session persistence and Markdown rendering for `/v1/chat/completions` endpoints, but require external plugins or paid API keys for search and do not implement automatic multi-slot section-parallel decoding, `<|box_start|>` tokenizer bridging, or the 50-line hybrid transient SSH scrollback streamer.
* **vs. `vllm chat` (Closest in Direct `vLLM` Integration):** `vllm chat` is a ~150-line single-file Python REPL shipped inside `vLLM`, with no web search, no clock skill, no rolling memory archivist, no `/think` toggle, and no per-turn `TTFT / tok/s` hardware telemetry.

---

## 1. Introduction & Motivation

Running frontier open-weight LLMs locally on dedicated hardware (`vLLM` on Tenstorrent Blackhole/Wormhole accelerators or NVIDIA DGX/CUDA systems) provides raw token throughput and total data privacy, but exposes a gap between **serving a model** and **having a reliable daily driver**:

1. **Raw `vLLM` Endpoints Are Stateless and Cut Off in Time:** A bare `/v1/chat/completions` endpoint has no awareness of the current date/weekday, no access to live market quotes, weather, or ocean buoy telemetry, and no rolling memory management once a conversation outgrows the sliding window.
2. **Tokenizer & Reasoning-Template Friction:** Serving distilled reasoning checkpoints under hardware-optimized server profiles (such as running `DeepSeek-R1-Distill-Qwen-32B` weights under a `Qwen/Qwen3-32B` `tt-inference-server` specification) introduces special-token ID mismatches (`<|box_start|>` / `<|box_end|>` vs. `<think>` / `</think>`). Naive OpenAI clients suffer from leaked Chain-of-Thought monologues, doubled responses, or wasted reasoning latency on simple factual turns.
3. **Shallow Search in Existing Local Frontends:** Most local LLM wrappers either require deploying heavy multi-container web stacks (Node.js, Postgres, Redis, vector DBs) or rely on basic DuckDuckGo/Serper meta-description snippets (~150 characters). Those snippets strip out the actual numbers from live stock indices (`Dow`, `S&P 500`, `Nasdaq`), multi-day weather tables, reference rankings, and NOAA NDBC buoy station tables—causing local models to either refuse or hallucinate figures.

### Intent of `cjllm-search-harness`
`cjllm-search-harness` was created to solve all three problems in **a single, self-contained Python script ([`cjllm.py`](./cjllm.py))** with zero external server dependencies, zero paid search API keys, and identical behavior across **Tenstorrent QuietBox 2** and **NVIDIA DGX Spark**:

* **Drop-in, Jitter-Free SSH Terminal Ergonomics:** Runs immediately over SSH or locally (`./run_chat`) with full GNU `readline` history, a **50-line hybrid transient Markdown streamer (`_BoundedMarkdownView` + `ProgressiveMarkdownStreamer`)** that commits completed Markdown blocks once into permanent scrollback (zero screen jitter, infinite scrollback), and instant slash commands (`/think`, `/v`, `/save`, `/load`, `/clear`).
* **Runtime Model & Tokenizer Auto-Adaptation:** Probes `/v1/models` and token behavior at startup (~100ms) to automatically configure stop tokens and reasoning prefills for `DeepSeek-R1-Distill` (`deepseek_qwen3_bridge`), native reasoning models (`native_think`), or standard instruct models (`standard_instruct`).
* **Search-by-Default Grounding & Head+Tail HTML `<table>` Extraction:** Eliminates post-2024 training-cutoff blindness by querying the web by default on every informational prompt (opting out only for clock skill, pure coding/math, or formatting prior turns) and running a 4-tier scraper that pulls Yahoo Finance multi-ticker/macro cards, 7-day weather forecasts, head+tail rows of structured HTML `<table>` blocks, and direct NOAA NDBC buoy telemetry (`ndbc.noaa.gov`).
* **Automatic Multi-Slot Section-Parallel Map-Reduce & Hardware Telemetry:** Automatically decomposes multi-part, multi-entity, and exhaustive table prompts into **2–5 simultaneous `vLLM` streams** (`2.8×–4.2×` effective throughput speedup) and reports real-time **Time-To-First-Token (`TTFT ms`)**, **Generation Time (`s`)**, **Output Tokens**, and **Decode Throughput (`t/s`)** on every turn.

---

## 2. Comparison Against Other Local LLM Harnesses

| Capability | `cjllm-search-harness` (`cjllm.py`) | Raw `vLLM` / Basic OpenAI CLI | Ollama CLI (`ollama run`) | Open WebUI | Perplexica / Local Search UIs |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Deployment Footprint** | **Single Python file** (`pip install -r requirements.txt`) | Single script | Go daemon + CLI | Heavy Docker/Node/DB stack | Multi-container Docker + SearXNG |
| **Model & Tokenizer Auto-Detection** | **Yes** (`deepseek_qwen3_bridge`, `native_think`, `standard_instruct`) | No (manual stop tokens & templates) | Bundled Modelfile only | Manual per-model configuration | Manual configuration |
| **Fast `OFF` Mode for Reasoning Models** | **Yes** (Closed micro-thought prefill skips CoT & cuts latency ~50%; toggle `/think` anytime) | No (always runs full CoT monologue) | Limited (`--think` on supported tags only) | Displays CoT, rarely suppresses generation | No |
| **Live Search Without Paid API Keys** | **Yes** (Keyless `ddgs` + `primp` browser-impersonated scraping) | None | None built-in | Basic snippet search or paid APIs | Requires local SearXNG container |
| **Live Financial & Multi-Index Quotes** | **Yes** (Auto-expands `Dow`, `S&P 500`, `Nasdaq` & ticker quote cards) | None | None | Meta-description blurbs only | Meta-description blurbs only |
| **Structured HTML `<table>` & NOAA Buoy Extraction** | **Yes** (Head+tail row extraction across up to 8 tables/page + auto-resolves 5-digit NOAA NDBC buoy IDs) | None | None | Full-page text dump (loses table structure) | Text chunks only |
| **Search-by-Default Context Router** | **Yes** (Searches by default to prevent post-2024 cutoff hallucinations; opts out only for clock/code/math/formatting) | None | None | Manual toggle or basic query rewrite | Yes (higher latency multi-agent chain) |
| **Automatic Multi-Slot Parallel Generation** | **Yes** (Auto-decomposes multi-part / multi-entity / large table prompts into 2–5 simultaneous streams) | Single stream | Single stream | Single stream | Single stream |
| **SSH Scrollback-Safe Live Streaming** | **Yes** (50-line transient Markdown preview + once-only block commits to permanent scrollback) | Raw text stdout | Raw text stdout | Browser DOM | Browser DOM |
| **Memory Management** | **30-pair (`~20K` token) verbatim window + 8-pair batch eviction into numerical summary buffer (KV prefix-cache friendly)** | Manual / unbounded | Simple context truncation | Full history / vector RAG | Per-thread history |
| **Crash-Safe Session Persistence** | **Yes** (Auto-saves `chat_cache.json` & `.chat_prompt_history` after every turn) | None | None | Database-backed | Database-backed |
| **Per-Turn Hardware Telemetry** | **Yes** (`TTFT ms`, `Gen Time s`, `Tokens`, `Speed t/s`, `/v` payload inspector) | None | `--verbose` summary at exit | Basic token speed | None |

---

## 3. Current Capabilities & Architecture

### Concise Feature Summary (`v6b068d8`)

| Subsystem | Key Functions in [`cjllm.py`](./cjllm.py) | Summary of Capability |
| :--- | :--- | :--- |
| **1. Model & Tokenizer Bridge** | `detect_model_profile`, `prepare_request_messages_and_extra`, `strip_thinking_and_dedup` | Auto-detects `deepseek_qwen3_bridge` (`<\|box_start\|>`/`<\|box_end\|>`), `native_think` (`<think>`/`</think>`), or `standard_instruct`. Suppresses CoT by default (`think_mode=OFF`) via completed micro-thought prefill (~50% lower latency); toggle CoT anytime with `/think`. |
| **2. Built-In Current Time & World Clock Skill** | `resolve_local_timezone`, `is_clock_skill_query`, `build_live_clock_skill_context`, `get_dual_clock_str` | Injects live local (`PDT`/`PST`, guarding against headless `Etc/UTC` hosts) and `UTC/GMT` clocks (12h + 24h) plus world timezone conversions into `messages[-1]` without mutating `messages[0]` (preserving KV prefix cache). |
| **3. Search-by-Default Query Formulator** | `is_search_opt_out_query`, `determine_search_query`, `_build_fallback_search_query` | Eliminates 2024 pre-training cutoff blindness by querying the web **by default** on all factual/informational prompts (resolving 3-turn pronouns & 5-digit buoy IDs), opting out **only** for clock skill, pure code/math/formulas, or formatting/recall commands on prior turns. |
| **4. 4-Tier Web, Finance, Weather & Head+Tail Table Scraper** | `fetch_web_context`, `fetch_yahoo_rich_cards`, `fetch_top_page_table_or_excerpt` | Pulls parallel Yahoo Finance multi-ticker/index/macro cards, multi-city 7-day weather forecasts, direct NOAA NDBC buoy pages, and up to 3 structured HTML `<table>` blocks (preserving both top `[:8]` and bottom `[-12:]` chronological rows while stripping MediaWiki `data-mw` & `navbox` noise). |
| **5. Quantitative Auto-Think Escalation** | `needs_quantitative_reasoning` | Automatically enables internal `<\|box_start\|>` reasoning and step-by-step intermediate equations on portfolio math, timezone deltas, and physics formulas even when `/think` is `OFF` (while excluding pure code generation). |
| **6. 30-Pair Verbatim Window & 8-Pair Batch Archivist** | `compress_memory_batch`, `estimate_history_tokens` | Retains up to `30` verbatim turn pairs (`~20,000` tokens) and batch-evicts the oldest `8` pairs in one archivist call into `summary_buffer`, keeping `messages[0]` byte-identical for 8 turns at a time (**12/12 `100%` multi-needle recall at `~18.6K` tokens**). |
| **7. Automatic Section-Parallel Map-Reduce Generation** | `detect_auto_parallel_sections`, `plan_parallel_sections`, `stream_parallel_sections` | Automatically decomposes multi-part (`1) 2)`), multi-ticker ($\ge 2$), multi-entity (`X and Y`), deep-dive, and exhaustive table (`"List all..."`) prompts into **2–5 simultaneous `vLLM` streams** (`2.8×–4.2×` effective throughput speedup). |
| **8. 50-Line Hybrid Transient Streamer & Scrollback Commit** | `_BoundedMarkdownView`, `ProgressiveMarkdownStreamer` | Streams up to `50` lines (clamped to `console.size.height - 2`) of live formatted Markdown above a 1-line token counter (`⚡ Streaming (N tok)`), then switches on completion to **Once-Only Progressive Markdown Block Commits** for jitter-free, infinite SSH scrollback. |

```
User Input
   │
   ├──► Slash Command Handler (/think, /v, /clear, /save, /load, chat_end, /exit)
   │
   ▼
1. Built-In Current Time Skill (`is_clock_skill_query`) & Parallel Planner (`plan_parallel_sections`)
   • Injects live `[SKILL: LIVE SYSTEM CLOCK & CALENDAR]` block when time/date/weekday is requested
   • Checks if prompt decomposes into 2–5 parallel sections (multi-entity, exhaustive table, or deep-dive)
   │
   ▼
2. Search-by-Default Query Formulator (`determine_search_query` + `is_search_opt_out_query`)
   • Opts out ONLY for: (1) Clock Skill, (2) Pure Coding/Math/Formulas, (3) Formatting/Recalling prior turns
   • For all other prompts: formulates a 2–10 word search query using 3-turn context & 2026 date grounding
   • Overrides any mistaken 'NONE' or declarative prose via `_build_fallback_search_query`
   │
   ├──► [Search Active] 4-Tier Web Context Builder (`fetch_web_context`)
   │       ├─ Tier 1: Parallel Yahoo Finance Quote, Multi-Index (`^DJI`, `^GSPC`, `^IXIC`), Macro & Weather Cards
   │       ├─ Tier 2: Head+Tail HTML `<table>` Extractor (skips navboxes, keeps `[:8]` + `[-12:]` rows + NOAA buoys)
   │       ├─ Tier 3: Cleaned Organic Web Snippets (`DDGS().text()`)
   │       └─ Tier 4: Recent News Articles (`DDGS().news()`)
   │
   ▼
3. Payload Construction & Model-Profile Adaptation (`prepare_request_messages_and_extra`)
   • Single merged system prompt (`BASE_SYSTEM_PROMPT` + `summary_buffer`)
   • Verbatim sliding window (`MAX_PAIRS = 30` pairs / `~20,000` tokens high-water mark, batch-evicting `8` pairs)
   • Active user turn (with 2026 web data + chronological table & exhaustive-list synthesis rules)
   • Automatic tokenizer prefill & stop-token adaptation based on `MODEL_PROFILE`
   │
   ▼
4. Execution: Single Stream OR Automatic 2–5 Stream Parallel Map-Reduce (`stream_parallel_sections`)
   • 50-line hybrid transient Markdown view (`_BoundedMarkdownView`) + 1-line token-count-only status footer
   • Upon completion: switches to Once-Only Progressive Markdown Block Commits (`ProgressiveMarkdownStreamer.finish`)
   • Computes and displays `TTFT (ms)`, `Gen Time (s)`, `Tokens`, and `Speed (t/s)` (plus parallel speedup `×`)
   │
   ▼
5. Sanitization, Rolling Batch Memory Compression & Crash-Safe Auto-Save
   • Strips CoT blocks and residual special tokens (`strip_thinking_and_dedup`)
   • Quality gate discards degenerate/corrupted responses before they pollute history
   • Batch-evicts 8 oldest turn pairs when exceeding `MAX_PAIRS = 30` (`compress_memory_batch`)
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

#### C. Built-In Current Time, Date & World Clock Skill (`build_live_clock_skill_context`) + Search-by-Default Router (`determine_search_query`)
* **Built-In Current Time, Calendar & World Clock Skill (`is_clock_skill_query` / `build_live_clock_skill_context`):**
  * Automatically resolves the user's local IANA timezone (`--tz` or `CJLLM_TZ`, defaulting to `America/Los_Angeles` `PDT`/`PST` when running on headless `Etc/UTC` Linux servers/containers so evening queries after 5:00 PM Pacific never roll forward into tomorrow's UTC date).
  * Intercepts queries asking for the current time, date, day of the week (*"What time is it now"*, *"What day of the week is tomorrow"*), or world timezones (*"What time is it in Tokyo and London right now?"*) and injects an authoritative `[SKILL: LIVE SYSTEM CLOCK & CALENDAR]` block into the active turn (`messages[-1]`) down to the second—without mutating `messages[0]` (preserving 100% KV prefix-cache hit rates) and overriding any older timestamps in conversation history.
* **Dual Local & UTC/GMT Clock Grounding (`get_dual_clock_str`):** Injects both the local system clock (`PDT/PST`) and the current `UTC/GMT` clock (in both 12-hour and 24-hour formats) into the search query formulator and web context header so UTC-stamped marine/aviation reports (like NOAA buoys crossing UTC midnight) are always converted and subtracted accurately.
* **Search-by-Default Philosophy (`is_search_opt_out_query` & `_build_fallback_search_query`):**
  * Because a model with a 2024 pre-training cutoff does not know which world facts changed between 2024 and 2026 (elections, CEOs, mergers, laws, hardware releases), [`determine_search_query`](./cjllm.py) operates in **Search-by-Default** mode: it **always** generates a search query and fetches live web context unless [`is_search_opt_out_query`](./cjllm.py) detects that the prompt is strictly:
    1. Built-in clock/calendar skill,
    2. Pure coding, math, algebra, or physics formula evaluation, or
    3. A formatting, sorting, conversion, or memory-recall command on previous conversation turns (*"sort that table"*, *"make it shorter"*, *"convert to JSON"*, *"summarize our conversation so far"*).
  * If the LLM router ever outputs `NONE` or declarative prose on a non-opt-out prompt, [`_build_fallback_search_query`](./cjllm.py) overrides it with a clean keyword query so 2024 pre-training confidence can never suppress live verification.
* **3-Turn Multi-Hop & Entity Resolution:** Inspects up to 3 conversation turn pairs (`600` chars/turn) and automatically indexes 5-digit NOAA buoy station IDs and location clauses from recent history so multi-hop follow-ups (*"Pull the readings for that buoy"*, *"the second one"*, *"for today only instead"*) preserve the exact station ID or ticker.

#### D. 4-Tier Live Finance, Weather, NOAA Buoy Telemetry & Generic HTML `<table>` Extractor (`fetch_web_context`)
1. **Live Market, Multi-Ticker, Macro-Asset, Multi-City Weather & Fact Cards (`fetch_yahoo_rich_cards`):**
   * **Stock, Multi-Index & Macro Quotes:** Automatically expands queries mentioning multiple US indices (`^DJI`, `^GSPC`, `^IXIC`, `^RUT`), macro assets (`Bitcoin`, `Ethereum`, `Gold`, `10-Year Treasury Yield`), or multiple individual stock tickers (*"Compare NVDA, AAPL, and AMD today"*) into parallel live Yahoo Finance quote card lookups.
   * **Multi-City Current Weather & 7-Day Forecast:** Automatically splits multi-location weather comparisons (*"Compare weather in Malibu, CA vs. Honolulu, HI"*) so each city gets its own live current conditions and 7-day forecast card.
   * **Structured Entity Fact Tables:** Extracts embedded financial and company metrics.
2. **Two-Pass Generic HTML `<table>` & NOAA Buoy Telemetry Scraper (`fetch_top_page_table_or_excerpt`):**
   * **Pass 1 (Tabular Priority with Head+Tail Row Preservation):** Scans up to 4 candidate URLs across prioritized tabular domains (`ndbc.noaa.gov`, `weather.gov`, `companiesmarketcap.com`, `stockanalysis.com`, `wikipedia.org`, `slickcharts.com`, `finance.yahoo.com`, `forbes.com`, `statranker.org`) as well as any general webpage containing structured tables, strips MediaWiki `data-mw` JSON attributes and `<sup>` footnote tags, skips footer/sidebar `<table class="navbox|ambox|mbox|metadata|toc">` elements, and extracts **both the top rows (`[:8]`) and bottom rows (`[-12:]`)** across up to 3 distinct table blocks (`4,500` chars)—ensuring long chronological tables (such as US Presidents `#1–#47`) always retain the 2025–2026 incumbent rows at the bottom.
   * **Automatic NOAA Buoy Telemetry Resolution:** Detects 5-digit **NOAA NDBC buoy station IDs** (`46221`, `46054`, `46025`, etc.) in either the search query or snippets and directly fetches `https://www.ndbc.noaa.gov/station_page.php?station=<ID>`.
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

#### G. Automatic Multi-Slot Concurrency Acceleration (Section-Parallel Map-Reduce Generation) & 50-Line Hybrid SSH Streamer
Because LLM decode on both **Tenstorrent QuietBox 2 (`P300X2`, 32 slots)** and **NVIDIA DGX Spark (`GB10`, 128+ slots)** is memory-bandwidth bound at `batch = 1`, `vLLM` continuous batching can decode multiple concurrent streams in a single forward pass by reading the 32B model weights from memory once per step. [`cjllm.py`](./cjllm.py) exploits those parallel slots **100% automatically with zero slash commands**:

* **Automatic Section-Parallel / Map-Reduce Generation (`detect_auto_parallel_sections` + `plan_parallel_sections` + `stream_parallel_sections`):**
  * Automatically detects when a prompt naturally decomposes into **2–5 independent parallel sections**, including:
    1. Explicit numbered sub-tasks (`1) ... 2) ...`),
    2. Multi-ticker / multi-company comparisons ($\ge 2$ tickers, e.g. *"Compare NVDA, AAPL, and AMD today"*),
    3. Comma-separated multi-topic explanations/lists (*"Explain NoC routing, Paged KV caching, RoPE embeddings, and SwiGLU activations"*),
    4. Multi-entity table/comparison requests (`X and Y` / `X vs Y`), and
    5. Exhaustive `"List all..."` table requests (*"List all of the named moons of Saturn and Jupiter, in a table"*, which automatically splits large 2-category enumerations into **4 parallel sub-table streams** with up to `1,400` tokens/stream).
  * Runs **after** live web search retrieval so every parallel slot receives the full 2026 web-grounded context, launches **all $N$ section streams simultaneously** with a 99%-identical prompt prefix (maximizing `vLLM` KV prefix caching across all $N$ slots), and achieves **`2.8×` to `4.2×` higher single-user throughput (`~40–75+ tok/s`)**.
* **50-Line Hybrid Transient Streamer + Once-Only Progressive Markdown Block Commits (`_BoundedMarkdownView` & `ProgressiveMarkdownStreamer`):**
  * **Why Naive `Live(Markdown(...))` Breaks SSH Scrollback:** Redrawing a growing multi-line Markdown document 15×/sec over SSH causes visible screen jitter, and once a response exceeds the physical terminal height (`console.size.height`), ANSI cursor-up sequences (`\x1b[<N>A`) clamp at Row 0 of the viewport—pushing duplicate frames into the terminal scrollback buffer on every token chunk.
  * **Hybrid 50-Line Transient Window + Permanent Block Commits:** While a response is generating, [`_BoundedMarkdownView`](./cjllm.py) renders up to **50 lines** (dynamically bounded by `console.size.height - 2`) of live formatted Markdown above a 1-line token-count-only footer (`⚡ Streaming (N tok)`) inside a `transient=True` `Live` window. If a response exceeds 50 lines while streaming, completed top-level blocks (`\n\n` outside code fences/tables/lists) are progressively committed once above the live window. The instant generation finishes, the transient window clears and [`ProgressiveMarkdownStreamer.finish()`](./cjllm.py) commits all remaining Markdown blocks once into the SSH scrollback buffer for indefinite review.

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
| `chat_end` / `exit` / `quit` / `/exit` / `/quit` | Save session to `chat_cache.json` and exit |

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
