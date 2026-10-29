# CJ vLLM Search Harness (`cjllm-search-harness`)

A high performance, single-file, zero-framework Python CLI inference and search-augmented generation harness ([`cjllm.py`](./cjllm.py)) built for **OpenAI-compatible `vLLM` endpoints** running on local AI workstations and accelerator appliances (including **Tenstorrent QuietBox 2 (`p300x2` Blackhole)** and **NVIDIA DGX Spark**).

**Author:** Christopher Jones ([@palisurf](https://github.com/palisurf))  
**Repository:** [https://github.com/palisurf/cjllm-harness](https://github.com/palisurf/cjllm-harness)

---

## How `cjllm.py` Compares to the Closest Single-File & Minimal CLI Harnesses

Most open-source LLM tools split into either **minimal terminal CLIs (without deep search or `vLLM` hardware tuning)** or **heavy multi-container web stacks (Open WebUI, Perplexica)**. [`cjllm.py`](./cjllm.py) sits in a niche of its own—combining the **single-file hackability** of Andrej Karpathy's `minbpe`/`llm.c` scripts, the **search-grounded synthesis** of Perplexica, and **hardware-aware `vLLM` continuous-batching / KV-cache optimization** tuned for appliances like the Tenstorrent QuietBox 2 and NVIDIA DGX Spark:

| Capability | [`cjllm.py`](./cjllm.py) | `aichat` | Simon Willison's `llm` | `tgpt` | `vllm chat` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Single-File Python Script** | **Yes** (`~2,850` LoC) | Compiled Rust crate | Multi-module Python pkg | Go binary | **Yes** (`~150` LoC) |
| **Keyless Search-by-Default + Auto-Failover** | **Yes** (`ddgs` + `primp` Yahoo failover + `Pass 0` APIs) | Paid API / MCP only | Plugin required | `-w` flag (snippets only) | No |
| **Multi-Query (`q1;q2;q3`), Direct APIs & `<table>` Scraper** | **Yes** (Finance/FX, Weather, NOAA Tides/Buoys, Patents, `arXiv`, `<table>`) | No | No | No | No |
| **Auto Parallel Slot Map-Reduce (`2–5` Streams)** | **Yes** (`2.8×–4.2×` speedup) | No | No | No | No |
| **KV Prefix-Cache-Aware Batch Memory (`15/30/40` pairs / `5/8/10` evict)** | **Yes** (3-tier auto-scaled) | Basic truncation | Full log | Basic truncation | Unbounded |
| **Tokenizer Bridge & vLLM Reasoning Parser Control** | **Yes** (4 auto-probed profiles) | Manual config | Manual | No | No |
| **50-Line Hybrid Transient SSH Scrollback Streamer** | **Yes** | Standard ANSI redraw | Raw stdout | Standard redraw | Raw stdout |

* **vs. `tgpt` / `chatblade` (Closest in Keyless Web Search):** `tgpt` and `chatblade` offer drop-in terminal chat with `-w` DuckDuckGo search, but only inject short ~150-character search snippets—lacking multi-query decomposition (`q1 ; q2 ; q3`), Yahoo Finance equity/FX/commodity cards, 7-day weather forecasts, direct USPTO Patent / `arXiv` / NOAA Tide & Buoy API resolvers, head+tail HTML `<table>` extraction, parallel `vLLM` slot fan-out, and KV prefix-cache memory management.
* **vs. `aichat` & Simon Willison's `llm` (Closest in Terminal REPL Ergonomics):** Both provide session persistence and Markdown rendering for `/v1/chat/completions` endpoints, but require external plugins or paid API keys for search and do not implement automatic multi-slot section-parallel decoding, `<|box_start|>` tokenizer bridging, or the 50-line hybrid transient SSH scrollback streamer.
* **vs. `vllm chat` (Closest in Direct `vLLM` Integration):** `vllm chat` is a ~150-line single-file Python REPL shipped inside `vLLM`, with no web search, no clock skill, no rolling memory archivist, no `/think` toggle, and no per-turn `TTFT / tok/s` hardware telemetry.

---

## 1. Introduction & Motivation

Running frontier open-weight LLMs locally on dedicated hardware (`vLLM` on Tenstorrent Blackhole/Wormhole accelerators or NVIDIA DGX/CUDA systems) provides raw token throughput and total data privacy, but exposes a gap between **serving a model** and **having a reliable daily driver**:

1. **Raw `vLLM` Endpoints Are Stateless and Cut Off in Time:** A bare `/v1/chat/completions` endpoint has no awareness of the current date/weekday, no access to live market quotes, forex/commodities, weather, ocean buoy/tide telemetry, patents, or post-cutoff `arXiv` papers, and no rolling memory management once a conversation outgrows the sliding window.
2. **Tokenizer & Reasoning-Template Friction:** Serving distilled reasoning checkpoints under hardware-optimized server profiles (such as running `DeepSeek-R1-Distill-Qwen-32B` weights under a `Qwen/Qwen3-32B` `tt-inference-server` specification) introduces special-token ID mismatches (`<|box_start|>` / `<|box_end|>` vs. `<think>` / `</think>`). Naive OpenAI clients suffer from leaked Chain-of-Thought monologues, doubled responses, or wasted reasoning latency on simple factual turns.
3. **Shallow Search & Compound-Query Collapse in Existing Local Frontends:** Most local LLM wrappers either require deploying heavy multi-container web stacks (Node.js, Postgres, Redis, vector DBs) or rely on single-string DuckDuckGo/Serper meta-description snippets (~150 characters). Single-query scrapers collapse when a prompt asks about multiple entities at once, rate-limit under multi-turn load, fail on exact patent/`arXiv` identifiers, and strip out the actual numbers from live stock indices, forex (`USD/JPY`), crude oil (`WTI`), multi-day weather tables, and NOAA tide/buoy tables.

### Intent of `cjllm-search-harness`
`cjllm-search-harness` was created to solve all three problems in **a single, self-contained Python script ([`cjllm.py`](./cjllm.py))** with zero external server dependencies, zero paid search API keys, and identical behavior across **Tenstorrent QuietBox 2** and **NVIDIA DGX Spark**:

* **Drop-in, Jitter-Free SSH Terminal Ergonomics:** Runs immediately over SSH or locally (`./run_chat`) with full GNU `readline` history, a **50-line hybrid transient Markdown streamer (`_BoundedMarkdownView` + `ProgressiveMarkdownStreamer`)** that commits completed Markdown blocks once into permanent scrollback (zero screen jitter, infinite scrollback), and instant slash commands (`/think`, `/v`, `/save`, `/load`, `/clear`).
* **Runtime Model & Tokenizer Auto-Adaptation:** Probes `/v1/models` and token behavior at startup (~100ms) to automatically configure stop tokens, reasoning prefills, and `chat_template_kwargs` across 4 profiles: `vllm_reasoning_parser` (`Qwen3.5`/`3.6`/`3.8` with `--reasoning-parser qwen3`), `deepseek_qwen3_bridge` (`DeepSeek-R1-Distill`), `native_think` (`<think>`/`</think>`), and `standard_instruct` (`Llama`, `Gemma`, `Mistral`).
* **Multi-Query Search-by-Default, `Pass 0` Direct Registries & Head+Tail HTML `<table>` Extraction:** Eliminates post-2024 training-cutoff blindness by decomposing compound prompts into up to 3 sub-queries (`q1 ; q2 ; q3`), resolving authoritative registries directly (`Pass 0`: Yahoo Finance `v8` Chart JSON API for equities/FX/commodities, NOAA CO-OPS Tides & NDBC Buoys, USPTO/Google Patents Dublin Core metadata + claims, official `arXiv` XML API + HTML benchmark tables, and `python.org`/`kernel.org`/`ubuntu.com` releases), extracting head+tail rows of structured HTML `<table>` blocks, and automatically failing over to keyless Yahoo Organic Search (`primp`) whenever `ddgs` rate-limits.
* **Automatic Multi-Slot Section-Parallel Map-Reduce & Hardware Telemetry:** Automatically decomposes multi-part, multi-entity, and exhaustive table prompts into **2–5 simultaneous `vLLM` streams** (`2.8×–4.2×` effective throughput speedup) and reports real-time **Time-To-First-Token (`TTFT ms`)**, **Generation Time (`s`)**, **Output Tokens**, and **Decode Throughput (`t/s`)** on every turn.

---

## 2. Comparison Against Other Local LLM Harnesses

| Capability | `cjllm-search-harness` (`cjllm.py`) | Raw `vLLM` / Basic OpenAI CLI | Ollama CLI (`ollama run`) | Open WebUI | Perplexica / Local Search UIs |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Deployment Footprint** | **Single Python file** (`pip install -r requirements.txt`) | Single script | Go daemon + CLI | Heavy Docker/Node/DB stack | Multi-container Docker + SearXNG |
| **Model & Tokenizer Auto-Detection** | **Yes** (`vllm_reasoning_parser`, `deepseek_qwen3_bridge`, `native_think`, `standard_instruct`) | No (manual stop tokens & templates) | Bundled Modelfile only | Manual per-model configuration | Manual configuration |
| **Fast `OFF` Mode for Reasoning Models** | **Yes** (`chat_template_kwargs.enable_thinking=False` or closed micro-thought prefill skips CoT; toggle `/think` anytime) | No (always runs full CoT monologue) | Limited (`--think` on supported tags only) | Displays CoT, rarely suppresses generation | No |
| **Live Search Without Paid API Keys** | **Yes** (Keyless `ddgs` + automatic `primp` Yahoo Organic failover) | None | None built-in | Basic snippet search or paid APIs | Requires local SearXNG container |
| **Live Equities, Indices, Forex & Commodities** | **Yes** (`^DJI`, `^GSPC`, `^IXIC`, `USD/JPY`, `EUR/USD`, `WTI Crude`, `Gold`, `BTC` via Yahoo `v8` API + cards) | None | None | Meta-description blurbs only | Meta-description blurbs only |
| **Direct Patents, `arXiv`, NOAA Tides/Buoys & HTML `<table>` Scraper** | **Yes** (`Pass 0` USPTO/Google Patents, `arXiv` XML API, NOAA CO-OPS Tides & NDBC Buoys + head/tail `<table>` rows) | None | None | Full-page text dump (loses table structure) | Text chunks only |
| **Multi-Query Search-by-Default & False-Premise Guard** | **Yes** (Splits compound queries `q1 ; q2 ; q3`, searches by default, and rejects false-premise trick questions) | None | None | Manual toggle or single query rewrite | Yes (higher latency multi-agent chain) |
| **Automatic Multi-Slot Parallel Generation** | **Yes** (Auto-decomposes multi-part / multi-entity / large table prompts into 2–5 simultaneous streams) | Single stream | Single stream | Single stream | Single stream |
| **SSH Scrollback-Safe Live Streaming** | **Yes** (50-line transient Markdown preview + once-only block commits to permanent scrollback) | Raw text stdout | Raw text stdout | Browser DOM | Browser DOM |
| **Memory Management** | **3-tier context-aware verbatim window (`15/30/40` pairs, `10K/20K/26K` tokens) + `5/8/10`-pair batch eviction into numerical summary buffer (KV prefix-cache friendly)** | Manual / unbounded | Simple context truncation | Full history / vector RAG | Per-thread history |
| **Crash-Safe Session Persistence** | **Yes** (Auto-saves `chat_cache.json` & `.chat_prompt_history` after every turn) | None | None | Database-backed | Database-backed |
| **Per-Turn Hardware Telemetry** | **Yes** (`TTFT ms`, `Gen Time s`, `Tokens`, `Speed t/s`, `/v` payload inspector) | None | `--verbose` summary at exit | Basic token speed | None |

---

## 3. Current Capabilities & Architecture

### Concise Feature Summary (`v297b2ee`)

| Subsystem | Key Functions in [`cjllm.py`](./cjllm.py) | Summary of Capability |
| :--- | :--- | :--- |
| **1. Model & Tokenizer Bridge** | `detect_model_profile`, `prepare_request_messages_and_extra`, `strip_thinking_and_dedup` | Auto-detects `vllm_reasoning_parser` (`chat_template_kwargs.enable_thinking`), `deepseek_qwen3_bridge` (`<\|box_start\|>`/`<\|box_end\|>`), `native_think` (`<think>`/`</think>`), or `standard_instruct`. Suppresses CoT by default (`think_mode=OFF`) and on utility calls; toggle CoT anytime with `/think`. |
| **2. Built-In Current Time & World Clock Skill** | `resolve_local_timezone`, `is_clock_skill_query`, `build_live_clock_skill_context`, `get_dual_clock_str` | Injects live local (`PDT`/`PST`, guarding against headless `Etc/UTC` hosts) and `UTC/GMT` clocks (12h + 24h) plus world timezone conversions into `messages[-1]` without mutating `messages[0]` (preserving KV prefix cache). |
| **3. Multi-Query Search-by-Default Router** | `is_search_opt_out_query`, `determine_search_query`, `_expand_compound_queries`, `_build_fallback_search_query` | Eliminates 2024 pre-training cutoff blindness by querying the web **by default**, decomposing multi-entity/multi-clause prompts into up to 3 sub-queries (`q1 ; q2 ; q3`), resolving 3-turn coreferences, and opting out **only** for clock skill, pure code/math/formulas, or formatting commands. |
| **4. `Pass 0` Direct APIs, Yahoo `v8` Finance & Head+Tail `<table>` Scraper** | `fetch_web_context`, `fetch_yahoo_rich_cards`, `_fetch_yahoo_chart_quote`, `_resolve_direct_authoritative_sources`, `fetch_top_page_table_or_excerpt`, `_extract_yahoo_organic_fallback` | Resolves equities, indices, FX (`USD/JPY`, `EUR/USD`), and commodities (`WTI Crude`, `Gold`) via Yahoo `v8` Chart API; resolves USPTO Patents (`patents.google.com`), `arXiv` XML API + HTML benchmark tables, NOAA CO-OPS Tides, 5-digit NDBC Buoys, and `python.org`/`kernel.org` directly (`Pass 0`); extracts head (`[:8]`) + tail (`[-12:]`) HTML `<table>` rows; and fails over to keyless Yahoo Organic Search when `ddgs` rate-limits. |
| **5. Quantitative Auto-Think & False-Premise Guard** | `needs_quantitative_reasoning`, `BASE_SYSTEM_PROMPT` | Automatically enables internal `<\|box_start\|>` reasoning on portfolio math, timezone deltas, and physics formulas even when `/think` is `OFF`, and enforces a **False-Premise & Trick-Question Guard** to reject fabricated premises when contradicted by live web data. |
| **6. 3-Tier Context-Aware Verbatim Window & Batch Archivist** | `resolve_memory_tier`, `compress_memory_batch`, `estimate_history_tokens` | Auto-scales verbatim window by model context length (`15/30/40` pairs, `10K/20K/26K` tokens, `5/8/10` batch eviction) while staying below the `32K` TT-Metal prefill cliff, keeping `messages[0]` byte-identical between batch evictions (**12/12 `100%` multi-needle recall at `~18.6K` tokens**). |
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
2. Multi-Query Search-by-Default Formulator (`determine_search_query` + `_expand_compound_queries`)
   • Opts out ONLY for: (1) Clock Skill, (2) Pure Coding/Math/Formulas, (3) Formatting/Recalling prior turns
   • Decomposes multi-entity / multi-clause prompts into up to 3 focused sub-queries (`q1 ; q2 ; q3`)
   • Overrides any mistaken 'NONE' or declarative prose via `_build_fallback_search_query`
   │
   ├──► [Search Active] Multi-Tier Web & Direct Registry Context Builder (`fetch_web_context`)
   │       ├─ Tier 0 (`Pass 0`): Direct Authoritative Registries (`_resolve_direct_authoritative_sources`)
   │       │     • USPTO / Google Patents Dublin Core Metadata & Claims (`patents.google.com/patent/US{id}/en`)
   │       │     • Official `arXiv` XML API (`export.arxiv.org`) + `arxiv.org/html/{id}` Benchmark Tables
   │       │     • NOAA CO-OPS Tide Predictions JSON API (`api.tidesandcurrents.noaa.gov`) & NDBC 5-Digit Buoys
   │       │     • Official Software Release Registries (`python.org/downloads/`, `kernel.org`, `ubuntu.com`)
   │       ├─ Tier 1: Yahoo Finance `v8` Chart JSON API (`_fetch_yahoo_chart_quote` for Equities, Indices, FX,
   │       │          WTI Crude, Gold, Crypto) + Multi-City 7-Day Weather & Entity Fact Cards (`fetch_yahoo_rich_cards`)
   │       ├─ Tier 2: Head+Tail HTML `<table>` Extractor (`<title>`/`<h1>` headers, skips `navbox` & `ltx_eqn_table`,
   │       │          keeps `[:8]` + `[-12:]` rows across up to 3 tables per sub-query)
   │       ├─ Tier 3: Cleaned Organic Snippets (`DDGS().text()` + automatic `_extract_yahoo_organic_fallback`)
   │       └─ Tier 4: Entity-Filtered News Articles (`DDGS().news()` with `_NEWS_GENERIC_STOPWORDS` overlap check)
   │
   ▼
3. Payload Construction & Model-Profile Adaptation (`prepare_request_messages_and_extra`)
   • Single merged system prompt (`BASE_SYSTEM_PROMPT` with False-Premise Guard + `summary_buffer`)
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
At startup, the harness queries `/v1/models` to auto-discover the active model ID and `max_model_len`, then runs a fast (~100ms, 3-token) probe to classify the server into one of four profiles:
1. **`vllm_reasoning_parser`:** Engaged when `vLLM` serves hybrid reasoning models (`Qwen/Qwen3.8-27B`, `Qwen/Qwen3.6-27B`, `Qwen/Qwen3.5`) with `--reasoning-parser qwen3` (populating `message.reasoning` / `delta.reasoning_content`). Automatically passes `chat_template_kwargs: {"enable_thinking": False}` when `think_mode = OFF` and on utility calls (search query router, rolling memory archivist, parallel section planner), and `{"enable_thinking": True}` when `/think` or `needs_quantitative_reasoning()` is active.
2. **`deepseek_qwen3_bridge`:** Engaged when the model closes `<|box_start|>` with `<|box_end|>` (`DeepSeek-R1-Distill-Qwen-32B` on `Qwen/Qwen3-32B`).
3. **`native_think`:** Engaged when the model uses literal `<think>` / `</think>` tokens inside `message.content` (native `DeepSeek-R1`, `Qwen3-32B`, `QwQ-32B`).
4. **`standard_instruct`:** Engaged for non-reasoning instruct models (`Llama-3.1-70B-Instruct`, `Llama-3.1-8B-Instruct`, `gemma-4-31B-it`, `Mistral`). Sends standard chat completion payloads while still supporting `/think` via prompt-guided `<think>...</think>` tags.

#### C. Built-In Current Time, Date & World Clock Skill (`build_live_clock_skill_context`) + Multi-Query Search-by-Default Router (`determine_search_query`)
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
* **Multi-Query Decomposition (`_expand_compound_queries`):** Automatically detects multi-entity or multi-clause prompts (*"Who are the current CEOs of Boeing, Starbucks, and Intel?"*, *"What is the latest stable version of Python, Ubuntu LTS, and Linux kernel?"*, *"Compare Loper Bright Chevron deference and the US TikTok divestiture law"*) and splits them into up to **3 focused sub-queries (`q1 ; q2 ; q3`)**, preventing DuckDuckGo from collapsing or starving secondary clauses.
* **3-Turn Multi-Hop & Entity Resolution:** Inspects up to 3 conversation turn pairs (`600` chars/turn) and automatically indexes 5-digit NOAA buoy station IDs, sports champions, and location clauses from recent history so multi-hop follow-ups (*"Pull the readings for that buoy"*, *"Who was the MVP of the most recent one and what was the final score?"*) preserve the exact target entity.

#### D. `Pass 0` Direct Authoritative Registries, Yahoo `v8` Finance/FX/Commodity API & Head+Tail HTML `<table>` Scraper (`fetch_web_context`)
1. **`Pass 0` Direct Authoritative Registry Resolvers (`_resolve_direct_authoritative_sources`):**
   * **USPTO / Google Patents Metadata & Claims:** Detects US patent numbers (`US Patent 11,487,979`, `US10997116B2`), fetches `https://patents.google.com/patent/US{id}/en` directly, and extracts Dublin Core `<meta name="DC.title">`, `<meta name="DC.contributor">` (Assignee & Inventors), `<meta name="DC.date">` (Filing & Grant dates), Abstract, and `<section itemprop="claims">`.
   * **Official `arXiv` XML API & HTML Benchmark Tables:** Detects `arXiv:YYMM.NNNNN` identifiers, queries `http://export.arxiv.org/api/query?id_list={id}` for canonical title, author list, submission date, and abstract, and prioritizes `https://arxiv.org/html/{id}` (following version redirects) to extract benchmark tables (`AIME 2024`, `MATH-500`, `H100 TFLOPs/s`).
   * **NOAA CO-OPS Tide Predictions & NDBC Ocean Buoys:** Queries `api.tidesandcurrents.noaa.gov` directly for coastal high/low tide predictions (`MLLW`, local time) and resolves 5-digit **NOAA NDBC buoy station IDs** (`46221`, `46054`, `46025`, etc.) directly to `https://www.ndbc.noaa.gov/station_page.php?station=<ID>`.
   * **Official Software & OS Release Portals:** Queries `https://www.python.org/downloads/` directly for the latest stable `Python 3.x.y` release and prioritizes `https://www.kernel.org/` and `https://ubuntu.com/project/docs/release-team/list-of-releases/`.
2. **Live Equities, Multi-Index, Forex, Commodity & Multi-City Weather Cards (`fetch_yahoo_rich_cards` & `_fetch_yahoo_chart_quote`):**
   * **Equities, Indices, Forex & Commodities:** Combines Yahoo Search rich cards with direct queries to Yahoo Finance's `v8/finance/chart/{sym}` JSON API for US indices (`^DJI`, `^GSPC`, `^IXIC`, `^RUT`), foreign exchange (`USD/JPY` `JPY=X`, `EUR/USD` `EURUSD=X`, `GBP/USD` `GBPUSD=X`), commodities (`WTI Crude Oil` `CL=F`, `Brent Crude` `BZ=F`, `Gold` `GC=F`, `Silver` `SI=F`), crypto (`BTC-USD`, `ETH-USD`), `10-Year Treasury Yield` (`^TNX`), and individual equities (`NVDA`, `AAPL`, `AMD`)—while guarding hardware/sport tokens (`RTX 5090`, `Super Bowl LX`, `Kenneth Walker III`) against false-positive ticker lookups.
   * **Multi-City Current Weather & 7-Day Forecast:** Automatically splits multi-location weather comparisons (*"Compare weather in Malibu, CA vs. Honolulu, HI"*) so each city gets its own live current conditions and 7-day forecast card.
3. **Two-Pass Generic HTML `<table>` Scraper (`fetch_top_page_table_or_excerpt`):**
   * **Pass 1 (Tabular Priority with Head+Tail Row Preservation):** Scans up to 4 candidate URLs per sub-query across prioritized tabular domains (`ndbc.noaa.gov`, `tidesandcurrents.noaa.gov`, `patents.google.com`, `arxiv.org`, `python.org`, `kernel.org`, `ubuntu.com`, `fred.stlouisfed.org`, `companiesmarketcap.com`, `stockanalysis.com`, `wikipedia.org`, `forbes.com`, `statranker.org`, `kalshi.com`, `basketball-reference.com`, `cbssports.com`), prepends the page `<title>` and `<h1>` to every table block, strips MediaWiki `data-mw` JSON attributes, skips `navbox` and LaTeX `ltx_eqn_table` math alignment blocks, and extracts **both the top rows (`[:8]`) and bottom rows (`[-12:]`)** across up to 3 tables per sub-query.
   * **Pass 2 (Plain-Text Fallback):** Falls back to a cleaned webpage excerpt when no candidate URL contains a valid HTML `<table>`.
4. **Organic Snippets (`DDGS().text()`) + Keyless Yahoo Organic Failover (`_extract_yahoo_organic_fallback`) & Filtered News (`DDGS().news()`):**
   * Automatically re-instantiates a fresh `DDGS()` session and fails over to [`_extract_yahoo_organic_fallback`](./cjllm.py) (`primp` desktop Chrome impersonation) whenever DuckDuckGo rate-limits or returns `< 2` results, and filters `DDGS().news()` items by entity-keyword overlap (`_NEWS_GENERIC_STOPWORDS`) so generic words like `"acquisition"` never inject unrelated breaking stock stories.

#### E. Quantitative Auto-Think Escalation, False-Premise Guard, High-Precision Memory & Crash-Safe Auto-Save
* **Quantitative Reasoning Escalation (`needs_quantitative_reasoning`):** When `think_mode = OFF` (default), prompts requiring multi-step math, share allocations, timezone deltas, or physics formulas (such as deep-water wavelength $L = \frac{gT^2}{2\pi}$) automatically enable internal `<|box_start|>` reasoning and instruct the model to write out intermediate equations, maintaining full mathematical accuracy.
* **False-Premise & Trick-Question Guard (`BASE_SYSTEM_PROMPT`):** Explicitly verifies that user premises match the entities in retrieved data—pushing back and correcting false premises (e.g., that Joe Biden withdrew in July 2024 and won `0` electoral votes, that OpenAI is privately held with no public stock ticker, or that the NVIDIA–ARM acquisition was terminated in February 2022) instead of sycophantically fabricating numbers.
* **3-Tier Context-Aware High-Water (`15/30/40` Pairs / `10K/20K/26K` Tokens) + Batch-Eviction Memory Architecture (`resolve_memory_tier` & `compress_memory_batch`):**
  * **Why 1-by-1 Sliding Windows Thrash the KV Prefix Cache:** In a naive sliding window (e.g., evicting 1 turn pair on every turn after Turn 10), folding the evicted turn into `summary_buffer` mutates `messages[0]` (the system prompt) on **every single turn** once the limit is reached. Because `vLLM` prefix caching hashes token blocks sequentially from token `0`, mutating `messages[0]` invalidates **100% of the cached KV blocks** on every subsequent turn.
  * **3-Tier Context-Aware Hysteresis Scaling (Capped Below the 32K Prefill Cliff):** [`resolve_memory_tier`](./cjllm.py) inspects `MODEL_PROFILE["max_model_len"]` at startup and automatically selects one of three hardware-safe tiers (unless overridden via `--max-pairs`, `--max-verbatim-tokens`, or `--evict-batch`):
    * **Tier 1 (`max_model_len <= 32,768`):** `MAX_PAIRS = 15`, `MAX_VERBATIM_TOKENS = 10,000`, `EVICT_BATCH_PAIRS = 5`
    * **Tier 2 (`32,769 – 262,143`, e.g. `131K` models):** `MAX_PAIRS = 30`, `MAX_VERBATIM_TOKENS = 20,000`, `EVICT_BATCH_PAIRS = 8`
    * **Tier 3 (`max_model_len >= 262,144`, e.g. `Qwen3.8-27B` `262K`):** `MAX_PAIRS = 40`, `MAX_VERBATIM_TOKENS = 26,000`, `EVICT_BATCH_PAIRS = 10`
  * Keeping `MAX_VERBATIM_TOKENS <= 26,000` (so `26K` verbatim + `~2K` live web context stays at `~28K` total tokens) avoids crossing the `32,768`-token (`32K`) TT-Metal prefill compilation cliff on Tenstorrent Blackhole (`P300X2`) and prevents per-step GDDR6 KV-cache bandwidth slowdown, while giving `262K` models **+33% more verbatim history (`40` pairs)** and **10-turn stretches** where `messages[0]` remains 100% byte-identical (see [`TT_PREFIX_CACHE_FIX.md`](./TT_PREFIX_CACHE_FIX.md)).
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

#### H. 12-Domain Web Query Accuracy & Ground-Truth Verification Suite (`23` Live & Post-Cutoff Prompts)
To ensure [`cjllm.py`](./cjllm.py) returns accurate, hallucination-free data across both live telemetry and post-2024 world events, the harness was evaluated on **Tenstorrent QuietBox 2 (`P300X2`)** across both **`Qwen/Qwen3.8-27B` (`vllm-tt-metal:0.25.0`, `262,144` max context)** and **`DeepSeek-R1-Distill-Qwen-32B` (`0.17.0-pr55630`, `131,072` max context)** against a **3-Tier Ground-Truth Oracle** (Tier 1 primary APIs: Yahoo Finance `v8`, NOAA CO-OPS/NDBC, Google Patents/USPTO, official `arXiv` XML API; Tier 2 canonical upstream registries; Tier 3 adversarial false-premise rejection rules).

##### Aggregate 23-Prompt Benchmark Comparison on Tenstorrent QuietBox 2 (`P300X2`)
| Model & TT-Metal Stack | Max Context | Reasoning Profile | Aggregate Score (`Max 23.0`) | Single-Run Accuracy (`%`) | Full Passes (`1.00`) | Partials (`0.85`) | Fails (`0.00`) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **`Qwen/Qwen3.8-27B` (`v0.25.0`, `qwen38_27b_qb2`)** | **`262,144`** | `vllm_reasoning_parser` | **`22.85 / 23.0`** | **`99.3%`** | **`22`** | **`1`** | **`0`** |
| **`DeepSeek-R1-Distill-Qwen-32B` (`v0.17.0-pr55630`, `tt_transformers`)** | `131,072` | `deepseek_qwen3_bridge` | **`22.55 / 23.0`** | **`98.0%`** | `20` | `3` | **`0`** |
| `DeepSeek-R1-Distill-Qwen-32B` *(Pre-Phase-2 Baseline)* | `131,072` | `deepseek_qwen3_bridge` | `13.68 / 23.0` | `59.5%` | `7` | `13` | `3` |

| Domain / Test Suite | What It Verifies | Verified Ground-Truth Result on `P300X2` (`Qwen/Qwen3.8-27B` & `DeepSeek-R1-Distill-Qwen-32B`) | Status |
| :--- | :--- | :--- | :---: |
| **1. Global Leaders, Pope & Fortune 500 CEOs** | Post-2024 political & corporate leadership turnover (`UK/Canada PMs`, `Pope Leo XIV`, `Boeing/Starbucks/Intel CEOs`) | Accurately identifies UK PM (`Andy Burnham`), Canada PM (`Mark Carney`), `Pope Leo XIV` (Robert Francis Prevost), and CEOs `Kelly Ortberg` (Boeing), `Brian Niccol` (Starbucks), & `Lip-Bu Tan` (Intel). | **PASS** |
| **2. Sports Champions & Award Winners** | Chronological tail rows (`Super Bowl LVIII–LX`, `World Series`, `2025/2026 Oscars`, `Nobel Physics`) | Extracts `Seattle Seahawks` / `Kenneth Walker III` (Super Bowl LX), `LA Dodgers` (World Series), *Anora* & *One Battle After Another* (Oscars), and `Francis Halzen` (Nobel Physics). | **PASS** |
| **3. Software, OS & Kernel Releases** | Direct release registries (`python.org`, `ubuntu.com`, `kernel.org`) | Resolves latest stable `Python 3.15.2`, `Ubuntu 26.04.1 LTS (Resolute Raccoon)` *(explicitly distinguished from interim `26.10`)*, and `Linux Kernel 7.3` (`7.2.13` stable, `6.18.59` LTS). | **PASS** |
| **4. Hardware Spec Comparisons** | Multi-GPU spec tables without false-positive `RTX` stock cards (`RTX 5090` vs `RTX 4090`) | Renders full Markdown spec table comparing `RTX 5090` (`Blackwell GB202`, `32 GB GDDR7`, `1,792 GB/s`, `575 W`, `$1,999`) vs `RTX 4090` (`Ada AD102`, `24 GB GDDR6X`, `1,008 GB/s`, `450 W`, `$1,599`). | **PASS** |
| **5. Macroeconomics, Forex & Commodities** | Non-ticker finance (`Fed target rate`, `US CPI`, `USD/JPY`, `EUR/USD`, `WTI Crude Oil`) | Reports FOMC target range (`3.75%–4.00%`, `3.88%` eff.), BLS CPI (`3.7%` YoY), live `USD/JPY` (`158.025`), `EUR/USD` (`1.1146`), and `WTI Crude` (`$102.01/bbl`). | **PASS** |
| **6. Spaceflight & Geopolitics** | Mission flight logs (`SpaceX Starship`) + alliance tables (`NATO` members) | Reports latest `Starship Flight 14` (`Sept 28, 2026`, Launch Successful) and `32` NATO member states (newest: `Finland [Apr 4, 2023]`, `Sweden [Mar 7, 2024]`). | **PASS** |
| **7. False-Premise / "Trick" Queries** | Sycophancy & fabrication resistance (`Biden 2024 EV`, `OpenAI ticker`, `NVIDIA–ARM completion`) | **3/3 Premise Rejections:** States Biden withdrew in July 2024 (`0` EV), OpenAI is private with no public ticker (`$991.47B` implied valuation), and NVIDIA–ARM was terminated on Feb 7, 2022. | **PASS** |
| **8. Global Rankings & Leaderboards** | Top-10 Market Cap companies + Top-5 Forbes Billionaires tables | Extracts Top 10 Market Cap table (`Nvidia $5.3T`, `Apple $5.1T`, `Alphabet $3.9T`, `Microsoft $3.9T`...) and Top 5 Forbes Billionaires (`Musk $839B`, `Page $257B`, `Brin $237B`, `Bezos $224B`, `Zuckerberg $222B`). | **PASS** |
| **9. Legal & Supreme Court Landmarks** | Post-cutoff constitutional & administrative law (`Loper Bright`, `TikTok v. Garland`) | Summarizes *Loper Bright* (`603 U.S. 369` overruling *Chevron* deference) and *TikTok, Inc. v. Garland* (`604 U.S. 56` upholding divestiture). | **PASS** |
| **10. Multi-Hop Conversational Follow-Ups** | 2-turn pronoun & event resolution (`"Who won the last 3 Super Bowls?"` $\rightarrow$ `"Who was the MVP of the most recent one and final score?"`) | Resolves `"the most recent one"` to Super Bowl LX (2026), naming MVP `Kenneth Walker III` and final score `Seahawks 29 – Patriots 13`. | **PASS** |
| **11. Patent Filings & Assignee Portfolios** | Direct USPTO / Google Patents lookup (`US Patent 11,487,979`) & Tenstorrent NoC/tensor patents | Extracts exact title, assignee (`Capital One Services LLC`), inventors, filing/grant dates, and Claim 1 for `US11487979` + 17-row Tenstorrent USPTO patent & application table (`11,301,264`, `11,269,628`, `11,245,643`, `20220245009` NoC overlay, etc.). | **PASS** |
| **12. Academic Publications & `arXiv` Preprints** | Direct `arXiv` XML/HTML lookup (`arXiv:2501.12948`) & literature comparison (`FlashAttention-3` + `Quantum LDPC`) | Extracts *DeepSeek-R1* (`arXiv:2501.12948`, `AIME 2024: 79.8%`, `MATH-500: 97.3%`) and comparative table for *FlashAttention-3* (`arXiv:2407.08608`) & *Quantum LDPC Codes* (`arXiv:2111.03654`, Panteleev & Kalachev). | **PASS** |
| **13. Coastal Tides & NBA Standings Tables** | Direct NOAA CO-OPS Station `9410840` tide tables & 15-team NBA Western Conference standings | Reports exact today/tomorrow Santa Monica/Malibu high & low tide times/heights (`MLLW`) and full 1–15 Western Conference standings table. | **PASS** |

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
| [`cjllm.py`](./cjllm.py) | Primary model-agnostic `vLLM` inference harness (`~2,825` LoC) with tokenizer bridge, multi-query search router, `Pass 0` direct API resolvers (Yahoo `v8`, NOAA Tides/Buoys, USPTO Patents, `arXiv`), head+tail HTML `<table>` scraper, automatic parallel Map-Reduce decoding, and 50-line hybrid SSH streamer |
| [`run_chat`](./run_chat) | Convenience launcher for `cjllm.py` (passes through any CLI arguments) |
| [`requirements.txt`](./requirements.txt) | Python package dependencies (`openai`, `rich`, `ddgs`, `primp`) |
| [`TT_PREFIX_CACHE_FIX.md`](./TT_PREFIX_CACHE_FIX.md) | Technical note on the `tt_transformers` (`PR #55630`) untraced prefix-cache page-table patch for Tenstorrent QuietBox 2 (`P300X2`) |
| [`.gitignore`](./.gitignore) | Git ignore rules (`__pycache__/`, `*.swp`, `.chat_prompt_history`, `chat_cache.json`, private test protocols/scripts) |
| [`chat_cache.json`](./chat_cache.json) | Default persisted conversation history and rolling summary buffer |
| [`README.md`](./README.md) | Motivation, harness comparison, architecture, feature reference, verification benchmarks, and usage documentation |
