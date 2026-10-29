# CJ vLLM Search Harness (`cjllm-search-harness`)

A high performance, single-file, zero-framework Python CLI inference and search-augmented generation harness ([`cjllm.py`](./cjllm.py)) built for **OpenAI-compatible `vLLM` endpoints** running on local AI workstations and accelerator appliances (including **Tenstorrent QuietBox 2 (`p300x2` Blackhole)** and **NVIDIA DGX Spark**).

**Author:** Christopher Jones ([@palisurf](https://github.com/palisurf))  
**Repository:** [https://github.com/palisurf/cjllm-harness](https://github.com/palisurf/cjllm-harness)

---

## 1. Overview & Motivation

Running frontier open-weight LLMs locally on dedicated hardware (`vLLM` on Tenstorrent Blackhole/Wormhole accelerators or NVIDIA DGX/CUDA systems) provides raw token throughput and total data privacy, but exposes three practical gaps between **serving a model** and **having a reliable daily driver**:

1. **Stateless Endpoints Cut Off in Time:** Bare `/v1/chat/completions` servers have no awareness of the current date/weekday, no access to live market quotes, forex/commodities, weather, ocean buoy/tide telemetry, patents, or post-cutoff `arXiv` papers, and no rolling memory management once a conversation outgrows the context window.
2. **Tokenizer & Reasoning-Parser Friction:** Serving diverse model families (`Qwen3.8-27B` with `--reasoning-parser qwen3`, `openai/gpt-oss-120b` with Harmony `reasoning_effort`, or `DeepSeek-R1-Distill-Qwen-32B` under a `Qwen3` tokenizer bridge) introduces incompatible CoT controls and special-token IDs (`<|box_start|>` vs. `<think>` vs. `delta.reasoning`), leading to leaked Chain-of-Thought monologues or wasted reasoning latency on simple factual turns.
3. **Shallow Search & Compound-Query Collapse:** Most local wrappers either require heavy multi-container web stacks (Node.js, Postgres, Redis, SearXNG) or inject short ~150-character DuckDuckGo snippets that collapse on multi-entity prompts and strip out tabular data from live stock indices, forex (`USD/JPY`), crude oil (`WTI`), weather forecasts, and NOAA tide/buoy tables.

[`cjllm.py`](./cjllm.py) solves all three in **a single self-contained Python script (`~2,900` LoC)** with zero external server dependencies and zero paid search API keys:

| Capability | [`cjllm.py`](./cjllm.py) | `aichat` / `llm` | `tgpt` | `vllm chat` | Open WebUI / Perplexica |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Deployment Footprint** | **Single Python file** (`~2,900` LoC) | Rust crate / Python pkg | Go binary | Single file (`~150` LoC) | Multi-container Docker + DB |
| **Model & Reasoning Parser Auto-Detection** | **Yes** (4 auto-probed profiles: `qwen3`, `gpt-oss`, `deepseek_bridge`, `instruct`) | Manual config | No | No | Manual per-model config |
| **Fast `OFF` Mode for Reasoning Models** | **Yes** (`enable_thinking=False`, `reasoning_effort="low"`, or micro-thought prefill; `/think` toggle) | Manual | No | No | Displays CoT, rarely suppresses |
| **Keyless Search-by-Default + Auto-Failover** | **Yes** (`ddgs` + `primp` Yahoo Organic failover + `Pass 0` direct APIs) | Plugin / Paid API | `-w` (snippets only) | No | Requires SearXNG or paid API |
| **Multi-Query (`q1;q2;q3`), Direct APIs & `<table>` Scraper** | **Yes** (Yahoo `v8` Finance/FX/WTI, Weather, NOAA Tides/Buoys, USPTO Patents, `arXiv`, head+tail `<table>`) | No | No | No | Text chunks only (loses tables) |
| **Auto Parallel Slot Map-Reduce (`2–5` Streams)** | **Yes** (`2.8×–4.2×` speedup) | No | No | No | Single stream |
| **KV Prefix-Cache-Aware Batch Memory** | **Yes** (3-tier auto-scaled: `15/30/40` pairs, `10K/20K/26K` tok, `5/8/10` batch evict) | Basic truncation / full log | Basic truncation | Unbounded | Full history / vector RAG |
| **50-Line Hybrid Transient SSH Streamer + Telemetry** | **Yes** (Jitter-free block commits + `TTFT ms`, `Gen Time s`, `Tokens`, `tok/s`, `/v` inspector) | Standard redraw | Raw stdout | Raw stdout | Browser DOM |

---

## 2. Architecture & Core Subsystems

| Subsystem | Key Functions in [`cjllm.py`](./cjllm.py) | Summary of Capability |
| :--- | :--- | :--- |
| **1. Model & Reasoning Bridge** | `detect_model_profile`, `prepare_request_messages_and_extra`, `strip_thinking_and_dedup` | Auto-detects `vllm_reasoning_parser` (`chat_template_kwargs.enable_thinking` for `Qwen3.5/3.6/3.8` or `reasoning_effort` for `gpt-oss-120b/20b`), `deepseek_qwen3_bridge` (`<\|box_start\|>`/`<\|box_end\|>`), `native_think` (`<think>`/`</think>`), or `standard_instruct`. Suppresses CoT by default (`think_mode=OFF`) and on utility calls; toggle CoT anytime with `/think`. |
| **2. Built-In Current Time & World Clock Skill** | `resolve_local_timezone`, `is_clock_skill_query`, `build_live_clock_skill_context`, `get_dual_clock_str` | Injects live local (`PDT`/`PST`, guarding against headless `Etc/UTC` hosts) and `UTC/GMT` clocks (12h + 24h) plus world timezone conversions into `messages[-1]` without mutating `messages[0]` (preserving KV prefix cache). |
| **3. Multi-Query Search-by-Default Router** | `is_search_opt_out_query`, `determine_search_query`, `_expand_compound_queries`, `_build_fallback_search_query` | Queries the web **by default** to eliminate 2024 cutoff blindness, decomposing multi-entity/multi-clause prompts into up to 3 sub-queries (`q1 ; q2 ; q3`), resolving 3-turn coreferences, and opting out **only** for clock skill, pure code/math/formulas, or formatting commands. |
| **4. `Pass 0` Direct APIs, Yahoo `v8` Finance & Head+Tail `<table>` Scraper** | `fetch_web_context`, `fetch_yahoo_rich_cards`, `_fetch_yahoo_chart_quote`, `_resolve_direct_authoritative_sources`, `fetch_top_page_table_or_excerpt`, `_extract_yahoo_organic_fallback` | Resolves equities, indices, FX (`USD/JPY`, `EUR/USD`), and commodities (`WTI Crude`, `Gold`) via Yahoo `v8` Chart API; resolves USPTO Patents (`patents.google.com`), `arXiv` XML API + HTML benchmark tables, NOAA CO-OPS Tides, 5-digit NDBC Buoys, and `python.org`/`kernel.org` directly (`Pass 0`); extracts head (`[:8]`) + tail (`[-12:]`) HTML `<table>` rows; and fails over to keyless Yahoo Organic Search when `ddgs` rate-limits. |
| **5. Quantitative Auto-Think & False-Premise Guard** | `needs_quantitative_reasoning`, `BASE_SYSTEM_PROMPT` | Automatically enables internal reasoning on portfolio math, timezone deltas, and physics formulas even when `/think` is `OFF`, and enforces a **False-Premise & Trick-Question Guard** to reject fabricated premises when contradicted by live web data. |
| **6. 3-Tier Context-Aware Verbatim Window & Batch Archivist** | `resolve_memory_tier`, `compress_memory_batch`, `estimate_history_tokens` | Auto-scales verbatim window by model context length (`15/30/40` pairs, `10K/20K/26K` tokens, `5/8/10` batch eviction) while staying below the `32K` TT-Metal prefill cliff, keeping `messages[0]` byte-identical between batch evictions (**12/12 `100%` multi-needle recall at `~18.6K` tokens**). |
| **7. Automatic Section-Parallel Map-Reduce Generation** | `detect_auto_parallel_sections`, `plan_parallel_sections`, `stream_parallel_sections` | Automatically decomposes multi-part (`1) 2)`), multi-ticker ($\ge 2$), multi-entity (`X and Y`), deep-dive, and exhaustive table (`"List all..."`) prompts into **2–5 simultaneous `vLLM` streams** (`2.8×–4.2×` effective throughput speedup). |
| **8. 50-Line Hybrid Transient Streamer & Scrollback Commit** | `_BoundedMarkdownView`, `ProgressiveMarkdownStreamer` | Streams up to `50` lines (clamped to `console.size.height - 2`) of live formatted Markdown above a 1-line token counter (`⚡ Streaming (N tok)`), then commits completed Markdown blocks once to permanent SSH scrollback. |

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
   • 3-tier context-aware verbatim sliding window (`MAX_PAIRS = 15/30/40` pairs, `MAX_VERBATIM_TOKENS = 10K/20K/26K` tokens, batch-evicting `5/8/10` pairs based on `max_model_len`)
   • Active user turn (with 2026 web data + chronological table & exhaustive-list synthesis rules)
   • Automatic tokenizer prefill, `enable_thinking`, `reasoning_effort`, & stop-token adaptation based on `MODEL_PROFILE`
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
   • Batch-evicts `EVICT_BATCH_PAIRS` (`5/8/10`) oldest turn pairs when exceeding `MAX_PAIRS` or `MAX_VERBATIM_TOKENS` (`compress_memory_batch`)
   • Auto-saves `chat_cache.json` and `.chat_prompt_history` after every completed turn
```

### Key Architectural Highlights

* **4-Profile Model & Reasoning Auto-Detection (`detect_model_profile`):**
  Queries `/v1/models` at startup to discover the active model ID and `max_model_len`, then configures one of four profiles:
  1. **`vllm_reasoning_parser` (`Qwen3.5/3.6/3.8` & `openai/gpt-oss-120b/20b`):** Handles models where `vLLM` separates Chain-of-Thought into `message.reasoning` / `delta.reasoning_content`. For `Qwen3.x` models (`--reasoning-parser qwen3`), passes `chat_template_kwargs: {"enable_thinking": False/True}`. For OpenAI Harmony models (`openai/gpt-oss-120b`), passes `reasoning_effort: "low"` for fast default/utility calls and `"medium"` when `/think` or `needs_quantitative_reasoning()` is active.
  2. **`deepseek_qwen3_bridge` (`DeepSeek-R1-Distill` on `Qwen3` tokenizer):** Maps `<think>`/`</think>`/`<｜end▁of▁sentence｜>` to Qwen3 token IDs `151648` (`<|box_start|>`), `151649` (`<|box_end|>`), and `151643` (`<|endoftext|>`), injecting a closed 1-sentence micro-thought prefill when `think_mode = OFF` to skip CoT latency without reasoning leaks.
  3. **`native_think`:** Handles literal `<think>` / `</think>` tags inside `message.content` (native `DeepSeek-R1`, `Qwen3-32B`, `QwQ-32B`).
  4. **`standard_instruct`:** Standard chat completion payloads for non-reasoning models (`Llama-3.1/3.3`, `Gemma`, `Mistral`), with prompt-guided `<think>...</think>` support when `/think` is toggled.

* **3-Tier Context-Aware Memory Scaling & KV Prefix-Cache Preservation (`resolve_memory_tier` & `compress_memory_batch`):**
  * **Why 1-by-1 Sliding Windows Thrash the KV Prefix Cache:** Folding 1 evicted turn pair into `summary_buffer` on every turn mutates `messages[0]` (the system prompt), invalidating 100% of `vLLM`'s sequential prefix cache on every turn.
  * **3-Tier Hysteresis Scaling (Capped Below the 32K Prefill Cliff):** Automatically scales the verbatim window from `MODEL_PROFILE["max_model_len"]` while keeping `verbatim + ~2K web context <= ~28K tokens` to avoid crossing the `32,768`-token (`32K`) TT-Metal prefill compilation cliff on Tenstorrent Blackhole (`P300X2`):
    * **Tier 1 (`max_model_len <= 32,768`):** `MAX_PAIRS = 15`, `MAX_VERBATIM_TOKENS = 10,000`, `EVICT_BATCH_PAIRS = 5`
    * **Tier 2 (`32,769 – 262,143`, e.g. `131K` models):** `MAX_PAIRS = 30`, `MAX_VERBATIM_TOKENS = 20,000`, `EVICT_BATCH_PAIRS = 8`
    * **Tier 3 (`max_model_len >= 262,144`, e.g. `Qwen3.8-27B` `262K`):** `MAX_PAIRS = 40`, `MAX_VERBATIM_TOKENS = 26,000`, `EVICT_BATCH_PAIRS = 10`
  * Between batch evictions (`5`, `8`, or `10` turns at a time), `messages[0]` remains 100% byte-identical, preserving full KV prefix-cache hit rates (see [`TT_PREFIX_CACHE_FIX.md`](./TT_PREFIX_CACHE_FIX.md)).

---

## 3. Benchmarks & Verification on Tenstorrent QuietBox 2 (`P300X2` Blackhole)

### A. 23-Prompt Web Query Accuracy & Performance Comparison Across 3 Frontier Models

[`cjllm.py`](./cjllm.py) was evaluated on **Tenstorrent QuietBox 2 (`P300X2`, 4× `p300c` dual-chip Blackhole cards = 8 Tensix dies, 128 GB GDDR6)** across three open-weight models spanning dense `27B`, dense `32B`, and `120B` Mixture-of-Experts (`MoE`) architectures against a **3-Tier Ground-Truth Oracle** (Tier 1 primary APIs: Yahoo Finance `v8`, NOAA CO-OPS/NDBC, Google Patents/USPTO, official `arXiv` XML API; Tier 2 canonical upstream registries; Tier 3 adversarial false-premise rejection rules):

| Metric / Capability | **`Qwen/Qwen3.8-27B`** | **`openai/gpt-oss-120b`** | **`DeepSeek-R1-Distill-Qwen-32B`** |
| :--- | :---: | :---: | :---: |
| **Architecture & Active Params** | Dense `27B` | MoE `120B` (`5.1B` active, `128` experts) | Dense `32B` |
| **TT-Metal Container Stack** | `v0.25.0` (`qwen38_27b_qb2`) | `v0.17.0-8c48a10` (`models.demos.gpt_oss`) | `v0.17.0-pr55630` (`tt_transformers`) |
| **Max Context Length (`max_model_len`)** | **`262,144`** | `131,072` | `131,072` |
| **`cjllm` Auto-Scaled Memory Tier** | **Tier 3** (`40` pairs / `26K` tok / `10` evict) | **Tier 2** (`30` pairs / `20K` tok / `8` evict) | **Tier 2** (`30` pairs / `20K` tok / `8` evict) |
| **`cjllm` Reasoning Profile** | `vllm_reasoning_parser` (`enable_thinking`) | `vllm_reasoning_parser` (`reasoning_effort`) | `deepseek_qwen3_bridge` (`<\|box_start\|>`) |
| **Single-Stream Warm TTFT** | `566.2 ms` | `1,195.8 ms` | **`87.0 ms`** |
| **Single-Stream Decode Speed** | `8.5 tok/s` | **`21.1 tok/s`** | `18.4 tok/s` |
| **23-Prompt Benchmark Score (`Max 23.0`)** | **`22.85 / 23.0`** | **`22.70 / 23.0`** | **`22.55 / 23.0`** |
| **Single-Run Accuracy (`%`)** | **`99.3%`** | **`98.7%`** | **`98.0%`** |
| **Full Passes (`1.00`)** | **`22 / 23`** | **`21 / 23`** | `20 / 23` |
| **Partials (`0.85`)** | **`1 / 23`** | `2 / 23` | `3 / 23` |
| **Fails (`0.00`)** | **`0 / 23`** | **`0 / 23`** | **`0 / 23`** |

* **Why Each Model Excels at a Different Workload on QB2 (`P300X2`):**
  * **`Qwen/Qwen3.8-27B` (`99.3%` Accuracy — Best All-Around & Long-Context Synthesis):** Highest accuracy (`22/23` full passes), `262,144` max context (`Tier 3: 40 pairs / 26K verbatim tokens`), and clean native `--reasoning-parser qwen3` integration.
  * **`openai/gpt-oss-120b` (`98.7%` Accuracy, `21.1 tok/s` — Fastest Single-Stream Decode):** Because only `5.1B` parameters (out of `120B` total across `128` experts) are active per token, single-stream decode reaches **`21.1 tok/s`**—faster than both dense `27B` and `32B` models—while holding `61 GB` of `bfp8` weights across the 4 cards.
  * **`DeepSeek-R1-Distill-Qwen-32B` (`98.0%` Accuracy, `87 ms` TTFT — Fastest Multi-Stream Parallel Throughput):** Lowest warm TTFT (`87 ms`) and highest continuous-batching throughput across 4–32 simultaneous streams (up to **`227.8 tok/s`** aggregate decode).

#### Domain-by-Domain Verification Summary (`23` Live & Post-Cutoff Prompts)

| Domain / Test Suite | What It Verifies | `Qwen3.8-27B` | `gpt-oss-120b` | `DeepSeek-R1-32B` | Verified Ground-Truth Result on `P300X2` |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **1. Global Leaders, Pope & CEOs (`2` prompts)** | Post-2024 political & Fortune 500 turnover (`UK/Canada PMs`, `Pope`, `Boeing/Starbucks/Intel CEOs`) | **2/2** | **2/2** | **2/2** | UK PM `Andy Burnham`, Canada PM `Mark Carney`, `Pope Leo XIV`, and CEOs `Kelly Ortberg` (Boeing), `Brian Niccol` (Starbucks), & `Lip-Bu Tan` (Intel). |
| **2. Sports & Award Winners (`3` prompts)** | Chronological tail rows (`Super Bowl LVIII–LX`, `World Series`, `2025/2026 Oscars`, `Nobel Physics`) | **3/3** | **3/3** | **3/3** | `Seattle Seahawks` / MVP `Kenneth Walker III` (Super Bowl LX), `LA Dodgers` (World Series), *Anora* & *One Battle After Another* (Oscars), and `Francis Halzen` (Nobel Physics). |
| **3. Software, OS & Kernel Releases (`1` prompt)** | Direct release registries (`python.org`, `ubuntu.com`, `kernel.org`) | **1/1** | **1/1** | `0.85` | `Python 3.15.2`, `Ubuntu 26.04.1 LTS (Resolute Raccoon)` *(distinguished from interim `26.10`)*, and `Linux Kernel 7.2.13` stable (`7.3` mainline). |
| **4. Hardware Spec Comparisons (`1` prompt)** | Multi-GPU spec tables without false-positive `RTX` stock cards (`RTX 5090` vs `RTX 4090`) | **1/1** | **1/1** | **1/1** | Full Markdown table: `RTX 5090` (`Blackwell GB202`, `32 GB GDDR7`, `1,792 GB/s`, `575 W`, `$1,999`) vs `RTX 4090` (`Ada AD102`, `24 GB GDDR6X`, `1,008 GB/s`, `450 W`, `$1,599`). |
| **5. Macroeconomics, FX & Commodities (`2` prompts)** | Non-ticker finance (`Fed target rate`, `US CPI`, `USD/JPY`, `EUR/USD`, `WTI Crude Oil`) | **2/2** | **2/2** | **2/2** | FOMC target range (`3.75%–4.00%`, `3.88%` eff.), BLS CPI (`3.7%` YoY), live `USD/JPY` (`~157.9–158.0`), `EUR/USD` (`$1.114`), and `WTI Crude` (`~$101.8–$102.0/bbl`). |
| **6. Spaceflight & Geopolitics (`1` prompt)** | Mission flight logs (`SpaceX Starship`) + alliance tables (`NATO` members) | **1/1** | `0.85` | **1/1** | `Starship Flight 14` (`Sept 28, 2026`, Launch Successful) and `32` NATO member states (newest: `Finland [Apr 4, 2023]`, `Sweden [Mar 7, 2024]`). |
| **7. False-Premise / "Trick" Queries (`3` prompts)** | Sycophancy & fabrication resistance (`Biden 2024 EV`, `OpenAI ticker`, `NVIDIA–ARM completion`) | **3/3** | **3/3** | **3/3** | **3/3 Premise Rejections:** States Biden withdrew in July 2024 (`0` EV), OpenAI is private with no public ticker (`$842.24` secondary implied price), and NVIDIA–ARM was terminated Feb 7, 2022. |
| **8. Global Rankings & Leaderboards (`1` prompt)** | Top-10 Market Cap companies + Top-5 Forbes Billionaires tables | **1/1** | **1/1** | **1/1** | Top 10 Market Cap table (`Nvidia $5.2–5.3T`, `Apple $5.1T`, `Alphabet $3.9T`, `Microsoft $3.9T`...) and Top 5 Forbes Billionaires (`Musk $839B`, `Page $257B`, `Brin $237B`, `Bezos $224B`, `Zuckerberg $222B`). |
| **9. Legal & Supreme Court Landmarks (`1` prompt)** | Post-cutoff constitutional & administrative law (`Loper Bright`, `TikTok v. Garland`) | **1/1** | **1/1** | **1/1** | Summarizes *Loper Bright* (`603 U.S. 369` overruling *Chevron* deference) and *TikTok, Inc. v. Garland* (`604 U.S. 56` upholding divestiture). |
| **10. Multi-Hop Follow-Ups (`2` prompts)** | 2-turn pronoun & event resolution (`"Who won the last 3 Super Bowls?"` $\rightarrow$ `"MVP of the most recent one and final score?"`) | **2/2** | **2/2** | **2/2** | Resolves `"the most recent one"` to Super Bowl LX (2026), naming MVP `Kenneth Walker III` and final score `Seahawks 29 – Patriots 13`. |
| **11. Patent Filings & Portfolios (`2` prompts)** | Direct USPTO / Google Patents lookup (`US Patent 11,487,979`) & Tenstorrent NoC/tensor patents | **2/2** | **2/2** | `1.85` | Exact title, assignee (`Capital One Services LLC`), inventors, filing/grant dates, and Claim 1 for `US11487979` + Tenstorrent USPTO granted patents (`11,301,264`, `11,269,628`, `11,245,643`) & published apps (`20220245009` NoC overlay). |
| **12. Academic & `arXiv` Papers (`2` prompts)** | Direct `arXiv` XML/HTML lookup (`arXiv:2501.12948`) & literature comparison (`FlashAttention-3` + `Quantum LDPC`) | `1.85` | `1.85` | `1.85` | Extracts *DeepSeek-R1* (`arXiv:2501.12948`, authors, submission date, `AIME 2024`) and comparative table for *FlashAttention-3* (`arXiv:2407.08608`, `740 TFLOPs/s`) & *Quantum LDPC Codes* (`arXiv:2111.03654`, Panteleev & Kalachev). |
| **13. Tides & NBA Standings (`2` prompts)** | Direct NOAA CO-OPS Station `9410840` tide tables & 15-team NBA Western Conference standings | **2/2** | **2/2** | **2/2** | Exact today/tomorrow Santa Monica/Malibu high & low tide times/heights (`MLLW`) and full 1–15 Western Conference standings table. |

---

### B. 32-Stream Continuous-Batching & "Lost in the Middle" Recall Benchmarks (`P300X2`)

#### 32-Stream Continuous-Batching Scaling (`DeepSeek-R1-Distill-Qwen-32B`)
| Simultaneous Streams | Total Output Tokens | Total Wall-Clock Time | TTFT (`min / avg / max`) | Per-Stream Decode Speed | Aggregate Decode Throughput | Net Wall-Clock Throughput | Scaling vs. 1 Stream |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 stream** | `128` | `7.04 s` | `87 / 87 / 87 ms` | **`18.4 t/s`** | **`18.4 t/s`** | **`18.2 t/s`** | **`1.00×`** |
| **4 streams** | `512` | `8.12 s` | `91 / 252 / 306 ms` | **`16.3 t/s`** | **`65.1 t/s`** | **`63.1 t/s`** | **`3.47×`** |
| **8 streams** | `1,024` | `9.43 s` | `95 / 523 / 587 ms` | **`14.4 t/s`** | **`115.0 t/s`** | **`108.6 t/s`** | **`5.97×`** |
| **16 streams** | `2,048` | `12.87 s` | `519 / 873 / 1,154 ms` | **`10.7 t/s`** | **`170.9 t/s`** | **`159.1 t/s`** | **`8.75×`** |
| **32 streams** | `4,096` | `20.21 s` | `117 / 2,209 / 2,284 ms` | **`7.1 t/s`** | **`227.8 t/s`** | **`202.7 t/s`** | **`11.15×`** |

#### "Lost in the Middle" Multi-Needle Recall (`30` Turn Pairs / `~18,567` Tokens, `24` Technical Distractor Turns)
| Conversation Depth | Turn Pair | Planted Needle | Model Recall Output at `~18,567` Tokens | Result |
| :--- | :--- | :--- | :--- | :---: |
| **7% (Early)** | Pair 2 | NOAA Buoy `46054` gust **`27.4 knots`**, water temp **`58.3 °F`** | `Peak wind gust of 27.4 knots, water temperature of 58.3°F` | **PASS (2/2)** |
| **30% (Early-Mid)** | Pair 9 | 4-worker LUT QC-LDPC FER **`1.42e-5`**, $E_b/N_0$ **`3.75 dB`** | `FER of 1.42e-5, Eb/N0 of 3.75 dB` | **PASS (2/2)** |
| **50% (Exact Middle)** | Pair 15 | Broadcom (`AVGO`) **`137 shares`** @ **`$1,084.60`** | `Limit order for 137 shares at $1,084.60 per share` | **PASS (2/2)** |
| **53% (Exact Middle)** | Pair 16 | Malibu Surfrider low tide **`-0.8 ft`** @ **`3:42 PM PDT`** | `Afternoon low tide of -0.8 ft at 3:42 PM PDT` | **PASS (2/2)** |
| **77% (Late-Mid)** | Pair 23 | `P300X2` `l1_small_size` **`24,576`** B, `trace_region_size` **`402,653,184`** B | `l1_small_size of 24,576 bytes, trace_region_size of 402,653,184 bytes` | **PASS (2/2)** |
| **97% (Recent)** | Pair 29 | Generac (`GNRC`) close **`$162.85`**, P/E **`23.40`** | `Closing price of $162.85, P/E ratio of 23.40` | **PASS (2/2)** |
| **Post-Turn-31 Overflow** | Evicted Pairs 1–8 | Batch-evict oldest 8 pairs (`31 → 23` pairs) into `summary_buffer` | Retained `NOAA Buoy 46054: Peak wind gust of 27.4 knots (NW), sea surface temp 58.3°F` verbatim | **PASS** |

---

## 4. Usage & Commands

### Installation & Starting the Harness
```bash
pip install -r requirements.txt

# Run via convenience script or directly with automatic model, reasoning parser, & context-tier detection:
./run_chat
# or:
python3 cjllm.py
```

### Optional CLI Arguments
By default, `--max-pairs`, `--max-verbatim-tokens`, and `--evict-batch` are **auto-scaled from the model's `max_model_len`** (`15 / 10,000 / 5` for $\le 32\text{K}$ models; `30 / 20,000 / 8` for $64\text{K}–131\text{K}$ models; `40 / 26,000 / 10` for $\ge 256\text{K}$ models like `Qwen/Qwen3.8-27B`). You can optionally override any of them via CLI flags:
```bash
python3 cjllm.py \
  --url http://localhost:8000/v1 \
  --model Qwen/Qwen3.8-27B \
  --max-pairs 40 \
  --max-verbatim-tokens 26000 \
  --evict-batch 10 \
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
| [`cjllm.py`](./cjllm.py) | Primary model-agnostic `vLLM` inference harness (`~2,900` LoC) with 4-profile reasoning/tokenizer bridge, multi-query search router, `Pass 0` direct API resolvers (Yahoo `v8`, NOAA Tides/Buoys, USPTO Patents, `arXiv`), head+tail HTML `<table>` scraper, 3-tier context-aware memory scaling, automatic parallel Map-Reduce decoding, and 50-line hybrid SSH streamer |
| [`run_chat`](./run_chat) | Convenience launcher for `cjllm.py` (passes through any CLI arguments) |
| [`requirements.txt`](./requirements.txt) | Python package dependencies (`openai`, `rich`, `ddgs`, `primp`) |
| [`TT_PREFIX_CACHE_FIX.md`](./TT_PREFIX_CACHE_FIX.md) | Technical note on the `tt_transformers` (`PR #55630`) untraced prefix-cache page-table patch for Tenstorrent QuietBox 2 (`P300X2`) |
| [`.gitignore`](./.gitignore) | Git ignore rules (`__pycache__/`, `*.swp`, `.chat_prompt_history`, `chat_cache.json`, private test protocols/scripts) |
| [`chat_cache.json`](./chat_cache.json) | Default persisted conversation history and rolling summary buffer |
| [`README.md`](./README.md) | Motivation, harness comparison, architecture, 3-model verification benchmarks, and usage documentation |
