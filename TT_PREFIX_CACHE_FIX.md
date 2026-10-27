# Tenstorrent `tt_transformers` Prefix-Cache Page-Table Fix (`PR #55630`) on QuietBox 2 (`P300X2`)

**Date:** October 26, 2026  
**Project:** [`cjllm-search-harness`](file:///z:/cjllm-search-harness/cjllm.py)  
**Hardware:** Tenstorrent QuietBox 2 (`p300x2`, 4x Blackhole `p300c` chips)  
**Host Stack:** `tt-kmd 2.11.0`, `tt-system-firmware 19.15.0`  
**Container Image:** `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.17.0-8c48a10-f52987a`

---

## 1. Symptom Observed

During multi-turn conversations in [`cjllm.py`](file:///z:/cjllm-search-harness/cjllm.py) on the QuietBox 2 (`p300x2`), once the conversation history had accumulated ~2,000+ cached tokens and a new turn arrived with a large live web-search payload (bringing the cumulative prompt length to ~3,855 tokens, e.g. `"What happened on the DOW, S&P, and Nasdaq today?"`), the model suddenly emitted a single degenerate token (`"!"`) or repeated loops instead of answering the prompt.

 [`cjllm.py`](file:///z:/cjllm-search-harness/cjllm.py)'s post-turn quality gate caught the degenerate 1-character response and printed:
```text
[Warning] Discarded low-quality or corrupted assistant turn from history.
```
Meanwhile, the exact same prompt and conversation history worked without issue on **NVIDIA DGX Spark** (standard CUDA `vLLM`).

---

## 2. Root Cause (`tt-metal` Commit `8c48a10` / Fixed Upstream in `PR #55630`)

The bug was **not** caused by the `DeepSeek-R1-Distill-Qwen-32B` model weights or by [`cjllm.py`](file:///z:/cjllm-search-harness/cjllm.py). It was a page-table slicing bug in Tenstorrent's `tt_transformers` backend (`/home/container_app_user/tt-metal/models/tt_transformers/tt/generator.py`) inside the `0.17.0-8c48a10-f52987a` container image:

1. On `P300X2`, `trace_prefill_supported_seq_lens` traces prefill buckets up to `1024` tokens (`[128, 256, 512, 1024]`).
2. When a multi-turn request has `num_cached_tokens > 0` (from vLLM Automatic Prefix Caching) **and** the uncached suffix requires a `2048`-token or `4096`-token prefill bucket (such as a live web search with Yahoo Finance cards + DuckDuckGo snippets), `enable_trace_current_prompt` evaluates to `False` (**untraced prefill path**):
   ```text
   INFO | models.tt_transformers.tt.generator:prefill_forward_text:721 - Prefilling User 1 up to 3855 tokens
   INFO | models.tt_transformers.tt.generator:prefill_forward_text:762 - Prefill seq len: 2048, max_prefill_chunk_size: 131072, trace: False
   ```
3. In `tt-metal` commit `8c48a10` (June 25, 2026), `Generator.prefill_forward_text` called `self._get_prefill_user_page_table(...)` without `use_full_prompt_len=True` on the untraced path.
4. Without `use_full_prompt_len=True`, `_get_prefill_user_page_table` truncated the user's KV page table to `prefill_seq_len` (`2048` tokens = `32` blocks of 64 tokens) instead of the cumulative `seq_len` (`3855` tokens = `61` blocks).
5. Downstream, `prefill_forward_single_user_text` saw that `61` blocks were needed for `seq_len + num_cached_tokens`, so it **padded blocks `32..60` with `torch.zeros`** and then sliced `chunk_page_table` at the absolute block offset of the uncached suffix (`blocks 31..60`).
6. Because blocks `32..60` were all `0`, `ttnn.experimental.paged_fill_cache` interpreted every `0` as **physical KV block `0`**, overwriting the request's own system prompt at the start of the KV cache and producing degenerate output (`"!"`).

Tenstorrent fixed this exact bug upstream in [`tenstorrent/tt-metal` PR `#55630` (`8b96a36e8e9ba54f22f5c05c3436343c8301c236`, Sept 8, 2026)](https://github.com/tenstorrent/tt-metal/commit/8b96a36e8e9ba54f22f5c05c3436343c8301c236) (*"tt_transformers: span cached tokens in untraced prefill page table"*).

---

## 3. Why `git pull` on `tt-inference-server` Alone Did Not Upgrade `Qwen3-32B` on `P300X2`

Inspecting `tenstorrent/tt-inference-server` (`main`) showed:
* In `workflows/model_specs/prod/llm.yaml`, Tenstorrent's automated release bot (`create_post_release_pr`) only bumps `(model, device)` leaves that were executed in that specific release's CI qualification matrix (`models-ci-config.json`).
* While `v0.25.0` (`tt_metal_commit: "978f93b"`, `vllm_commit: "a723030"`) updated `google/gemma-4-31B-it` (`P300X2`), `Qwen/Qwen3.8-27B` (`P300X2`), and `Qwen/Qwen3-32B` (`BLACKHOLE_GALAXY`), the `Qwen/Qwen3-32B` entry for `P300X2` (lines `365–393` of `prod/llm.yaml`) was left pinned to `version: "0.17.0"`, `tt_metal_commit: "8c48a10"`, `vllm_commit: "f52987a"`.
* Furthermore, `v0.25.0` introduced new `P300X2` mesh/fabric settings (`MESH_DEVICE: "P150x4"`, `fabric_config: FABRIC_1D`, `l1_small_size: 24576`) that have not yet been formally qualified for the `Qwen/Qwen3-32B` `tt_transformers` leaf in `prod/llm.yaml`.

---

## 4. How the Fix Was Applied

To keep the hardware-qualified `0.17.0-8c48a10-f52987a` runtime for `Qwen3-32B` on `P300X2` while fixing native prefix caching (so [`cjllm.py`](file:///z:/cjllm-search-harness/cjllm.py) needs no `[Turn-ID]` workaround and behaves identically on QB2 and DGX Spark):

### A. Surgical Upstream Patch (`PR #55630`) Applied to `generator.py`
Inside the container (`/home/container_app_user/tt-metal/models/tt_transformers/tt/generator.py` at line `767`), Tenstorrent's official diff from `PR #55630` (`8b96a36e8e9ba54f22f5c05c3436343c8301c236`) was applied:

```diff
--- /home/container_app_user/tt-metal/models/tt_transformers/tt/generator.py.orig
+++ /home/container_app_user/tt-metal/models/tt_transformers/tt/generator.py
@@ -767,6 +767,16 @@
                 # For batched prefill: pass full page_table (function handles slot placement)
                 # For non-batched prefill: pass sliced page_table for current user (like original code)
                 page_table_for_user = page_table if use_batched_prefill else page_table[idx : idx + 1]
+                # A resumed chunk needs a page table spanning the cached tokens
+                # too: ``prefill_forward_single_user_text`` slices
+                # ``chunk_page_table`` at an absolute block offset, so a
+                # chunk-width table leaves that slice inside its own zero pad and
+                # ``paged_fill_cache`` writes the chunk into physical block 0.
+                # ``seq_len`` is the cumulative prompt length; ``prefill_seq_len``
+                # is only this chunk's padded width.
+                page_table_use_full_len = bool(
+                    not use_batched_prefill and not enable_trace_current_prompt and num_cached_tokens
+                )
                 page_table_user = self._get_prefill_user_page_table(
                     page_table_for_user,
                     kv_cache[model_id],
@@ -776,6 +786,7 @@
                     use_batched_prefill=use_batched_prefill,
                     user_id=batch_user_ids if use_batched_prefill else user_id,
                     padded_batch_size=padded_batch if use_batched_prefill else None,
+                    use_full_prompt_len=page_table_use_full_len,
                 )
                 full_page_table_user = None
                 if enable_trace_current_prompt and not use_batched_prefill:
```

### B. Persistent Local Image Tag + Launch Script Integration
Because `run_docker_server.py` runs `docker pull` against the default tag whenever `--override-docker-image` is not passed (which overwrites local `docker commit` changes to `:0.17.0-8c48a10-f52987a`), we save the patched image under a dedicated local tag (`:0.17.0-8c48a10-f52987a-pr55630`) so `run.py` uses it directly without pulling over it.

### C. Removed `[Turn-ID]` Workaround from [`cjllm.py`](file:///z:/cjllm-search-harness/cjllm.py)
The temporary `detect_tt_transformers_backend()` and `[Turn-ID: ...]` system-prompt prefix bypass were backed out of [`cjllm.py`](file:///z:/cjllm-search-harness/cjllm.py#L235-L253). Cached tokens (`Automatic prefix caching`) now operate natively on QB2 just like on DGX Spark.

### D. Live Multi-Turn Verification on QB2 (`P300X2`)
Tested a 2-turn sequence against the patched container (`bcdf1f5c4045`) where Turn 1 populated the prefix cache and Turn 2 resumed from the cached prefix on the **untraced `2048`-token prefill path** (`Finished prefill for all users up to 7523 tokens`):
```text
Turn 1 (populate prefix cache):
  'Based on the web search data, here is the surf forecast for Malibu, CA for the ...'
Turn 2 (resumed untraced 2048 prefill — "What happened on the DOW, S&P, and Nasdaq today?"):
  'The DOW, S&P, and Nasdaq indices as of today (Monday, October 26, 2026) are as follows:
   - **Dow Jones (^DJI)**: 47,951.85 (-0.28%)
   - **S&P 500 (^GSPC)**: 7,089.14 (-0.25%)
   - **Nasdaq (^IXIC)**: 23,825.46 (-0.83%)'
```

---

## 5. Backups & How to Revert (`Back Out`) if Ever Needed

Full backups of the original unpatched state were saved before making any changes:

1. **Original `generator.py` Backup on Host:**
   * `/home/ttuser/cj_DeepSeek_32B_bac/generator.py.orig`
   * Inside container: `/home/container_app_user/tt-metal/models/tt_transformers/tt/generator.py.orig`
2. **Original Untouched Docker Image Tag on Host:**
   * `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.17.0-8c48a10-f52987a-orig`
3. **To Revert a Running Container Back to Unpatched `generator.py`:**
   ```bash
   docker exec <container_name> cp -p \
     /home/container_app_user/tt-metal/models/tt_transformers/tt/generator.py.orig \
     /home/container_app_user/tt-metal/models/tt_transformers/tt/generator.py
   ```
   Or simply run `/home/ttuser/cj_DeepSeek_32B_bac/run_DeepSeek32B.orig` (which pulls and runs the stock upstream `:0.17.0-8c48a10-f52987a` image from `ghcr.io`).
