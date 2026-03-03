# Detailed Evolutionary Search Workflow — With Function Signatures

> Extended from `docs/workflow_detailed.md`. Every function signature is sourced directly
> from the codebase. Arguments show their **default values** as they appear in the source;
> the "Called with" column shows what `full_pipeline.py` actually passes at runtime.

---

## 1. High-Level Overview

The system runs a loop of `--epochs` iterations. Each epoch:

1. An **Implementer** (LLM agent) proposes research ideas and generates code patches.
2. A **Scheduler** (external, HuggingFace-based) distributes patched codebases to GPU workers.
3. **Workers** run training jobs and log metrics to Weights & Biases (W&B).
4. The Implementer retrieves logs, ranks ideas, and updates an evolving database for the next epoch.

Entry point: `agent/full_pipeline.py` (command-line via `argparse`).

---

## 2. CLI Entry Point

### `full_pipeline.py` — `argparse` defaults

| Argument | Type | Default | Description |
|---|---|---|---|
| `--epochs` | `int` | `10` | Total number of evolutionary epochs |
| `--num_ideas_per_epoch` | `int` | `80` | Ideas generated and evaluated per epoch |
| `--continue_from_epoch` | `int` | `0` | Resume from this epoch (skips earlier epochs) |
| `--skip_log_retrieval_when_continue` | `flag` | `False` | Skip Step 5 on the resume epoch |
| `--skip_idea_generation_when_continue` | `flag` | `False` | Skip Steps 1–3 on the resume epoch |
| `--run_name` | `str` | `"nanogpt_claude_opus_bsz80"` | Unique run identifier used in all filenames |
| `--env_dir` | `str` | `"env/nanogpt"` | Path to the base training environment |
| `--entity` | `str` | `"hashimoto-group"` | W&B entity (org/user) |
| `--project` | `str` | `"nanogpt_ES_claude"` | W&B project name |
| `--model_name` | `str` | `"claude-opus-4-5"` | LLM backend for idea generation and diff generation |

---

## 3. Full Epoch State Machine — With Signatures

```
for epoch in range(start_epoch, epochs):           # full_pipeline.py:64
│
├─ STEP 1 — IDEA GENERATION
│   agent_call_idea(...)                            # full_pipeline.py:68
│   │
│   ├─ IF epoch == 0:
│   │   Loop (num_ideas / 10) batches:
│   │     agent_call_idea_simple(num_ideas=10, ...)  # agent.py:21
│   │     └─ LLM prompt: codebase context + "generate N diverse experiments"
│   │        thinking_budget=3000, temperature=1.0   # agent.py:124
│   │        → Parses [Experiment]...[Code Changes]...[End] blocks
│   │        → Appends to cache_file (cumulative JSON array)
│   │
│   └─ IF epoch >= 1:
│       agent_call_idea_evolutionary(...)            # agent.py:560 → evolutionary_search.py:527
│       │
│       ├─ update_database(run_name, epoch-1)        # evolutionary_search.py:529
│       │   └─ Reads training_logs_{run}/epochN/ranked_ideas.json
│       │      Merges into ideas_{run}/database.json (deduped, ranked)
│       │
│       ├─ Compute exploit/explore split:            # evolutionary_search.py:532
│       │   max_exploit_ratio = min(0.5 + 0.1*(epoch//2), 0.8)
│       │   num_exploit = round_down_to_10(total * max_exploit_ratio)
│       │   num_explore = total - num_exploit
│       │
│       ├─ EXPLOIT batches (num_exploit / 10 calls):
│       │   agent_call_idea_evolutionary_exploit(num_ideas=10, ...)
│       │   # evolutionary_search.py:203
│       │   ├─ Filter database.json:
│       │   │   GRPO:    best_eval_accuracy > 0.49      # :218
│       │   │   NanoGPT: 0 < lowest_val_loss < 3.255    # :222, :228
│       │   ├─ Sample up to top_k=100 from winners (random.sample)  # :238
│       │   ├─ Build prompt: "combine/refine these winning ideas"
│       │   │   thinking_budget=1500, temperature=1.0   # :344
│       │   └─ Cache-penalty: append all prior ideas to avoid duplicates # :336-343
│       │
│       └─ EXPLORE batches (num_explore / 10 calls):
│           agent_call_idea_evolutionary_explore(num_ideas=10, ...)
│           # evolutionary_search.py:371
│           ├─ Sample up to sample_k=100 from ALL database ideas (random.sample) # :384-388
│           ├─ Build prompt: "generate new ideas, avoid failed patterns"
│           │   thinking_budget=1500, temperature=1.0   # :501
│           └─ Cache-penalty: append all prior ideas    # :493-500
│
│   Output: ideas_{run_name}/ideas_epoch{N}.json
│           (flat JSON array of "[Experiment]...[Code Changes]..." strings)
│
├─ STEP 2 — CODE DIFF GENERATION
│   generate_code_diff_parallel(max_trials=10, ...)   # full_pipeline.py:72, agent.py:536
│   │
│   ThreadPoolExecutor(max_workers=10)                 # agent.py:548
│   │
│   Per idea (parallel):
│   _generate_code_diff_parallel_helper(idea_idx, ...) # agent.py:507
│   │
│   trial = 0
│   WHILE trial < 10:
│     trial += 1
│     ├─ IF trial > 1 and error_msg is not None:
│     │   generate_code_diff(...,
│     │     prev_diff_file = "diffs_dir/code_diff_idea_N.diff",
│     │     prev_diff_error = error_msg)              # agent.py:516
│     └─ ELSE:
│         generate_code_diff(...)                      # agent.py:518
│         # agent.py:288
│         ├─ Read idea text from idea_lst[idea_idx]
│         ├─ context_prompt(base_dir) → numbered codebase  # evolutionary_search.py:14
│         │   Walks env_dir, includes all .py and .sh files (except evaluate.py, fineweb.py, run.sh)
│         │   Format: "===== filepath =====\n1: line1\n2: line2\n..."
│         ├─ Build prompt: codebase + idea + format constraints
│         │   thinking_budget=3000, temperature=1.0, max_tokens=10000  # agent.py:348
│         ├─ Strip markdown fences (strip_response, token="```diff")   # agent.py:354
│         └─ Rewrite diff headers to variant paths:   # agent.py:358-366
│             "--- filename.py" → "--- repo_variants_{run}_epochN/idea_I/filename.py"
│             "+++ filename.py" → "+++ repo_variants_{run}_epochN/idea_I/filename.py"
│
│     ├─ Write raw diff to diffs_dir/code_diff_idea_N.diff            # agent.py:519-520
│     ├─ code_diff_fixer(diff_file)                                    # agent.py:521
│     │   # agent.py:228 — recomputes @@ hunk line counts from actual body lines
│     ├─ apply_code_diff(env_dir, new_repo_dir, diff_file)             # agent.py:525
│     │   # agent.py:370
│     │   ├─ shutil.copytree(main_repo_dir → new_repo_dir)
│     │   └─ subprocess: patch -p0 < diff_file  (cwd=project_root)
│     ├─ ON SUCCESS: break
│     └─ ON FAILURE: error_msg = str(e); rmtree(new_repo_dir)         # agent.py:528-534
│         (next trial passes error_msg back to LLM)
│
│   Silently skips ideas that fail all 10 trials (no output directory)
│
│   Output per successful idea:
│     diffs_{run_name}_epoch{N}/code_diff_idea_{i}.diff
│     repo_variants_{run_name}_epoch{N}/idea_{i}/   (full patched codebase copy)
│
├─ STEP 3 — STATS + UPLOAD
│   compute_idea_stats(...)                           # full_pipeline.py:75, compute_idea_stats.py:4
│   ├─ Lists idea_{i} dirs under repo_variants dir
│   └─ Output: idea_stats_{run_name}/epoch{N}.json
│
│   zip_and_upload_repo_variants(...)                 # full_pipeline.py:76, upload_repo_variants.py
│   └─ Zips each idea dir, uploads to HuggingFace
│      (path: /juice5b/scr5b/nlp/aihinton/repo_variants/{run_name}/epoch{N})
│
│   move_diffs_and_repo_variants(...)                 # full_pipeline.py:79
│   └─ shutil.move → diffs_claude/, repo_variants_claude/
│
├─ STEP 4 — FIXED WAIT (external GPU scheduler window)
│   time.sleep(90 * 60)                              # full_pipeline.py:82
│   │
│   [External system, not in this repo:]
│   Scheduler polls HuggingFace → allocates B200 GPUs → Workers run training
│   W&B run naming: {run_name}_epoch{N}_b200_idea_{i}  # retrieve_training_logs.py:61-65
│
└─ STEP 5 — LOG RETRIEVAL LOOP
    Load idea_stats → num_ideas_submitted = success_count  # full_pipeline.py:90-93
    last_num_logs_retrieved = 0
    WHILE num_logs_retrieved <= 0.30 * num_ideas_submitted:  # full_pipeline.py:96
      retrieve_training_logs(...)                    # retrieve_training_logs.py:68
      │
      ├─ W&B API: api.runs(entity/project)
      │   Filter by fnmatch: "{run_name}_epoch{N}_b200_idea_*"  # :63, :89-99
      │   Sort by run.created_at ascending (latest overwrites for same idea_id)
      │
      ├─ For each matched run:
      │   ├─ Download output.log → training_logs_{run}/epochN/idea_{i}/output.log
      │   ├─ IF GRPO:
      │   │   run.history()["eval/mean_reward"] → extract non-NaN values
      │   │   final_reward = max(eval_rewards)      # :147
      │   └─ IF NanoGPT:
      │       Parse output.log for "val_loss:" lines # :39-48
      │       Format: "step:N/M val_loss:3.21 train_time:1500ms"
      │       final_reward = min(val_losses)         # :149
      │       time_to_target: first step where val_loss <= 3.28
      │
      ├─ Write training_logs_{run}/epochN/idea_{i}/metrics.json
      ├─ Write training_logs_{run}/epochN/ranked_ideas.json
      │   GRPO: sorted descending by final_reward
      │   NanoGPT: sorted ascending (sentinel -999 pushed to end)
      │
      ├─ IF retrieved > 30% of submitted: break      # full_pipeline.py:101
      └─ ELSE: time.sleep(20 * 60)                  # full_pipeline.py:105

After all epochs: update_database(run_name, epochs-1)  # full_pipeline.py:108
```

---

## 4. Function Signatures Reference

### `agent/evolutionary_search.py`

---

#### `context_prompt(base_dir)`
**File:** `evolutionary_search.py:14`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `base_dir` | `str` | `"env_grpo"` | Root directory of the training environment to read |

**Returns:** `str` — Concatenated, line-numbered content of all `.py` and `.sh` files
in `base_dir` (excluding `evaluate.py`, `fineweb.py`, `run.sh`).
Format per file: `"===== filepath =====\n1: line\n2: line\n..."`

---

#### `update_database(run_name, epoch_num)`
**File:** `evolutionary_search.py:74`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `run_name` | `str` | `"GRPO-env-test"` | Run identifier; controls file paths and `grpo`/`nanogpt` branching |
| `epoch_num` | `int` | `0` | Epoch whose `ranked_ideas.json` is merged into the database |

**Called with (full_pipeline.py:68):**
```python
update_database(run_name=run_name, epoch_num=epoch-1)
# final call: update_database(run_name=run_name, epoch_num=epochs-1)
```

**Reads:**
- `ideas_{run_name}/ideas_epoch{epoch_num}.json` — raw idea strings
- `training_logs_{run_name}/epoch{epoch_num}/ranked_ideas.json` — `[{"idea_N": reward}, ...]`
- `ideas_{run_name}/database.json` (if exists) — existing database to merge into

**Writes:**
- `ideas_{run_name}/database.json` — merged, deduplicated, sorted database

**Returns:** `None`

**Dedup rule:** Keeps best result per `(epoch, idea_id)` pair.
GRPO: highest `best_eval_accuracy`; NanoGPT: lowest `lowest_val_loss` (excludes `< 2.5` as reward-hacking).

---

#### `agent_call_idea_evolutionary_exploit(num_ideas, idea_database, top_k, cache_file, env_dir, model_name)`
**File:** `evolutionary_search.py:203`  Decorated: `@retry(tries=3, delay=2)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_ideas` | `int` | `10` | Ideas to generate per batch |
| `idea_database` | `str` | `"ideas_GRPO-env-test/database.json"` | Path to `database.json` |
| `top_k` | `int` | `10` | Max winners to sample as parent ideas |
| `cache_file` | `str` | `"ideas_GRPO-env-test/ideas_epoch1_evolutionary.json"` | Output ideas JSON (cumulative) |
| `env_dir` | `str` | `"env_grpo"` | Training env dir (controls `grpo`/`nanogpt` prompt branch) |
| `model_name` | `str` | `"gpt-5"` | LLM backend |

**Called with (evolutionary_search.py:542):**
```python
agent_call_idea_evolutionary_exploit(
    num_ideas=10,
    idea_database=f"ideas_{run_name}/database.json",
    top_k=top_k,          # passed from full_pipeline: top_k=100
    cache_file=cache_file,
    env_dir=env_dir,
    model_name=model_name
)
```

**Winner filter:**
- GRPO: `best_eval_accuracy > 0.49`
- NanoGPT: `0 < lowest_val_loss < 3.255`

**LLM call:** `apiqa(..., claude_thinking_budget=1500, temperature=1.0)`

**Writes:** appends to `cache_file` (cumulative JSON array of idea strings)

**Returns:** `(response: str, thinking: str)` — raw LLM text and thinking trace

---

#### `agent_call_idea_evolutionary_explore(num_ideas, idea_database, sample_k, cache_file, env_dir, model_name)`
**File:** `evolutionary_search.py:371`  Decorated: `@retry(tries=3, delay=2)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_ideas` | `int` | `10` | Ideas to generate per batch |
| `idea_database` | `str` | `"ideas_GRPO-env-test/database.json"` | Path to `database.json` |
| `sample_k` | `int` | `100` | Max ideas to randomly sample from the full database for context |
| `cache_file` | `str` | `"ideas_GRPO-env-test/ideas_epoch1_evolutionary.json"` | Output ideas JSON (cumulative) |
| `env_dir` | `str` | `"env_grpo"` | Training env dir |
| `model_name` | `str` | `"gpt-5"` | LLM backend |

**Called with (evolutionary_search.py:546):**
```python
agent_call_idea_evolutionary_explore(
    num_ideas=10,
    idea_database=f"ideas_{run_name}/database.json",
    sample_k=sample_k,    # passed from full_pipeline: sample_k=100
    cache_file=cache_file,
    env_dir=env_dir,
    model_name=model_name
)
```

**LLM call:** `apiqa(..., claude_thinking_budget=1500, temperature=1.0)`

**Writes:** appends to `cache_file`

**Returns:** `(response: str, thinking: str)`

---

#### `agent_call_idea_evolutionary(total_num_ideas, run_name, epoch_num, top_k, sample_k, cache_file, env_dir, model_name)`
**File:** `evolutionary_search.py:527`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `total_num_ideas` | `int` | `200` | Total ideas to generate this epoch |
| `run_name` | `str` | `"GRPO-env-test"` | Run identifier |
| `epoch_num` | `int` | `1` | Current epoch (must be ≥ 1) |
| `top_k` | `int` | `10` | Passed to exploit function |
| `sample_k` | `int` | `100` | Passed to explore function |
| `cache_file` | `str` | `"ideas_GRPO-env-test/ideas_epoch1_evolutionary.json"` | Output file |
| `env_dir` | `str` | `"env_grpo"` | Training env dir |
| `model_name` | `str` | `"gpt-5"` | LLM backend |

**Exploit/explore split:**
```python
max_exploit_ratio = min(0.5 + 0.1 * (epoch_num // 2), 0.8)
num_exploit = (int(total_num_ideas * max_exploit_ratio) // 10) * 10
num_explore  = total_num_ideas - num_exploit
```

**Returns:** `None` (results written to `cache_file` by nested calls)

---

### `agent/agent.py`

---

#### `agent_call_idea_simple(num_ideas, cache_file, env_dir, model_name)`
**File:** `agent.py:21`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_ideas` | `int` | `10` | Ideas to generate per call |
| `cache_file` | `str` | `"agent/all_ideas.json"` | Cumulative output JSON (appended per call) |
| `env_dir` | `str` | `"env_grpo"` | Training env dir; selects GRPO vs NanoGPT prompt |
| `model_name` | `str` | `"gpt-5"` | LLM backend |

**Called with (agent.py:566 inside `agent_call_idea`):**
```python
agent_call_idea_simple(
    num_ideas=10,
    cache_file=f"ideas_{run_name}/ideas_epoch{epoch}.json",
    env_dir=env_dir,
    model_name=model_name
)
```

**LLM call:** `apiqa(..., claude_thinking_budget=3000, temperature=1.0)`

**Writes:** `cache_file` — appends parsed `[Experiment]...[Code Changes]...` strings

**Returns:** `(response: str, thinking: str | None)`

---

#### `agent_call_idea(num_ideas, cache_file, run_name, epoch_num, prev_ideas_file, prev_training_logs, top_k, sample_k, env_dir, model_name)`
**File:** `agent.py:559`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_ideas` | `int` | `200` | Total ideas this epoch |
| `cache_file` | `str` | `"ideas/all_ideas_claude_epoch2.json"` | Output JSON path |
| `run_name` | `str` | `"GRPO-env-test"` | Run identifier |
| `epoch_num` | `int` | `1` | Current epoch; 0 → simple, ≥1 → evolutionary |
| `prev_ideas_file` | `str` | `"ideas/all_ideas_0826_epoch1.json"` | Previous epoch ideas (unused by evolutionary path) |
| `prev_training_logs` | `str` | `"training_logs/epoch1_retrywrapper/"` | Previous logs dir (unused by evolutionary path) |
| `top_k` | `int` | `20` | Exploit pool size |
| `sample_k` | `int` | `100` | Explore sample size |
| `env_dir` | `str` | `"env_grpo"` | Training env dir |
| `model_name` | `str` | `"gpt-5"` | LLM backend |

**Called with (full_pipeline.py:68):**
```python
agent_call_idea(
    num_ideas=num_ideas_per_epoch,           # default 80
    cache_file=f"ideas_{run_name}/ideas_epoch{epoch}.json",
    run_name=run_name,
    epoch_num=epoch,
    prev_ideas_file=f"ideas_{run_name}/ideas_epoch{epoch-1}.json",
    prev_training_logs=f"training_logs_{run_name}/epoch{epoch-1}/",
    top_k=100,
    sample_k=100,
    env_dir=args.env_dir,
    model_name=args.model_name
)
```

**Returns:** `None`

---

#### `generate_code_diff(idea_idx, base_dir, variant_dir, idea_file, diff_dir, prev_diff_file, prev_diff_error, idea_lst, model_name)`
**File:** `agent.py:288`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `idea_idx` | `int` | `6` | Index into `idea_lst` or `idea_file` |
| `base_dir` | `str` | `"env_grpo"` | Base training env (read-only template) |
| `variant_dir` | `str` | `"repo_variants_testing"` | Output root; individual variant at `variant_dir/idea_{idx}` |
| `idea_file` | `str` | `"agent/all_ideas.json"` | Ideas JSON (used only if `idea_lst` is None) |
| `diff_dir` | `str` | `"diffs_testing"` | Directory for output `.diff` files |
| `prev_diff_file` | `str \| None` | `None` | Path to previous failed diff (for retry) |
| `prev_diff_error` | `str \| None` | `None` | Error message from previous patch attempt (for retry) |
| `idea_lst` | `list \| None` | `None` | In-memory ideas list (avoids re-reading file in parallel) |
| `model_name` | `str` | `"gpt-5"` | LLM backend |

**Called with (agent.py:520–522 inside `_generate_code_diff_parallel_helper`):**
```python
# First trial:
generate_code_diff(
    idea_idx=idea_idx, base_dir=env_dir, variant_dir=repo_dir,
    idea_file=idea_file, diff_dir=diffs_dir,
    idea_lst=idea_lst, model_name=model_name
)
# Retry trial (trial > 1 and error_msg is not None):
generate_code_diff(
    idea_idx=idea_idx, base_dir=env_dir, variant_dir=repo_dir,
    idea_file=idea_file, diff_dir=diffs_dir,
    prev_diff_file=f"{diffs_dir}/code_diff_idea_{idea_idx}.diff",
    prev_diff_error=error_msg,
    idea_lst=idea_lst, model_name=model_name
)
```

**LLM call:** `apiqa(..., claude_thinking_budget=3000, temperature=1.0, max_tokens=10000)`

**Returns:** `(thinking: str, response: str)` — thinking trace and unified diff text
(diff headers already rewritten to `variant_dir/idea_{idx}/filename.py`)

---

#### `code_diff_fixer(diff_file)`
**File:** `agent.py:228`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `diff_file` | `str` | `"diffs/code_diff_idea_0.diff"` | Path to unified diff file to fix in-place |

**Called with (agent.py:525):**
```python
code_diff_fixer(diff_file=f"{diffs_dir}/code_diff_idea_{idea_idx}.diff")
```

**Behavior:** Reads diff, recomputes actual old/new line counts for each `@@ ... @@` hunk
header by counting `-`/`+`/context lines in the hunk body, then rewrites the file if any
count was wrong.

**Returns:** `None` (modifies file in-place)

---

#### `apply_code_diff(main_repo_dir, new_repo_dir, diff_file)`
**File:** `agent.py:370`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `main_repo_dir` | `str` | `"env_grpo"` | Source template repo to copy from |
| `new_repo_dir` | `str` | `"repo_variants/idea_0"` | Destination for the patched copy |
| `diff_file` | `str` | `"diffs/code_diff_idea_0.diff"` | Diff to apply with `patch -p0` |

**Called with (agent.py:529):**
```python
apply_code_diff(
    main_repo_dir=env_dir,
    new_repo_dir=f"{repo_dir}/idea_{idea_idx}",
    diff_file=f"{diffs_dir}/code_diff_idea_{idea_idx}.diff"
)
```

**Behavior:**
1. `shutil.copytree(main_repo_dir, new_repo_dir, symlinks=True)` — fresh copy
2. `patch -p0 < {abs_diff_file}` with `cwd=project_root` (repo root, not env dir)

**Returns:** `Path` — resolved path to `new_repo_dir`

**Raises:** `RuntimeError(f"Failed to apply diff with patch:\n{stderr}")` on non-zero exit

---

#### `_generate_code_diff_parallel_helper(idea_idx, max_trials, env_dir, repo_dir, idea_file, diffs_dir, idea_lst, model_name)`
**File:** `agent.py:511`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `idea_idx` | `int` | — | Idea index (0-based) |
| `max_trials` | `int` | — | Retry limit before giving up silently |
| `env_dir` | `str` | — | Training env template dir |
| `repo_dir` | `str` | — | Output root for patched variant dirs |
| `idea_file` | `str` | — | Ideas JSON path |
| `diffs_dir` | `str` | — | Output root for `.diff` files |
| `idea_lst` | `list \| None` | `None` | In-memory idea list (avoids disk read per thread) |
| `model_name` | `str` | `"gpt-5"` | LLM backend |

**Returns:** `None` — success: `repo_dir/idea_{idx}/` exists; failure: directory absent

---

#### `generate_code_diff_parallel(max_trials, diffs_dir, repo_dir, env_dir, idea_file, idea_lst, model_name, total_workers)`
**File:** `agent.py:540`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_trials` | `int` | `10` | Per-idea retry limit |
| `diffs_dir` | `str` | `"diffs_epoch1_parallel_corrected"` | Output dir for `.diff` files |
| `repo_dir` | `str` | `"repo_variants_epoch1_parallel_corrected"` | Output dir for patched repos |
| `env_dir` | `str` | `"env_grpo"` | Template env dir |
| `idea_file` | `str` | `"ideas/all_ideas_0826_epoch1.json"` | Ideas JSON path |
| `idea_lst` | `list \| None` | `None` | In-memory alternative to `idea_file` |
| `model_name` | `str` | `"gpt-5"` | LLM backend |
| `total_workers` | `int` | `10` | `ThreadPoolExecutor` concurrency |

**Called with (full_pipeline.py:72):**
```python
generate_code_diff_parallel(
    max_trials=10,
    diffs_dir=f"diffs_{run_name}_epoch{epoch}",
    repo_dir=f"repo_variants_{run_name}_epoch{epoch}",
    env_dir=args.env_dir,
    idea_file=f"ideas_{run_name}/ideas_epoch{epoch}.json",
    model_name=args.model_name
)
```

**Returns:** `None` — results on disk in `diffs_dir/` and `repo_dir/`

---

### `agent/compute_idea_stats.py`

---

#### `compute_idea_stats(idea_file, repo_variants_dir, idea_stats_file)`
**File:** `compute_idea_stats.py:4`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `idea_file` | `str` | `"ideas/all_ideas_0826_epoch1.json"` | Ideas JSON (used for total count) |
| `repo_variants_dir` | `str` | `"repo_variants_epoch1_parallel"` | Dir containing `idea_{i}/` subdirs |
| `idea_stats_file` | `str` | `"idea_stats/epoch1.json"` | Output stats JSON path |

**Called with (full_pipeline.py:75):**
```python
compute_idea_stats(
    idea_file=f"ideas_{run_name}/ideas_epoch{epoch}.json",
    repo_variants_dir=f"repo_variants_{run_name}_epoch{epoch}",
    idea_stats_file=f"idea_stats_{run_name}/epoch{epoch}.json"
)
```

**Returns:** `None`

**Writes `idea_stats_file`:**
```json
{
  "successful_ideas": [0, 1, 3, 5],
  "failed_ideas": [2, 4],
  "success_count": 4,
  "total_ideas": 6,
  "success_percent": 66.67
}
```
`successful_ideas` = indices of `idea_{i}/` dirs that exist; `failed_ideas` = missing indices.

---

### `agent/upload_repo_variants.py`

---

#### `zip_and_upload_repo_variants(original_ideas, folder_path, run_name, epoch_num, upload_path, upload_to_hf, n_ideas_cap)`
**File:** `upload_repo_variants.py:27`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `original_ideas` | `str` | — | Source dir with `idea_{i}/` subdirs to zip |
| `folder_path` | `str` | — | Local staging path for zip files |
| `run_name` | `str \| None` | `None` | Run identifier (used in HF path) |
| `epoch_num` | `int \| None` | `None` | Epoch number (used in HF path) |
| `upload_path` | `str` | `"/juice5b/scr5b/nlp/aihinton/repo_variants/"` | Root on server/HF |
| `upload_to_hf` | `bool` | `True` | Whether to actually upload to HuggingFace |
| `n_ideas_cap` | `int` | `400` | Max number of `idea_*` dirs to zip/upload |

**Called with (full_pipeline.py:76):**
```python
zip_and_upload_repo_variants(
    original_ideas=f"repo_variants_{run_name}_epoch{epoch}",
    folder_path=f"/juice5b/scr5b/nlp/aihinton/repo_variants/{run_name}/epoch{epoch}",
    run_name=run_name,
    epoch_num=epoch
)
```

**HuggingFace repo:** `CLS/repo_variants` (created if not exists)

**HF path in repo:** `{run_name}/epoch{epoch_num}`

**Returns:** `None`

---

### `agent/full_pipeline.py`

---

#### `move_diffs_and_repo_variants(src_diffs, dst_diffs, src_repo, dst_repo)`
**File:** `full_pipeline.py:15`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `src_diffs` | `str` | — | Source diffs dir (e.g. `diffs_{run}_epoch{N}`) |
| `dst_diffs` | `str` | — | Destination parent dir (e.g. `diffs_claude/`) |
| `src_repo` | `str` | — | Source repo_variants dir |
| `dst_repo` | `str` | — | Destination parent dir (e.g. `repo_variants_claude/`) |

**Called with (full_pipeline.py:79):**
```python
move_diffs_and_repo_variants(
    src_diffs=f"diffs_{run_name}_epoch{epoch}",
    dst_diffs="diffs_claude",
    src_repo=f"repo_variants_{run_name}_epoch{epoch}",
    dst_repo="repo_variants_claude"
)
```

**Behavior:** `shutil.move(src, dst)` for both dirs. If destination already exists, removes
it first (handles re-run of the same epoch).

**Returns:** `None`

---

### `agent/retrieve_training_logs.py`

---

#### `get_run_name(run_name, epoch_num, idea_number)`
**File:** `retrieve_training_logs.py:51`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `run_name` | `str` | — | Run identifier |
| `epoch_num` | `int` | — | Current epoch |
| `idea_number` | `int \| None` | `None` | If set, returns exact name; else returns glob pattern |

**Returns:** `str`
- `idea_number=None`: `"{run_name}_epoch{N}_b200_idea_*"` (fnmatch glob)
- `idea_number=K`:   `"{run_name}_epoch{N}_b200_idea_{K}"` (exact)

---

#### `extract_metrics(series, metric_name)`
**File:** `retrieve_training_logs.py:26`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `series` | `pd.Series` | — | W&B history column, e.g. `run.history()["eval/mean_reward"]` |
| `metric_name` | `str` | — | Key to use in output dicts, e.g. `"eval_reward"` |

**Returns:** `list[dict]` — e.g. `[{"step": 100, "eval_reward": 0.48}, ...]`
(NaN rows are dropped)

---

#### `extract_metrics_nanogpt(lines, target_loss)`
**File:** `retrieve_training_logs.py:35`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `lines` | `list[str]` | — | Lines from `output.log` |
| `target_loss` | `float` | `3.28` | Val loss threshold for `time_to_target` |

Parses lines matching: `"step:N/M val_loss:X train_time:Yms"`

**Returns:** `(eval_rewards: list[dict], time_to_target: int)`
- `eval_rewards`: `[{"step": N, "val_loss": X, "train_time_ms": Y}, ...]`
- `time_to_target`: ms when `val_loss <= target_loss` first occurs; `-999` if never

---

#### `_retry_wandb(call, desc, max_attempts, base_delay)`
**File:** `retrieve_training_logs.py:10`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `call` | `callable` | — | Zero-argument lambda wrapping a W&B operation |
| `desc` | `str` | — | Human-readable label for logging |
| `max_attempts` | `int` | `9` | Maximum retry attempts |
| `base_delay` | `float` | `2.0` | Base delay in seconds; doubles each attempt (exponential backoff) |

**Returns:** Return value of `call()` on success

**Raises:** Re-raises last exception after `max_attempts` failures

---

#### `retrieve_training_logs(run_name, epoch_num, env_dir, entity, project, target_loss)`
**File:** `retrieve_training_logs.py:68`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `run_name` | `str` | — | Run identifier (controls `grpo`/`nanogpt` branching) |
| `epoch_num` | `int` | — | Current epoch |
| `env_dir` | `str` | `"env_nanogpt"` | Training env dir (currently only used for branching via `run_name`) |
| `entity` | `str` | `"hashimoto-group"` | W&B entity |
| `project` | `str` | `"nanogpt-training"` | W&B project |
| `target_loss` | `float` | `3.28` | NanoGPT target val loss for `time_to_target` |

**Called with (full_pipeline.py:97):**
```python
num_logs_retrieved, ranked_ideas_dicts = retrieve_training_logs(
    run_name=run_name,
    epoch_num=epoch,
    env_dir=args.env_dir,
    entity=args.entity,
    project=args.project
)
```

**W&B API:** `wandb.Api(timeout=300)`; run list fetched via `_retry_wandb` wrapper

**Writes:**
- `training_logs_{run_name}/epoch{N}/idea_{i}/output.log` — downloaded from W&B
- `training_logs_{run_name}/epoch{N}/idea_{i}/metrics.json` — extracted metrics
- `training_logs_{run_name}/epoch{N}/ranked_ideas.json` — sorted by final reward

**Returns:** `(num_logs_retrieved: int, ranked_ideas_dicts: list[dict])`
- `num_logs_retrieved` = count of runs where `output.log` was successfully downloaded
- `ranked_ideas_dicts` = `[{"idea_N": reward}, ...]` sorted by reward

---

## 5. JSON Data Contracts

### `ideas_{run_name}/ideas_epoch{N}.json`

```json
[
  "[Experiment] Add sequence-level GRPO loss normalization.\n[Code Changes] Modify compute_loss() in grpo.py ...",
  "[Experiment] Cosine annealing for clip epsilon.\n[Code Changes] ..."
]
```

- Written by `agent_call_idea_simple` / `agent_call_idea_evolutionary_exploit/explore`
- Each element is a raw string starting with `[Experiment]` and ending before `[End]`

### `ideas_{run_name}/database.json`

```json
[
  {
    "epoch": 0,
    "idea_id": 3,
    "idea": "[Experiment] ...[Code Changes] ...",
    "best_eval_accuracy": 0.65
  },
  {
    "epoch": 1,
    "idea_id": 7,
    "idea": "[Experiment] ...[Code Changes] ...",
    "lowest_val_loss": 3.21
  }
]
```

- GRPO entries use key `best_eval_accuracy`; NanoGPT entries use `lowest_val_loss`
- Sorted: GRPO descending by accuracy; NanoGPT ascending by loss (invalid/NaN entries excluded)
- Deduplicated: only best result per `(epoch, idea_id)` pair kept
- NanoGPT entries with `reward_value < 2.5` are excluded as reward-hacking (`evolutionary_search.py:114`)

### `idea_stats_{run_name}/epoch{N}.json`

```json
{
  "successful_ideas": [0, 1, 3, 5],
  "failed_ideas": [2, 4],
  "success_count": 4,
  "total_ideas": 6,
  "success_percent": 66.67
}
```

- `successful_ideas`: indices of `idea_{i}` dirs that exist in `repo_variants`
- `failed_ideas`: ideas where all 10 diff-generation trials failed

### `training_logs_{run_name}/epoch{N}/ranked_ideas.json`

```json
[
  {"idea_3": 0.72},
  {"idea_0": 0.65},
  {"idea_7": -999.0}
]
```

- GRPO: sorted descending by `eval_reward`; NanoGPT: ascending by `val_loss` (sentinels at end)
- `idea_key` matches `idea_{i}` where `i` is the numeric suffix of the W&B run name
- Sentinel `-999.0` = run had no eval metrics (crashed or no `output.log`)

### `training_logs_{run_name}/epoch{N}/idea_{i}/metrics.json` (GRPO)

```json
{
  "eval_rewards": [
    {"step": 100, "eval_reward": 0.48},
    {"step": 200, "eval_reward": 0.61}
  ],
  "train_rewards": [
    {"step": 50, "train_reward": 0.42}
  ]
}
```

### `training_logs_{run_name}/epoch{N}/idea_{i}/metrics.json` (NanoGPT)

```json
{
  "eval_rewards": [
    {"step": 500, "val_loss": 3.41, "train_time_ms": 120000},
    {"step": 1000, "val_loss": 3.28, "train_time_ms": 240000}
  ],
  "time_to_target": 240000
}
```

- `time_to_target`: ms elapsed when `val_loss <= 3.28` (first occurrence); `-999` if never reached
- Parsed from `output.log` lines matching: `step:N/M val_loss:X train_time:Yms`

---

## 6. Hardcoded Constants

| Constant | Value | Purpose | Source |
|---|---|---|---|
| GRPO baseline threshold | `0.49` | Filter exploit pool (winners only) | `evolutionary_search.py:218` |
| NanoGPT baseline threshold | `3.255` | Filter exploit pool (winners only) | `evolutionary_search.py:222` |
| NanoGPT reward-hacking filter | `2.5` | Skip suspiciously low val_loss (reward hacking) | `evolutionary_search.py:114` |
| Max exploit ratio | `0.8` | Cap on how many ideas are exploit vs. explore | `evolutionary_search.py:532` |
| Exploit ratio schedule | `min(0.5 + 0.1*(epoch//2), 0.8)` | Gradually anneal from 50% to 80% exploit | `evolutionary_search.py:532` |
| Exploit ratio rounding | `(N // 10) * 10` | Round exploit count to nearest 10 (batch size) | `evolutionary_search.py:535` |
| Thinking budget (idea gen) | `1500` tokens | Claude extended thinking for exploit/explore | `evolutionary_search.py:344,501` |
| Thinking budget (code diff) | `3000` tokens | Claude extended thinking for diff generation | `agent.py:348` |
| Thinking budget (simple) | `3000` tokens | Claude extended thinking for epoch 0 | `agent.py:124` |
| Max code diff retries | `10` | Per-idea retry limit before giving up | `agent.py:511` |
| Code diff workers | `10` | ThreadPoolExecutor max_workers | `agent.py:548` |
| Post-upload wait | `90 min` | Fixed sleep waiting for GPU jobs to start | `full_pipeline.py:82` |
| W&B poll interval | `20 min` | Sleep between log retrieval attempts | `full_pipeline.py:105` |
| W&B completion threshold | `30%` | Stop polling once ≥30% of ideas have logs | `full_pipeline.py:96,101` |
| Machine suffix | `"b200"` | Embedded in W&B run name pattern | `retrieve_training_logs.py:61` |
| W&B API timeout | `300 s` | Timeout for W&B API calls | `retrieve_training_logs.py:69` |
| W&B retry attempts | `9` | Max attempts for W&B operations with backoff | `retrieve_training_logs.py:11` |
| NanoGPT target loss | `3.28` | Threshold for `time_to_target` metric | `retrieve_training_logs.py:193` |
| Ideas per batch | `10` | LLM call granularity (ideas generated per call) | `agent.py:555, evolutionary_search.py:539` |
| Default ideas per epoch | `80` | CLI default for `--num_ideas_per_epoch` | `full_pipeline.py:45` |

---

## 7. Prompt Architecture

All prompts share the same skeleton:

```
[System message: "You are a research scientist..."]

[1. CODEBASE CONTEXT]
===== filepath =====
1: line1
2: line2
...
(All .py and .sh files except evaluate.py, fineweb.py, run.sh)
Generated by: context_prompt(base_dir)  →  evolutionary_search.py:14

[2. TASK DESCRIPTION]
(Role-specific: generate ideas / implement experiment / fix diff)

[3. PARENT IDEAS] (epochs ≥ 1 only)
Idea: [Experiment]...[Code Changes]...
Eval Accuracy/Final Validation Loss: X.XX

[4. CONSTRAINTS BLOCK] (model-type-specific)
GRPO: "not allowed to change evaluation logic / eval metrics / eval frequency / wandb name"
NanoGPT: "not allowed to change loss function / val hyperparams / 1500s time limit / wandb name"
         "do not break forward_with_cache / forward_safe autoregressive behavior"

[5. OUTPUT FORMAT SPEC]
[Experiment] ...
[Code Changes] ...
[End]
(For code diffs: "return a single diff file with no other text")

[6. DEDUPLICATION PENALTY] (if cache_file exists)
"Avoid any similar ideas to the following experiments that have been proposed before:
<all prior ideas concatenated>"
Generated by: checking os.path.exists(cache_file) in each generation function
```

### Per-function details

| Function | File:Line | Thinking Budget | Temperature | Context Injected |
|---|---|---|---|---|
| `agent_call_idea_simple` | `agent.py:21` | 3000 | 1.0 | Full codebase |
| `agent_call_idea_evolutionary_exploit` | `evolutionary_search.py:203` | 1500 | 1.0 | Full codebase + winners from database |
| `agent_call_idea_evolutionary_explore` | `evolutionary_search.py:371` | 1500 | 1.0 | Full codebase + random sample from database |
| `generate_code_diff` | `agent.py:288` | 3000 | 1.0 | Full codebase + idea + prev diff + prev error |

---

## 8. Key State Transitions

```
database.json state:
  ┌────────────────────────────────────────────────────────────┐
  │  Epoch 0 ideas                                             │
  │  → generate_code_diff_parallel → repo_variants            │
  │  → [GPU training] → ranked_ideas.json (W&B logs)          │
  │  → update_database() → database.json (epoch 0 entries)    │
  └────────────────────────────────────────────────────────────┘
  ┌────────────────────────────────────────────────────────────┐
  │  Epoch 1 ideas (read from database.json)                   │
  │  exploit: sample top winners, combine/refine               │
  │  explore: sample all past, generate new                    │
  │  → same pipeline as epoch 0                               │
  │  → update_database() APPENDS epoch 1 entries              │
  │    (dedup: keeps best per (epoch, idea_id) pair)           │
  └────────────────────────────────────────────────────────────┘
```

The `database.json` grows across all epochs. Entries are never deleted; only the best result per
`(epoch, idea_id)` is kept when the same run is seen multiple times (e.g., re-retrieval).

---

## 9. Multi-Model Support

The pipeline supports multiple LLM backends via `api.py`:

| Model string pattern | Backend | Notes |
|---|---|---|
| `"claude-*"` | Anthropic API direct | Supports extended thinking |
| `"global.anthropic.*"` | AWS Bedrock (Claude) | Supports extended thinking |
| `"gpt-*"`, `"o1-*"`, `"o3-*"`, `"o4-*"` | OpenAI Responses API | `is_o_model` path for reasoning |
| `"deepseek-chat"` / `"deepseek-reasoner"` | DeepSeek API | Optional discount-time gating |
| `"gemini-*"` | Google Vertex AI | Via `google.genai` |

Default model in `full_pipeline.py`: `claude-opus-4-5` (`full_pipeline.py:53`).

---

## 10. File System Layout (at runtime)

```
Automated-AI-Researcher/
├── agent/                          # Source code
├── env/grpo/                       # Base GRPO training env (read-only template)
├── env/nanogpt/                    # Base NanoGPT training env (read-only template)
│
├── ideas_{run_name}/
│   ├── ideas_epoch0.json           # Raw idea strings for epoch 0
│   ├── ideas_epoch1.json           # Raw idea strings for epoch 1
│   └── database.json               # Merged, ranked, deduplicated idea database
│
├── idea_stats_{run_name}/
│   └── epoch{N}.json               # Success/fail counts for each epoch
│
├── diffs_claude/                   # Archive of all epoch diff dirs
│   └── diffs_{run_name}_epoch{N}/
│       └── code_diff_idea_{i}.diff
│
├── repo_variants_claude/           # Archive of all epoch variant dirs
│   └── repo_variants_{run_name}_epoch{N}/
│       └── idea_{i}/               # Full patched codebase copy
│
└── training_logs_{run_name}/
    └── epoch{N}/
        ├── ranked_ideas.json
        └── idea_{i}/
            ├── output.log          # Downloaded from W&B
            └── metrics.json        # Extracted eval metrics
```
