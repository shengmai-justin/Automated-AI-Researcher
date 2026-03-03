# Detailed Evolutionary Search Workflow

> Reverse-engineered from the codebase. Every claim is sourced to a specific file and line number.

---

## 1. High-Level Overview

The system runs a loop of `--epochs` iterations. Each epoch:

1. An **Implementer** (LLM agent) proposes research ideas and generates code patches.
2. A **Scheduler** (external, HuggingFace-based) distributes patched codebases to GPU workers.
3. **Workers** run training jobs and log metrics to Weights & Biases (W&B).
4. The Implementer retrieves logs, ranks ideas, and updates an evolving database for the next epoch.

Entry point: `agent/full_pipeline.py` (command-line via `argparse`).

---

## 2. Full Epoch State Machine

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
│     │   └─ subprocess: patch -p0 < diff_file  (cwd=".")  ← BUG: see robust_implementer.md
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

## 3. JSON Data Contracts

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

## 4. Hardcoded Constants

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

## 5. Prompt Architecture

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

## 6. Key State Transitions

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

## 7. Multi-Model Support

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

## 8. File System Layout (at runtime)

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
