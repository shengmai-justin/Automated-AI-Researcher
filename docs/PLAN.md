# Plan: Detailed Workflow + Robust Implementer Pipeline

## Context

The paper's Figure 1 shows a high-level 3-box diagram (Implementer → Scheduler → Worker). The actual codebase is far richer. This plan produces:

1. **A detailed workflow document** reverse-engineered from the code — exact data contracts, timing, retry logic, and state transitions that the paper figure omits.
2. **A proposed robust implementer pipeline** — a redesigned `implementer` subsystem addressing the critical robustness gaps found in the current code.

Both outputs will be written as Markdown documents in the repository.

---

## Deliverable 1: Detailed Evolutionary Search Workflow (`docs/workflow_detailed.md`)

### What to document (reverse-engineered from code)

#### A. Full Epoch State Machine

```
Epoch N
├─ 1. IDEA GENERATION  (agent.py + evolutionary_search.py)
│   ├─ Epoch 0: agent_call_idea_simple() — random batch of 10 ideas × (num_ideas/10) batches
│   └─ Epoch N≥1: agent_call_idea_evolutionary()
│       ├─ update_database(epoch N-1)  ← reads ranked_ideas.json from last epoch
│       ├─ Compute split: exploit_ratio = min(0.5 + 0.1*(N//2), 0.8), rounded to nearest 10
│       ├─ EXPLOIT batches: agent_call_idea_evolutionary_exploit()
│       │   ├─ Filter database.json: GRPO→accuracy>0.49, NanoGPT→0<loss<3.255
│       │   ├─ Random sample top_k (default 100) from winners
│       │   └─ Prompt: "combine and refine these winning ideas"
│       └─ EXPLORE batches: agent_call_idea_evolutionary_explore()
│           ├─ Random sample sample_k (default 100) from ALL past ideas
│           └─ Prompt: "generate entirely new ideas, avoid patterns that failed"
│   → Output: ideas_{run_name}/ideas_epoch{N}.json  (array of [Experiment]...[Code Changes]... strings)
│
├─ 2. CODE DIFF GENERATION  (agent.py, ThreadPoolExecutor 10 workers)
│   └─ Per idea (parallel, max 10 retries each):
│       ├─ generate_code_diff() → raw LLM diff text
│       ├─ Strip markdown ``` fences
│       ├─ Rewrite --- / +++ headers to absolute variant paths
│       ├─ code_diff_fixer() → fix @@ hunk line count mismatches
│       ├─ apply_code_diff() → copy env_dir, run `patch -p0 < diff_file`
│       ├─ ON FAILURE: log error_msg, pass to next generate_code_diff() as prev_diff_error
│       └─ After max 10 trials: silently skip (no output directory)
│   → Output: diffs_{run_name}_epoch{N}/code_diff_idea_{i}.diff
│              repo_variants_{run_name}_epoch{N}/idea_{i}/  (patched codebase)
│
├─ 3. STATS + UPLOAD  (compute_idea_stats.py, upload_repo_variants.py)
│   ├─ compute_idea_stats() → count which idea_{i} dirs exist
│   │   → idea_stats_{run_name}/epoch{N}.json  {success_count, failed_ideas, ...}
│   ├─ zip_and_upload_repo_variants() → zip each idea dir, upload to HuggingFace
│   └─ move_diffs_and_repo_variants() → archive to diffs_claude/, repo_variants_claude/
│
├─ 4. FIXED WAIT: sleep(90 minutes)  ← External GPU scheduler picks up zips from HuggingFace
│   [External: Scheduler polls HuggingFace, allocates GPUs, Workers run training, log to W&B]
│   W&B run naming: {run_name}_epoch{N}_b200_idea_{i}
│
└─ 5. LOG RETRIEVAL LOOP  (retrieve_training_logs.py)
    ├─ target = success_count from idea_stats (ideas that got a patchable diff)
    ├─ WHILE retrieved ≤ 30% of target:
    │   ├─ api.runs() filtered by fnmatch pattern → sorted by created_at
    │   ├─ For each matched run:
    │   │   ├─ Download output.log
    │   │   ├─ GRPO: run.history()["eval/mean_reward"] → max value = final_reward
    │   │   └─ NanoGPT: parse output.log for "val_loss:" lines → min value = final_reward
    │   ├─ Write training_logs_{run_name}/epoch{N}/idea_{i}/metrics.json
    │   ├─ Write ranked_ideas.json: [{idea_i: reward_value}, ...]
    │   └─ IF retrieved ≤ 30%: sleep(20 minutes), repeat
    └─ update_database() → merge into ideas_{run_name}/database.json
```

#### B. JSON Data Contracts (exact schemas)

| File | Schema |
|------|--------|
| `ideas_epoch{N}.json` | `["[Experiment]...[Code Changes]...", ...]` |
| `ranked_ideas.json` | `[{"idea_0": 0.65}, {"idea_1": -999.0}, ...]` (sorted: GRPO desc, NanoGPT asc) |
| `database.json` | `[{"epoch":N,"idea_id":i,"idea":"...","best_eval_accuracy":0.65}, ...]` |
| `metrics.json` (GRPO) | `{"eval_rewards":[{"step":N,"eval_reward":0.5}], "train_rewards":[...]}` |
| `metrics.json` (NanoGPT) | `{"eval_rewards":[{"step":N,"val_loss":3.2,"train_time_ms":N}], "time_to_target":-999}` |
| `idea_stats.json` | `{"successful_ideas":[0,1,3],"failed_ideas":[2],"success_count":3,"total_ideas":4,"success_percent":75.0}` |

#### C. Hardcoded Constants to Surface

| Constant | Value | Purpose | File:Line |
|---|---|---|---|
| GRPO baseline threshold | 0.49 | Filter exploit pool | evolutionary_search.py:218 |
| NanoGPT baseline threshold | 3.255 | Filter exploit pool | evolutionary_search.py:222 |
| NanoGPT reward-hacking filter | 2.5 | Skip suspiciously low losses | evolutionary_search.py:114 |
| Max exploit ratio | 0.8 | Cap on exploitation | evolutionary_search.py:532 |
| Exploit ratio schedule | 0.5 + 0.1*(epoch//2) | Anneal explore→exploit | evolutionary_search.py:532 |
| Thinking budget (ideas) | 1500 tokens | Claude extended thinking | evolutionary_search.py:344 |
| Thinking budget (code diff) | 3000 tokens | Claude extended thinking | agent.py:348 |
| Max code diff retries | 10 | Per-idea retry limit | agent.py:511 |
| Post-upload wait | 90 min | Wait for GPU jobs | full_pipeline.py:82 |
| W&B poll interval | 20 min | Log retrieval polling | full_pipeline.py:105 |
| W&B completion threshold | 30% | Stop waiting threshold | full_pipeline.py:96 |
| Machine suffix | "b200" | W&B run name pattern | retrieve_training_logs.py:61 |

#### D. Prompt Architecture Summary

For each function, document: context injected (full codebase with line numbers), parent ideas format, constraints block, output format tags `[Experiment]...[Code Changes]...[End]`, and deduplication via cache penalty string.

---

## Deliverable 2: Robust Implementer Pipeline (`docs/robust_implementer.md`)

### What to redesign (addressing all critical gaps)

#### A. Critical Bug Fixes (must-do)

1. **`apply_code_diff` wrong cwd** (agent.py:398)
   - Current: `run(f"patch -p0 < {diff_file}", cwd=".")` — patches from project root
   - Fix: Run patch from within `new_repo_dir` with `-p1` for git-style paths OR use absolute diff paths correctly

2. **`feedback_loop` parameter name mismatch** (agent.py:491)
   - Current: calls `generate_code_diff(..., previous_diff_file=...)` but function signature has `prev_diff_file`
   - Fix: Align parameter names

3. **`apiqa` returns None** (api.py:285)
   - Current: silently returns `None` after all retries fail
   - Fix: Raise `RuntimeError("API call failed after N retries: last_error")`

4. **`_claude_qa` unchecked content array** (api.py:127-128)
   - Current: `response.content[0].thinking` and `response.content[1].text` without bounds check
   - Fix: Validate content array structure before indexing

5. **`ThreadPoolExecutor.map` swallows exceptions** (agent.py:548-549)
   - Current: one thread exception stops all workers
   - Fix: Wrap worker in try/except, collect results with `Future` objects

#### B. Proposed Robust Implementer Architecture

```
Input: batch of (idea_idx, idea_text) pairs
           │
           ▼
┌─────────────────────────────────────────────┐
│  1. DIFF GENERATION (parallel, per idea)    │
│                                             │
│  For trial in range(max_trials=10):         │
│    ├─ Build prompt (idea + codebase + prev  │
│    │  diff + structured error analysis)     │
│    ├─ Call LLM API (with retry/backoff)     │
│    ├─ VALIDATE response is parseable diff   │
│    │   (check ---/+++ headers present,      │
│    │    check @@ headers present)           │
│    ├─ code_diff_fixer() with logging        │
│    ├─ DRY RUN: patch --dry-run (no disk I/O)│
│    │   ├─ SUCCESS → write diff, apply patch │
│    │   └─ FAIL → parse_patch_error(),       │
│    │            build structured error msg  │
│    └─ Log outcome to per-idea audit log     │
│                                             │
│  Output:                                    │
│  ├─ SUCCESS: repo_variants/idea_N/ + diff   │
│  └─ FAIL: error_log/idea_N/audit.json       │
└─────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  2. STRUCTURED ERROR ANALYSIS               │
│                                             │
│  parse_patch_error(stderr, diff_content):   │
│  ├─ Extract: line number, hunk index, file  │
│  ├─ Classify: hunk_mismatch / no_such_file  │
│  │            / context_mismatch / etc.     │
│  ├─ Extract: 5 lines around failure point   │
│  └─ Return structured dict (not raw stderr) │
│                                             │
│  This replaces raw error string in prompt   │
└─────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  3. OBSERVABILITY                           │
│                                             │
│  Per-idea audit log (error_log/idea_N/):    │
│  ├─ audit.json: {idea_idx, trials: [        │
│  │    {trial, outcome, error_type,          │
│  │     error_detail, timestamp, tokens}]}   │
│  └─ batch_summary.json: {total, success,    │
│       fail_reasons: {hunk_mismatch: N, ...}}│
│                                             │
│  Session metrics:                           │
│  ├─ Diff generation success rate            │
│  ├─ Most common failure reason              │
│  ├─ API retry rate per model                │
│  └─ Avg trials to success                  │
└─────────────────────────────────────────────┘
```

#### C. Diff Validation Function (new utility)

```python
def validate_diff(diff_text: str) -> tuple[bool, str]:
    """
    Returns (is_valid, error_message).
    Checks:
    1. Has at least one --- header
    2. Has at least one +++ header
    3. Has at least one @@ hunk header
    4. File pairs match (--- and +++ for same file)
    5. Hunk bodies are not empty
    """
```

#### D. Structured Error Parser (new utility)

```python
def parse_patch_error(stderr: str, diff_lines: list[str]) -> dict:
    """
    Returns structured error context for LLM prompt:
    {
      "error_type": "hunk_mismatch" | "no_such_file" | "context_mismatch" | "unknown",
      "failed_file": "grpo.py",
      "failed_hunk_idx": 2,
      "failed_at_line": 45,
      "diff_context": "..5 lines around failure...",
      "raw_error": "..."
    }
    """
```

#### E. API Robustness Improvements

- `apiqa`: Raise on all-retry exhaustion instead of returning `None`
- Exponential backoff cap: `min(2**tries, 60)` seconds
- Model-specific exception handling (RateLimitError vs APIError vs connection error)
- `_claude_qa`: Validate `response.content` structure before indexing

#### F. What NOT to Change

- The LLM prompt content itself (idea prompts, code diff prompts) — these are tuned
- The evolutionary search ratio schedule — paper-verified
- The W&B integration or data schemas — changes would break log retrieval
- The `env/grpo` and `env/nanogpt` training code

---

## Files to Create

| Output File | Content |
|---|---|
| `docs/PLAN.md` | **Step 0**: Copy of this plan saved into the repo for reference |
| `docs/workflow_detailed.md` | Full epoch state machine, JSON schemas, constants table, prompt architecture |
| `docs/robust_implementer.md` | Current bugs found, proposed architecture diagram, new utilities spec |

## Files to Modify (bug fixes only, minimal diffs)

| File | Change |
|---|---|
| `agent/agent.py` | Fix `apply_code_diff` cwd, `feedback_loop` param names, `ThreadPoolExecutor` exception handling |
| `agent/api.py` | Fix `apiqa` None return, `_claude_qa` content bounds check, backoff cap |

## Verification

1. **Workflow doc**: Cross-check every claim against actual line numbers in code; all constants must be sourced.
2. **Bug fixes**: Run `python -m agent.full_pipeline --epochs 1 --num_ideas_per_epoch 2` (requires valid `keys.json`) to confirm no crashes in implementer path.
3. **New utilities**: Unit test `validate_diff()` with known-good and known-bad diffs; test `parse_patch_error()` with captured `patch` stderr strings.
