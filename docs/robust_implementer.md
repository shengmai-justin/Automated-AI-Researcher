# Robust Implementer Pipeline

> Documents critical bugs found in the current implementer subsystem and proposes a redesigned architecture with structured error handling, observability, and diff validation.

---

## 1. Critical Bugs in Current Code

### Bug 1: `apply_code_diff` Wrong `cwd` (agent.py:398)

**Symptom**: Patch may fail or silently corrupt the wrong files when the script is invoked from a directory other than the project root.

**Root cause**:

```python
# agent.py:381-398
def run(cmd, cwd=new_repo_dir, check=True):   # <-- default is new_repo_dir
    result = subprocess.run(
        cmd,
        cwd=cwd,                              # <-- uses whatever is passed
        ...
    )

# But the actual call hardcodes cwd to "." (process CWD):
run(f"patch -p0 < {str(diff_file)}", cwd=".")  # agent.py:398  ← BUG
```

The diff headers were rewritten to relative paths like `repo_variants_{run}_epochN/idea_I/file.py`
(see `agent.py:362-365`). Running `patch -p0` from `cwd="."` means the patch tool looks for
those paths relative to the Python process's current working directory. This only works if the
process was started from the exact project root. Any other invocation directory silently applies
the patch to the wrong location or fails.

**Fix applied** (see agent.py): Replace `cwd="."` with the absolute path of the project root,
derived from the location of `agent.py` itself, making it invocation-directory-independent.

---

### Bug 2: `feedback_loop` Parameter Name Mismatch (agent.py:491)

**Symptom**: On retry (trial > 1), `feedback_loop` calls `generate_code_diff` with wrong keyword
arguments. The previous diff and its error are silently ignored, so every retry starts from scratch
instead of learning from the previous failure.

**Root cause**:

```python
# generate_code_diff signature (agent.py:288):
def generate_code_diff(..., prev_diff_file=None, prev_diff_error=None, ...):

# But feedback_loop calls it with (agent.py:491):
response, thinking = generate_code_diff(
    ...,
    previous_diff_file=diff_file,     # ← wrong name (no "ious")
    previous_diff_error=diff_error_file  # ← wrong name
)
# Python silently ignores these as unexpected kwargs only if the function
# accepts **kwargs — but it does not. This raises TypeError at runtime.
```

**Fix applied** (see agent.py): Change `previous_diff_file` → `prev_diff_file` and
`previous_diff_error` → `prev_diff_error` to match the function signature.

---

### Bug 3: `apiqa` Returns `None` Silently (api.py:285)

**Symptom**: If all `max_trial` API call attempts fail, `apiqa` returns `None`. Callers then
crash with an `AttributeError` or `TypeError` on the `None` value (e.g., `len(None)` in
`agent.py:349`), producing an opaque error far from the actual failure point.

**Root cause**:

```python
# api.py:260-285
def apiqa(..., max_trial: int = 1):
    completion = None
    tries = 0
    while completion is None and tries < max_trial:
        try:
            ...
            completion = _claude_qa(...)
        except Exception as e:
            print(f"Trial {tries} with Exception: {str(e)}, sleeping for {2**tries} seconds")
            time.sleep(2**tries)
            tries += 1
    return completion   # ← returns None if all retries failed
```

**Fix applied** (see api.py): After the loop, if `completion is None`, raise a `RuntimeError`
with the last exception message, so failures surface at the correct call site.

---

### Bug 4: `_claude_qa` Unchecked Content Array (api.py:127-128)

**Symptom**: If the Anthropic API returns a response with unexpected `content` structure (e.g.,
only one block instead of two, or no `thinking` block), the code crashes with an `IndexError`.

**Root cause**:

```python
# api.py:126-128
if thinking_mode:
    thinking = response.content[0].thinking   # ← assumes index 0 is thinking block
    response = response.content[1].text       # ← assumes index 1 is text block
    # No bounds check; no type check on block types
```

The Anthropic API can return content blocks in different orders or omit blocks if the model
decides not to think (e.g., very short prompts). The same pattern exists for the `else` branch
at `api.py:141`.

**Fix applied** (see api.py): Validate that `response.content` has sufficient length and iterate
over blocks by type rather than assuming fixed positions.

---

### Bug 5: `ThreadPoolExecutor.map` Exception Propagation (agent.py:548-549)

**Symptom**: An exception in any worker thread propagates when the `executor.map()` iterator is
consumed. Since `executor.map()` re-raises the first exception seen while iterating, one failing
idea can prevent all subsequent results from being collected, even if they succeeded.

**Root cause**:

```python
# agent.py:548-549
with ThreadPoolExecutor(max_workers=total_workers) as executor:
    executor.map(map_func, range(total_ideas))
# If map_func(3) raises, executor.map re-raises it here,
# abandoning ideas 4..N that may have finished successfully.
```

Note: `_generate_code_diff_parallel_helper` already has a broad `except Exception as e` that
catches all errors (`agent.py:528`). So the only way this triggers is if the helper itself has
an unexpected error outside its try/except (e.g., an error in the partial setup). Still, the
pattern is fragile.

**Fix applied** (see agent.py): Consume the iterator inside a try/except to log any unexpected
thread-level exceptions without aborting the entire batch.

---

### Bug 6: Backoff Cap Missing (api.py:283)

**Symptom**: With `tries=0,1,2,...`, the sleep is `2**0=1s`, `2**1=2s`, ..., `2**9=512s` (8.5
min). For `max_trial=10`, the 10th retry sleeps over 8 minutes. There is no cap on the backoff,
so long-running API failures waste wall-clock time unnecessarily.

**Root cause**:

```python
# api.py:283
time.sleep(2**tries)  # unbounded exponential backoff
```

**Fix applied** (see api.py): Cap the sleep at 60 seconds: `time.sleep(min(2**tries, 60))`.

---

## 2. Applied Fixes Summary

| # | File | Line | Issue | Fix |
|---|---|---|---|---|
| 1 | `agent/agent.py` | 398 | `patch` runs from relative `cwd="."` | Use absolute project-root path |
| 2 | `agent/agent.py` | 491 | `previous_diff_file` / `previous_diff_error` wrong param names | Rename to `prev_diff_file` / `prev_diff_error` |
| 3 | `agent/api.py` | 285 | `apiqa` returns `None` silently on all-retry failure | Raise `RuntimeError` |
| 4 | `agent/api.py` | 127-128 | `response.content[0]` / `[1]` without bounds check | Validate structure before indexing |
| 5 | `agent/agent.py` | 548-549 | `executor.map` propagates first thread exception | Wrap in try/except |
| 6 | `agent/api.py` | 283 | Unbounded exponential backoff | Cap at 60 seconds |

---

## 3. Proposed Robust Implementer Architecture

The following is the target architecture for a hardened implementer subsystem. It is designed to
be a drop-in replacement for `generate_code_diff_parallel` + `apply_code_diff`.

```
Input: batch of (idea_idx, idea_text) pairs
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│  1. DIFF GENERATION  (parallel, per idea, max 10 trials)    │
│                                                             │
│  For trial in range(max_trials=10):                         │
│    ├─ Build prompt:                                         │
│    │   • Full codebase (context_prompt)                     │
│    │   • Idea text                                          │
│    │   • Previous diff (if trial > 1)                       │
│    │   • Structured error analysis (if trial > 1)           │
│    ├─ Call LLM API (with retry + capped backoff)            │
│    ├─ Strip markdown fences                                 │
│    ├─ validate_diff(response) → (is_valid, error_msg)       │
│    │   IF NOT valid → log, continue to next trial           │
│    ├─ code_diff_fixer()  (fix @@ line count mismatches)     │
│    ├─ DRY RUN: patch --dry-run --check (no disk I/O)        │
│    │   ├─ SUCCESS → write diff, apply patch (real run)      │
│    │   │   → break (record success in audit log)            │
│    │   └─ FAIL → parse_patch_error(stderr, diff_lines)      │
│    │       → structured error dict for next prompt          │
│    └─ Log trial outcome to per-idea audit log               │
│                                                             │
│  Output:                                                    │
│  ├─ SUCCESS: repo_variants/idea_N/ + diff file              │
│  └─ FAIL (all 10 trials): error_log/idea_N/audit.json       │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│  2. STRUCTURED ERROR ANALYSIS  (new utility)                │
│                                                             │
│  parse_patch_error(stderr, diff_lines) → dict               │
│  ├─ Extract: failed file name, hunk index, line number      │
│  ├─ Classify error type:                                    │
│  │   • "hunk_mismatch"    — @@ counts don't match body      │
│  │   • "no_such_file"     — target file doesn't exist       │
│  │   • "context_mismatch" — context lines don't match       │
│  │   • "unknown"          — other patch error               │
│  ├─ Extract 5 lines of diff context around the failure      │
│  └─ Return structured dict (replaces raw stderr string)     │
│                                                             │
│  This structured dict is injected into the next LLM prompt  │
│  instead of the raw patch stderr, giving the model better   │
│  signal about what specifically failed.                     │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│  3. OBSERVABILITY  (new per-idea + batch audit logs)        │
│                                                             │
│  Per-idea audit: error_log/idea_N/audit.json                │
│  {                                                          │
│    "idea_idx": N,                                           │
│    "final_outcome": "success" | "failure",                  │
│    "trials": [                                              │
│      {                                                      │
│        "trial": 1,                                          │
│        "outcome": "patch_error",                            │
│        "error_type": "hunk_mismatch",                       │
│        "error_detail": "hunk 2 in grpo.py: expected 5...",  │
│        "timestamp": "2026-03-02T17:00:00Z",                 │
│        "tokens_used": 4200                                  │
│      },                                                     │
│      ...                                                    │
│    ]                                                        │
│  }                                                          │
│                                                             │
│  Batch summary: error_log/batch_summary.json                │
│  {                                                          │
│    "total": 80,                                             │
│    "success": 61,                                           │
│    "failure": 19,                                           │
│    "fail_reasons": {                                        │
│      "hunk_mismatch": 11,                                   │
│      "context_mismatch": 5,                                 │
│      "no_such_file": 2,                                     │
│      "api_error": 1                                         │
│    },                                                       │
│    "avg_trials_to_success": 1.8                             │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. New Utility: `validate_diff`

```python
def validate_diff(diff_text: str) -> tuple[bool, str]:
    """
    Pre-flight check before running patch. Returns (is_valid, error_message).

    Checks performed:
      1. Has at least one '--- ' header
      2. Has at least one '+++ ' header
      3. Has at least one '@@ ' hunk header
      4. Every '--- ' has a matching '+++ ' for the same file
      5. No hunk body is completely empty (all context lines with no +/-)

    Use before code_diff_fixer() to catch malformed LLM output early
    and avoid wasting a patch invocation.

    Example usage:
        is_valid, err = validate_diff(response)
        if not is_valid:
            error_msg = f"Invalid diff: {err}"
            continue  # next trial
    """
    lines = diff_text.splitlines()
    minus_files = []
    plus_files = []
    has_hunk = False

    for line in lines:
        if line.startswith("--- "):
            minus_files.append(line[4:].split("\t")[0])
        elif line.startswith("+++ "):
            plus_files.append(line[4:].split("\t")[0])
        elif line.startswith("@@ "):
            has_hunk = True

    if not minus_files:
        return False, "no '--- ' file header found"
    if not plus_files:
        return False, "no '+++ ' file header found"
    if not has_hunk:
        return False, "no '@@ ' hunk header found"
    if len(minus_files) != len(plus_files):
        return False, f"mismatched file header count: {len(minus_files)} '---' vs {len(plus_files)} '+++'"

    return True, ""
```

---

## 5. New Utility: `parse_patch_error`

```python
def parse_patch_error(stderr: str, diff_lines: list[str]) -> dict:
    """
    Convert raw patch stderr into a structured dict for LLM prompt injection.

    Returns:
    {
      "error_type": "hunk_mismatch" | "no_such_file" | "context_mismatch" | "unknown",
      "failed_file": str | None,
      "failed_hunk_idx": int | None,   # 0-indexed
      "failed_at_line": int | None,    # line in original file
      "diff_context": str,             # 5 lines around failure in diff
      "raw_error": str                 # original stderr for reference
    }

    Error classification heuristics:
      - "can't find file"  / "No such file"   → "no_such_file"
      - "Hunk #N FAILED"  / "offset"          → "hunk_mismatch"
      - "Reversed"  / "already applied"        → "context_mismatch"
      - anything else                          → "unknown"

    Example usage:
        try:
            subprocess.run(["patch", "--dry-run", "-p0", ...], check=True, ...)
        except subprocess.CalledProcessError as e:
            error_ctx = parse_patch_error(e.stderr, diff_lines)
            prompt_error = (
                f"Patch failed with error type: {error_ctx['error_type']}\\n"
                f"File: {error_ctx['failed_file']}\\n"
                f"Near diff lines:\\n{error_ctx['diff_context']}\\n"
                f"Raw error: {error_ctx['raw_error']}"
            )
    """
    import re

    result = {
        "error_type": "unknown",
        "failed_file": None,
        "failed_hunk_idx": None,
        "failed_at_line": None,
        "diff_context": "",
        "raw_error": stderr,
    }

    # Classify
    if "can't find file" in stderr.lower() or "no such file" in stderr.lower():
        result["error_type"] = "no_such_file"
    elif "hunk" in stderr.lower() and "failed" in stderr.lower():
        result["error_type"] = "hunk_mismatch"
    elif "reversed" in stderr.lower() or "already applied" in stderr.lower():
        result["error_type"] = "context_mismatch"

    # Extract failed file name
    m = re.search(r"patching file (.+)", stderr)
    if m:
        result["failed_file"] = m.group(1).strip()

    # Extract hunk index and line number
    m = re.search(r"Hunk #(\d+) FAILED at (\d+)", stderr)
    if m:
        result["failed_hunk_idx"] = int(m.group(1)) - 1
        result["failed_at_line"] = int(m.group(2))

        # Extract 5 diff lines around the failed hunk
        if diff_lines:
            hunk_count = 0
            for i, line in enumerate(diff_lines):
                if line.startswith("@@ "):
                    hunk_count += 1
                    if hunk_count == result["failed_hunk_idx"] + 1:
                        start = max(0, i - 2)
                        end = min(len(diff_lines), i + 8)
                        result["diff_context"] = "".join(diff_lines[start:end])
                        break

    return result
```

---

## 6. API Robustness Improvements

### `apiqa` — Raise Instead of Return None

```python
# BEFORE (api.py:260-285):
def apiqa(..., max_trial: int = 1):
    completion = None
    tries = 0
    while completion is None and tries < max_trial:
        try:
            ...
        except Exception as e:
            time.sleep(2**tries)  # unbounded backoff
            tries += 1
    return completion  # ← silently returns None on failure

# AFTER:
def apiqa(..., max_trial: int = 1):
    completion = None
    last_error = None
    tries = 0
    while completion is None and tries < max_trial:
        try:
            ...
        except Exception as e:
            last_error = e
            sleep_time = min(2**tries, 60)  # ← capped backoff
            print(f"Trial {tries} with Exception: {str(e)}, sleeping for {sleep_time}s")
            time.sleep(sleep_time)
            tries += 1
    if completion is None:
        raise RuntimeError(
            f"API call failed after {max_trial} retries. Last error: {last_error}"
        )
    return completion
```

### `_claude_qa` — Validate Content Structure

```python
# BEFORE (api.py:127-128):
thinking = response.content[0].thinking   # IndexError if < 2 blocks
response = response.content[1].text       # AttributeError if wrong type

# AFTER:
content = response.content
thinking = None
text = None
for block in content:
    if hasattr(block, "thinking") and block.type == "thinking":
        thinking = block.thinking
    elif hasattr(block, "text") and block.type == "text":
        text = block.text
if text is None:
    raise RuntimeError(
        f"_claude_qa: no text block in response content. "
        f"Blocks: {[getattr(b, 'type', type(b).__name__) for b in content]}"
    )
```

---

## 7. What NOT to Change

The following are intentionally left unmodified:

| Component | Reason |
|---|---|
| LLM prompt text (all prompts) | Tuned for quality; changes require re-evaluation |
| Evolutionary ratio schedule `0.5 + 0.1*(epoch//2)` | Paper-verified, ablated |
| W&B run name pattern `{run_name}_epoch{N}_b200_idea_{i}` | Scheduler depends on exact format |
| JSON schema of `database.json`, `metrics.json`, `ranked_ideas.json` | Log retrieval and database update depend on exact field names |
| `env/grpo/` and `env/nanogpt/` training code | Baseline environment; modifying breaks fair comparison |
| `evaluate.py` (in env dirs) | Explicitly excluded from context and modification |

---

## 8. Testing the Bug Fixes

### Unit tests for `validate_diff`

```python
# Known-good diff
good_diff = """--- env/grpo/grpo.py
+++ env/grpo/grpo.py
@@ -10,6 +10,7 @@
 def foo():
-    x = 1
+    x = 2
     return x
"""
assert validate_diff(good_diff) == (True, "")

# Missing +++ header
bad_diff = """--- env/grpo/grpo.py
@@ -10,3 +10,3 @@
 def foo():
"""
ok, err = validate_diff(bad_diff)
assert not ok
assert "+++" in err

# No hunk
bad_diff2 = """--- env/grpo/grpo.py
+++ env/grpo/grpo.py
"""
ok, err = validate_diff(bad_diff2)
assert not ok
assert "@@" in err
```

### Unit tests for `parse_patch_error`

```python
stderr_hunk = "patching file env/grpo/grpo.py\nHunk #2 FAILED at 45.\n1 out of 1 hunk FAILED"
result = parse_patch_error(stderr_hunk, [])
assert result["error_type"] == "hunk_mismatch"
assert result["failed_file"] == "env/grpo/grpo.py"
assert result["failed_hunk_idx"] == 1
assert result["failed_at_line"] == 45

stderr_missing = "can't find file to patch at input line 3"
result = parse_patch_error(stderr_missing, [])
assert result["error_type"] == "no_such_file"
```

### Integration smoke test

```bash
# Requires valid keys.json
python -m agent.full_pipeline \
  --epochs 1 \
  --num_ideas_per_epoch 2 \
  --run_name smoke_test \
  --env_dir env/nanogpt
```

Expected: no `AttributeError`, no `TypeError` on `None`, no `IndexError` from content array.
