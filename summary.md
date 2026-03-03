# Paper & Codebase Summary: "Towards Execution-Grounded Automated AI Research"

---

## Paper Summary

**Core Idea:** Instead of just having LLMs generate research ideas (which often sound plausible but don't work), this paper builds a fully automated pipeline that: generates an idea → writes the code → runs a real GPU training job → measures the result → feeds that feedback back to generate better ideas.

### Key High-Level Ideas

1. **Execution Grounding**: LLMs are notorious for generating ideas that look good on paper but fail when implemented. The paper grounds ideation in actual execution results (real training runs on GPUs), creating a tight feedback loop.

2. **Two Research Environments:**
   - **GRPO (post-training)**: Fine-tune Qwen2.5-Math-1.5B on the MATH dataset using the GRPO RL algorithm. Metric: validation accuracy.
   - **nanoGPT (pre-training)**: Pre-train a 124M GPT-2 model on FineWeb. Metric: time to reach 3.28 validation loss.

3. **Automated Executor**: A 3-component system:
   - **Implementer** (CPU): LLM generates unified diff files → patches into baseline codebase → zips and uploads to cloud
   - **Scheduler**: Polls cloud, allocates GPU resources
   - **Worker**: Runs training on GPUs, logs results to Weights & Biases

4. **Evolutionary Search** (their main method, Algorithm 1):
   - Epoch 0: Sample N random ideas
   - Subsequent epochs: Mix **exploitation** (refine top ideas above baseline) + **exploration** (sample broadly from all past ideas)
   - Exploitation ratio increases over time (50% → 80%)
   - Results beat human experts on GRPO (69.4% vs. 68.8% best student), and significantly beat baseline on nanoGPT (19.7 min vs. 35.9 min)

5. **Reinforcement Learning** (alternative approach):
   - Fine-tune Qwen3-30B using execution results as reward
   - Improves *average* reward but **not** the upper bound
   - Causes mode collapse: model converges on a few easy-to-implement ideas, diversity collapses

6. **Key Findings:**
   - Claude-4.5-Opus shows a clear scaling trend (more epochs = better); Sonnet/GPT-5 saturate early
   - Models genuinely generate algorithmic ideas (not just hyper-parameter tweaks)
   - Models can rediscover recent research papers (e.g., group diversity rewards similar to recent papers)
   - 90%+ execution rates for Claude models on nanoGPT

---

## Codebase Architecture

### `agent/` — The Search Scaffold

| File | Role |
|---|---|
| `api.py` | Unified API wrapper supporting Claude (direct + AWS Bedrock), GPT (OpenAI), Gemini, DeepSeek. Handles thinking mode, retries, key loading from `keys.json` |
| `agent.py` | **Idea generation** (`agent_call_idea_simple` for epoch 0) + **Code diff generation** (`generate_code_diff`) + **Diff application** (`apply_code_diff`) with up to 10 auto-retry trials. Also has `code_diff_fixer` to auto-correct malformed diff hunk headers |
| `evolutionary_search.py` | **Evolutionary search logic**: `agent_call_idea_evolutionary_exploit` (refine winners) + `agent_call_idea_evolutionary_explore` (explore novel directions) + `update_database` (ranks all ideas by metric across epochs) |
| `full_pipeline.py` | **Orchestrator**: runs the full multi-epoch loop — generate ideas → generate diffs → zip/upload → wait 90 minutes → retrieve W&B logs → repeat |
| `compute_idea_stats.py` | Checks which ideas successfully produced a repo variant (code was patchable), computes success rate |
| `upload_repo_variants.py` | Zips patched codebases and uploads to HuggingFace Hub for the GPU cluster to download |
| `retrieve_training_logs.py` | Polls W&B for completed runs, extracts metrics, ranks ideas, saves `ranked_ideas.json` |

### `env/grpo/` — Post-training Environment

- **`grpo.py`**: Full GRPO training loop — vLLM for rollout sampling, AdamW for gradient updates, group-normalized advantage computation, supports `grpo_clip` and `reinforce_with_baseline` loss types. Logs to W&B.
- **`grpo_utils.py`**: Advantage computation (group normalization), microbatch training step
- **`sample.py`**: Rollout sampling with vLLM
- **`drgrpo_grader.py`**: Training reward function (grading math answers)
- **`evaluate.py`**: Evaluation reward (kept separate so the LLM agent cannot modify it — anti-reward-hacking)
- **`run_job.sh`**: Shell script to launch a single training run (B200 GPU); `run.sh` is the A100 variant

### `env/nanogpt/` — Pre-training Environment

- **`train.py`**: nanoGPT training loop with `forward_with_cache` / `forward_safe` functions that enforce autoregressive inference (no future-token leakage during validation)
- **`fineweb.py`**: Downloads/preprocesses FineWeb data
- **`run_job.sh`**: 8-GPU training launch script; `run.sh` is a 2-GPU debug variant

---

## The Full Pipeline Flow

```
full_pipeline.py (per epoch):
  1. agent_call_idea()             → ideas_epoch{N}.json
  2. generate_code_diff_parallel() → diffs + patched repo_variants (parallel, 10 threads)
  3. compute_idea_stats()          → tracks success/failure rate
  4. zip_and_upload_repo_variants() → HuggingFace Hub
  5. sleep(90 min)                 → wait for GPU cluster to finish
  6. retrieve_training_logs()      → poll W&B, rank ideas
  7. update_database()             → ranked cumulative database across epochs
```

The pipeline is designed to run fully autonomously — once started, it needs no human intervention until the final results are in.
