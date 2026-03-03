# Automated AI Researcher — Workflow Diagram

> Generated from `docs/workflow_detailed.md`

```mermaid
flowchart TD
  A([Start: full_pipeline.py]) --> B{for each epoch}

  subgraph S1 ["STEP 1 — Idea Generation"]
    B --> C{epoch == 0?}
    C -- Yes --> D["agent_call_idea_simple\nbatches × num_ideas/10\nbudget=3000, temp=1.0\nFull codebase context"]
    C -- No  --> E["update_database()\nranked_ideas → database.json\n(deduped)"]
    E --> F["Compute exploit/explore ratio\nmin(0.5 + 0.1×epoch//2, 0.8)"]
    F --> G["EXPLOIT batches\ntop winners, budget=1500"]
    F --> H["EXPLORE batches\nrandom sample, budget=1500"]
    G --> I["ideas_epochN.json"]
    H --> I
    D --> I
  end

  subgraph S2 ["STEP 2 — Code Diff Generation  (ThreadPoolExecutor × 10)"]
    I --> J{trial < 10?}
    J -- Yes --> K["generate_code_diff()\n+ prev_diff + error if retry\nbudget=3000"]
    K --> L["code_diff_fixer()\nrecompute @@ hunks"]
    L --> M["apply_code_diff()\ncopytree + patch -p0"]
    M --> N{patch OK?}
    N -- Yes --> O["repo_variants/idea_i/\ndiffs/idea_i.diff"]
    N -- No  --> J
    J -- No  --> P[skip idea silently]
  end

  subgraph S3 ["STEP 3 — Stats + Upload"]
    O --> Q["compute_idea_stats()\n→ idea_stats/epochN.json"]
    O --> R["zip_and_upload_repo_variants()\n→ HuggingFace"]
    O --> S["move diffs/variants\n→ *_claude/ archive"]
  end

  subgraph S4 ["STEP 4 — Wait 90 min  ── external boundary ──"]
    Q & R & S --> T["HuggingFace Scheduler\nB200 GPUs → training\n→ W&B logs"]
  end

  subgraph S5 ["STEP 5 — Log Retrieval Loop"]
    T --> U{retrieved ≥ 30%\nof submitted?}
    U -- No  --> V["retrieve_training_logs()\nW&B API poll\nGRPO: max eval_reward\nNanoGPT: min val_loss"]
    V --> W["Write metrics.json\n+ ranked_ideas.json"]
    W --> X[sleep 20 min]
    X --> U
  end

  U -- Yes --> Y{more epochs?}
  Y -- Yes --> B
  Y -- No  --> Z["update_database final\n→ database.json"]
```

## Key Design Points

| Aspect | Detail |
|---|---|
| **Parallelism** | 10 threads generate code diffs simultaneously |
| **Self-correction** | Up to 10 retry trials per diff, feeding error back to LLM |
| **Exploit/Explore** | Ratio starts 50/50 at epoch 1, ramps to 80/20 by epoch 8 |
| **Polling guard** | Only stops waiting once ≥30% of submitted ideas return W&B logs |
| **External boundary** | GPU scheduling (HuggingFace + B200 workers) is entirely outside the repo |
