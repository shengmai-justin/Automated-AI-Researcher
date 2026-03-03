# Automated AI Researcher — Workflow Diagram

> Generated from `docs/workflow_detailed.md`

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     AUTOMATED AI RESEARCHER — EPOCH LOOP                        ║
║                        entry: agent/full_pipeline.py                            ║
╚══════════════════════════════════════════════════════════════════════════════════╝

                    ┌─────────────────────────────────┐
                    │   for epoch in range(0, epochs)  │◄──────────────────────┐
                    └──────────────────┬──────────────┘                        │
                                       │                                        │
                    ╔══════════════════▼══════════════════╗                     │
                    ║       STEP 1: IDEA GENERATION        ║                     │
                    ╚══════════════════╤══════════════════╝                     │
                                       │                                        │
               ┌───────────────────────┴────────────────────────┐               │
               │                                                 │               │
        epoch == 0                                        epoch >= 1             │
               │                                                 │               │
               ▼                                                 ▼               │
  ┌─────────────────────────┐                    ┌──────────────────────────┐   │
  │  agent_call_idea_simple │                    │  update_database()       │   │
  │  (num_ideas/10 batches) │                    │  ranked_ideas.json →     │   │
  │  thinking_budget=3000   │                    │  database.json (deduped) │   │
  │  temperature=1.0        │                    └────────────┬─────────────┘   │
  │  Full codebase context  │                                 │                 │
  └───────────┬─────────────┘                    ┌───────────▼──────────────┐   │
              │                                   │  Compute exploit/explore │   │
              │                                   │  ratio:                  │   │
              │                                   │  max = min(0.5+0.1*      │   │
              │                                   │       (epoch//2), 0.8)   │   │
              │                                   └─────────┬──────┬─────────┘   │
              │                                             │      │             │
              │                                    EXPLOIT  │      │  EXPLORE    │
              │                              (top winners)  │      │  (all past) │
              │                                             ▼      ▼             │
              │                          ┌──────────────────┐  ┌──────────────┐ │
              │                          │  exploit_call()  │  │ explore_call │ │
              │                          │  Filter: acc>0.49│  │ Sample k=100 │ │
              │                          │  Sample top_k=100│  │ "new ideas,  │ │
              │                          │  "combine/refine │  │ avoid failed │ │
              │                          │  winners"        │  │ patterns"    │ │
              │                          │  budget=1500     │  │ budget=1500  │ │
              │                          └────────┬─────────┘  └──────┬───────┘ │
              │                                   └────────┬──────────┘         │
              └───────────────────────────────────────────┘                    │
                                       │                                        │
                                       ▼                                        │
                         ideas_{run}/ideas_epoch{N}.json                        │
                                       │                                        │
                    ╔══════════════════▼══════════════════╗                     │
                    ║     STEP 2: CODE DIFF GENERATION     ║                     │
                    ╚══════════════════╤══════════════════╝                     │
                                       │                                        │
                       ThreadPoolExecutor(max_workers=10)                       │
                                       │                                        │
                     ┌─────────────────▼──────────────────┐                    │
                     │  Per idea (parallel):               │                    │
                     │                                     │                    │
                     │  trial = 0                          │                    │
                     │  WHILE trial < 10:                  │                    │
                     │    trial += 1                       │                    │
                     │    ┌──────────────────────────┐     │                    │
                     │    │  generate_code_diff()    │     │                    │
                     │    │  (+ prev_diff + error    │     │                    │
                     │    │   if retry)              │     │                    │
                     │    │  thinking_budget=3000    │     │                    │
                     │    └────────────┬─────────────┘     │                    │
                     │                 │                    │                    │
                     │    ┌────────────▼─────────────┐     │                    │
                     │    │  code_diff_fixer()       │     │                    │
                     │    │  (recompute @@ hunks)    │     │                    │
                     │    └────────────┬─────────────┘     │                    │
                     │                 │                    │                    │
                     │    ┌────────────▼─────────────┐     │                    │
                     │    │  apply_code_diff()       │     │                    │
                     │    │  copytree + patch -p0    │     │                    │
                     │    └────────────┬─────────────┘     │                    │
                     │          ✓ OK?  │  ✗ FAIL?          │                    │
                     │        break◄──┤  ├──► retry        │                    │
                     │                │  │    (error_msg    │                    │
                     │                │  │     fed back)    │                    │
                     └────────────────┼──┴──────────────────┘                    │
                                      │                                          │
                                      ▼                                          │
               diffs_{run}_epoch{N}/code_diff_idea_{i}.diff                      │
               repo_variants_{run}_epoch{N}/idea_{i}/  (patched codebase)        │
                                       │                                        │
                    ╔══════════════════▼══════════════════╗                     │
                    ║      STEP 3: STATS + UPLOAD          ║                     │
                    ╚══════════════════╤══════════════════╝                     │
                                       │                                        │
              ┌────────────────────────┼────────────────────────┐               │
              ▼                        ▼                        ▼               │
  compute_idea_stats()    zip_and_upload_repo_variants()   move_diffs_and       │
  → idea_stats_{run}/     → HuggingFace storage            repo_variants        │
    epoch{N}.json                                          → *_claude/ dirs      │
                                       │                                        │
                    ╔══════════════════▼══════════════════╗                     │
                    ║   STEP 4: FIXED WAIT (90 minutes)    ║                     │
                    ╚══════════════════╤══════════════════╝                     │
                                       │                                        │
                         [EXTERNAL — NOT IN REPO]                               │
                         HuggingFace Scheduler → B200 GPUs                      │
                         Workers run training → log to W&B                      │
                         W&B run: {run}_epoch{N}_b200_idea_{i}                  │
                                       │                                        │
                    ╔══════════════════▼══════════════════╗                     │
                    ║    STEP 5: LOG RETRIEVAL LOOP        ║                     │
                    ╚══════════════════╤══════════════════╝                     │
                                       │                                        │
           last_retrieved = 0          │                                        │
           WHILE retrieved ≤ 30% of submitted:                                  │
                                       │                                        │
              ┌────────────────────────▼────────────────────────┐               │
              │  retrieve_training_logs()                        │               │
              │  W&B API: filter runs by fnmatch pattern         │               │
              │                                                  │               │
              │  GRPO:    history["eval/mean_reward"]            │               │
              │           final = max(eval_rewards)              │               │
              │                                                  │               │
              │  NanoGPT: parse output.log "val_loss:" lines     │               │
              │           final = min(val_losses)                │               │
              │           time_to_target = first step ≤ 3.28    │               │
              └────────────┬────────────────────────────────────┘               │
                           │                                                     │
              ┌────────────▼──────────┐    ┌──────────────────┐                 │
              │  Write metrics.json   │    │  Write           │                 │
              │  per idea             │    │  ranked_ideas.json│                 │
              └───────────────────────┘    └──────────────────┘                 │
                           │                                                     │
           retrieved > 30%?├─ YES ──────────────────────────────────────────────┘
                           │
                           └─ NO → sleep(20 min) → retry poll
                                                          │
                                                          └──► (loop back to poll)


                    ┌──────────────────────────────────┐
                    │  After ALL epochs complete:       │
                    │  update_database(run, epochs-1)   │
                    │  → Final database.json merge      │
                    └──────────────────────────────────┘


╔══════════════════════════════════════════════════════════════════════════════════╗
║                            DATABASE.JSON EVOLUTION                               ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  Epoch 0:  ideas → diffs → training → ranked_ideas → update_database()          ║
║                                                          │                       ║
║                                                          ▼                       ║
║  Epoch 1:  database.json (epoch 0 entries)                                       ║
║            ├─ EXPLOIT: top winners (acc > 0.49 / loss < 3.255)                  ║
║            └─ EXPLORE: random sample of all entries                              ║
║                                                          │                       ║
║                                                          ▼                       ║
║  Epoch N:  database grows, exploit ratio increases 50% → 80%                    ║
║            Dedup: best result per (epoch, idea_id) kept                         ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

## Key Design Points

| Aspect | Detail |
|---|---|
| **Parallelism** | 10 threads generate code diffs simultaneously |
| **Self-correction** | Up to 10 retry trials per diff, feeding error back to LLM |
| **Exploit/Explore** | Ratio starts 50/50 at epoch 1, ramps to 80/20 by epoch 8 |
| **Polling guard** | Only stops waiting once ≥30% of submitted ideas return W&B logs |
| **External boundary** | GPU scheduling (HuggingFace + B200 workers) is entirely outside the repo |
