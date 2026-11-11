# Training + evaluation — Kaggle GPU required

These are the exact scripts from the competition build. They do NOT
run on CPU and are NOT part of the CPU reproduce. They are published
so the training and evaluation method is auditable.

## Files

- `full_train_manual_lora_v19prep.py` — manual LoRA training in pure
  torch (r=8, alpha=16), no PEFT (unusable in the Kaggle environment).
  Pre-registered retrain: end-of-sequence appended after the fixed
  file + bug-class label dropped, on the frozen v18 corpus (20,000
  records). Staged: epoch 1 (~5,000 steps) first, kill pilot before
  epoch 2. Kill criteria: KILL the LoRA route if <4/32 gate-accepted
  fixes on the stratified held-out singles pilot, or fixes in at most
  one operator, or parsable non-identical output <30%.
  Run on Kaggle GPU (L4x4); needs the Gemma 4 12B QAT checkpoint from
  the competition input and the `train_records_20k.jsonl` corpus.
- `eval_dev.py` — held-out synthetic evaluation with the trained
  adapter (or `--no-adapter` for the base-model baseline). Same Kaggle
  GPU requirement. Label every number from it "held-out synthetic".

## Status (honest, updated 8 Nov 2026)

- v1 training completed 1 Nov (10,000 steps, training loss 0.0191,
  training loss only). Held-out pilots: 0/18 — the v1 model never
  emitted EOS and looped corrupted output to the 512-token cap.
- Gate 1 (adapter attaches, 328/328) PASSED. Gate 2 (serving)
  INCONCLUSIVE; vLLM serving of a trained adapter UNTESTED.
- v19 retrain + pre-registered kill test ran 7 Nov 2026 (EOS
  appended, bug-class label dropped): 0/32 fixed, 0 operators,
  4/32 parsable — the LoRA route was KILLED. The failure is in
  the training, not the tasks.
- The trained v1 adapter is NOT in this repo (failed artefact,
  0/18), and the v19 adapter does not ship either (0/32 on the
  kill test). A future adapter would ship as a GitHub Release
  asset with its evaluation numbers only if it passes the kill
  test — never in the git tree (see `.gitignore`).
