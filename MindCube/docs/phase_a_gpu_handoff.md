# Phase A GPU Handoff

## Full training (paper-like task target)

```bash
# Rebuild canonical data first, then run full SFT
bash scripts/phase_a/run_phase_a_smoke.sh
bash experiments/sft/train_qwen_sft.sh experiments/sft/config_plain_cgmap_ffr_out.sh
```

## Smoke training (quick sanity)

```bash
bash experiments/sft/train_qwen_sft.sh experiments/sft/config_plain_cgmap_ffr_out_smoke.sh
```

Smoke output root:
`checkpoints/sft/smoke/plain_cgmap_ffr_out_smoke/`

## Multi-checkpoint inference + evaluation

```bash
python scripts/phase_a/eval_checkpoints.py \
  --task plain_cgmap_ffr_out_smoke \
  --checkpoints-root checkpoints/sft/smoke/plain_cgmap_ffr_out_smoke \
  --input-file data/prompts/general/MindCube_tinybench_plain_cgmap_ffr_out_smoke.jsonl \
  --results-dir data/results/sft/smoke/plain_cgmap_ffr_out_smoke \
  --eval-dir data/evaluate/sft/smoke/plain_cgmap_ffr_out_smoke
```

## Main knobs to tune

- `experiments/sft/config_hardware.sh`
: `GPU_DEVICES`, `NUM_PROCESSES`, `BATCH_SIZE`, `GRAD_ACCUM_STEPS`
- `experiments/sft/config_plain_cgmap_ffr_out.sh`
: `LEARNING_RATE`, `NUM_EPOCHS`, `SAVE_STEPS`, `MODEL_MAX_LENGTH`, output root `checkpoints/sft/plain_cgmap_ffr_out/`
- `experiments/sft/config_plain_cgmap_ffr_out_smoke.sh`
: `MAX_STEPS`, `SAVE_STEPS`, `SAVE_TOTAL_LIMIT`, output root `checkpoints/sft/smoke/plain_cgmap_ffr_out_smoke/`
- `experiments/sft/train_qwen_sft.sh`
: optional env/config knobs `MAX_STEPS`, `REPORT_TO`, `USE_BF16`, `DATALOADER_NUM_WORKERS`, `EXTRA_TRAIN_ARGS`
