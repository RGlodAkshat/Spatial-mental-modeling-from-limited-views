# Phase A GPU Handoff

## Full training (paper-like task target)

```bash
bash scripts/phase_a/run_phase_a_smoke.sh
bash experiments/sft/train_qwen_sft.sh experiments/sft/config_plain_cgmap_ffr_out.sh
```

## Smoke training (quick sanity)

```bash
bash experiments/sft/train_qwen_sft.sh experiments/sft/config_plain_cgmap_ffr_out_smoke.sh
```
