<h1 align="center">
  <img src="assets/icon.png" alt="MindCube Icon" width="28" height="28" style="vertical-align: middle; margin-right: 0px;">
  MindCube: Spatial Mental Modeling from Limited Views
</h1>

<!-- Badges -->
<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2506.21458-b31b1b.svg)](https://arxiv.org/abs/2506.21458)
[![Homepage](https://img.shields.io/badge/🏠-Homepage-blue.svg)](https://mind-cube.github.io/)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-yellow.svg)](https://huggingface.co/datasets/MLL-Lab/MindCube)
[![Checkpoints](https://img.shields.io/badge/🤗-Checkpoints-green.svg)](https://huggingface.co/MLL-Lab/models)

</div>

## 🌟 Overview

MindCube is a modular framework for generating and evaluating spatial reasoning datasets for multimodal AI models.

## ⚡ Phase A (Deterministic 50/50 Smoke Pipeline)

This mode is designed for fast correctness checks before full GPU training.

### One-command run

```bash
bash scripts/phase_a/run_phase_a_smoke.sh
BOOTSTRAP_SKIP_INSTALL=true bash scripts/phase_a/run_phase_a_smoke.sh
```
