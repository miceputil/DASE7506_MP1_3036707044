# DASE7506 MP1

**Student ID:** 3036707044

**Final full-test BPB:** 1.599273

This project improves a small GPT trained from random initialization using only the course-provided WikiText-2 training data. The supplied BPE-2048 tokenizer and evaluation protocol are unchanged. Settings were selected on validation before freezing the final model for full-test evaluation.

See the experimental in [report](report.pdf) for methods, comparisons, ablations, and critical analysis. 

## 1. Model and results

The final decoder-only Transformer uses six layers, width 192, and six causal self-attention heads with head dimension 32. Its main changes are a SwiGLU feed-forward network (hidden size 512), RoPE (θ=10000), head-wise Q/K LayerNorm, and residual dropout 0.10. It retains LayerNorm, tied token embeddings, and a 256-token context.

Lower bits per byte (BPB) is better. All scoring uses CPU FP32.

| Metric | Baseline | Final model |
|---|---:|---:|
| Validation BPB | 2.0711 | **1.575184** |
| Full-test BPB | 2.1013 | **1.599273** |
| CPU full-test scoring time | 5.844 s | 9.624 s (1.65× baseline) |
| Parameters | 1,088,256 | 3,057,792 |

## 2. Training platform and settings

Experiments were run locally on a **MacBook Pro with an Apple M1 Pro chip and 32 GB RAM**, using **macOS, Python 3.12, and PyTorch 2.7.1**. Training and evaluation used **CPU FP32 with four PyTorch threads**.

The final run uses seed 17, batch size 32, 4,800 updates (39,321,600 training targets), and AdamW with betas (0.9, 0.999), weight decay 0.1, learning-rate scale 0.001, 200-step warmup, and cosine decay. EMA is disabled. The original run took approximately **44 minutes**; experiment results and search costs are recorded in [`code/run_log.csv`](code/run_log.csv).

## 3. Installation

On macOS, run from the repository root:

```bash
cd code
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## 4. Evaluate the frozen model

The frozen checkpoint is stored locally at `code/runs/lr-verify-warmup200-cosine-dropout010-s4800-b32-s17/checkpoint.pt`. Checkpoint and implementation hashes are listed in report Section 4.3.

Run from `code/`:

```bash
python evaluate.py \
  --checkpoint runs/lr-verify-warmup200-cosine-dropout010-s4800-b32-s17/checkpoint.pt \
  --device cpu --threads 4 --precision fp32 --split test
```

The evaluator saves `test_cpu_fp32.json` in the run directory. Report its **`bpb`** field; the frozen model's full-precision result is `1.5992733523443317`.

## 5. Reproduce training

Run from `code/`, choosing a new or empty `--run-dir`:

```bash
python train.py \
  --implementation student \
  --config configs/student_swiglu_rope_width192_head32_theta10000_qknorm_dropout010.json \
  --device cpu --threads 4 --seed 17 --steps 4800 \
  --eval-every 600 --save-every 600 \
  --run-dir runs/reproduce-final-s17
```

Training saves `checkpoint.pt`, `metrics.json`, and intermediate checkpoints, with validation and checkpoint saving every 600 steps.

## 6. Data attribution

WikiText-2 was introduced by Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher in [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843). The text is by Wikipedia contributors. The [upstream dataset](https://huggingface.co/datasets/Salesforce/wikitext) identifies [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and the [GNU Free Documentation License](https://www.gnu.org/licenses/fdl-1.3.html); retain these notices when redistributing the data.

The supplied `wikitext-2-raw-v1` splits preserve revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Rows are joined with newlines and encoded as UTF-8; the tokenizer is fitted only to training text. Dataset hashes are in [`code/data/manifest.json`](code/data/manifest.json). These dataset notices do not assign a new license to the surrounding classroom code.

## 7. AI assistance

AI tools assisted with improvement planning, controlled and ablation experiment design, code implementation, result organization, and report drafting. All numerical results are based on saved project runs.
