# DASE7506 MP1

This project trains a small GPT from random initialization on the course-provided WikiText-2 data, fixed BPE-2048 tokenizer, and evaluation protocol. The final model combines SwiGLU, RoPE, head-wise Q/K LayerNorm, residual dropout, a larger architecture, and a longer training schedule.

The [experiment report](report/report.md) presents the method, comparisons, ablations, and limitations. See the [assignment guide](GUIDE.md) for the rules and [code documentation](code/README.md) for detailed installation and evaluation instructions.

## Final results

The model was selected on validation and frozen before full-test FP32 CPU evaluation. Lower bits per byte (BPB) is better.

| Metric | Baseline | Final model |
|---|---:|---:|
| Validation BPB | 2.0711 | **1.5752** |
| Full-test BPB | 2.1013 | **1.5993** |
| CPU full-test scoring time | 5.844 s | 9.624 s (1.65× baseline) |
| Parameters | 1,088,256 | 3,057,792 |

The final model's measured peak evaluation RAM was **2.112 GiB**, and its uncompressed inference assets total approximately **11.82 MiB**. All three resource measurements are within the course limits: at most 5× baseline CPU scoring time, 4 GiB RAM, and 64 MiB of inference assets. See the [measurement procedure](code/README.md#4-benchmark-and-resource-measurements).

The final configuration uses width 192, depth 6, six attention heads (head dimension 32), FFN hidden size 512, LayerNorm, SwiGLU, RoPE with θ=10000, head-wise Q/K LayerNorm, and residual dropout 0.10. Training used seed 17, batch size 32, 4,800 steps, AdamW, a 200-step warmup, and cosine learning-rate decay. EMA was not used.

## Installation and checks

Python 3.12 is required. Run the commands below from the repository root. On macOS, `requirements.txt` can be installed directly. On Linux or Windows, first choose the appropriate PyTorch installation command in the [code documentation](code/README.md#1-install).

```bash
cd code
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The code documentation also gives the PowerShell activation command. Training and evaluation use the included data and tokenizer; no external training corpus or pretrained weights are needed.

## Evaluate the frozen checkpoint

Place the frozen `checkpoint.pt` at `code/runs/lr-verify-warmup200-cosine-dropout010-s4800-b32-s17/checkpoint.pt`. Its SHA-256 is:

```text
04f8d2c00a70d528b2a3b31de3ad3717cca93c18df616550f69cb59645bacbfa
```

The matching `code/student.py` SHA-256 is:

```text
47477124058e73a234d60d75af31af1d80e12d3a89d4cb1c1f831e7ae0ee27e5
```

**Release status:** The checkpoint exists locally, but `code/runs/` is Git-ignored and this repository does not yet provide a download link for a fresh clone. Before final submission, an immutable checkpoint bundle matching the hash above must be published and linked here. The direct evaluation command below requires that file; retraining is not a substitute for a ready-to-evaluate checkpoint bundle.

After obtaining the checkpoint, run from `code/`:

```bash
python evaluate.py \
  --checkpoint runs/lr-verify-warmup200-cosine-dropout010-s4800-b32-s17/checkpoint.pt \
  --device cpu --threads 4 --precision fp32 --split test
```

Report the `bpb` field from the output JSON, not `token_ppl`. The recorded full-test output is `test_cpu_fp32.json` in the local run directory. The test split was not used for model selection.

## Reproduce training from scratch

Run from `code/`, choosing a new or empty `--run-dir`:

```bash
python train.py \
  --implementation student \
  --config configs/student_swiglu_rope_width192_head32_theta10000_qknorm_dropout010.json \
  --device cpu --threads 4 --seed 17 --steps 4800 \
  --eval-every 600 --save-every 600 \
  --run-dir runs/reproduce-final-s17
```

Training writes `checkpoint.pt`, `metrics.json`, and intermediate checkpoints every 600 steps. This command reproduces the experiment. To verify the reported final score, use the specified frozen checkpoint rather than treating a new training run as the same artifact.

## Data attribution

WikiText-2 was introduced by Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher in [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843). The text is by Wikipedia contributors. The [upstream dataset](https://huggingface.co/datasets/Salesforce/wikitext) identifies [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and the [GNU Free Documentation License](https://www.gnu.org/licenses/fdl-1.3.html); retain these notices when redistributing the data.

The supplied `wikitext-2-raw-v1` splits preserve revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Rows are joined with newlines and encoded as UTF-8; the tokenizer is fitted only to training text. Dataset hashes are in [`code/data/manifest.json`](code/data/manifest.json). These dataset notices do not assign a new license to the surrounding classroom code.

## Repository layout

| Path | Purpose |
|---|---|
| [`GUIDE.md`](GUIDE.md) | Assignment rules and resource limits |
| [`code/README.md`](code/README.md) | Detailed installation, training, and evaluation instructions |
| [`code/model.py`](code/model.py) | Unmodified baseline GPT |
| [`code/student.py`](code/student.py) | Final model implementation |
| [`code/train.py`](code/train.py) | Training entry point and learning-rate schedule |
| [`code/configs/student_swiglu_rope_width192_head32_theta10000_qknorm_dropout010.json`](code/configs/student_swiglu_rope_width192_head32_theta10000_qknorm_dropout010.json) | Final model configuration |
| [`code/run_log.csv`](code/run_log.csv) | Experiment log |
| [`report/report.md`](report/report.md) | Experiment report and figures |

## AI assistance

AI tools assisted with improvement planning, controlled and ablation experiment design, code implementation, result organization, and report drafting. All numerical results are based on saved project runs.
