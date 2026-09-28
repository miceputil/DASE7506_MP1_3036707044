# MP1 code — installation and usage

Read the [repository overview](../README.md) and [project guide](../GUIDE.md) for the model summary, assignment rules, deadlines and peer review. This README contains the detailed installation, training and evaluation instructions.

All commands below run from **code/**. Data and the tokenizer are included. No API key, pretrained weights or additional dataset download is needed; after installing dependencies, training and evaluation work offline.

## 1. Install

Use **Python 3.12**. From the extracted package directory:

```bash
cd code
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.

Install PyTorch for **one** device:

```bash
# Linux/Windows CPU: recommended; no GPU needed
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

For an NVIDIA GPU with a compatible driver, use this command **instead**:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

For macOS, install `torch==2.7.1` from the default PyPI index and run on CPU. After installing PyTorch, install the remaining dependencies and check the model:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Linux CPU commands were verified with Python 3.12 and PyTorch 2.7.1+cpu. Windows/macOS timings have not been measured.

## 2. Train and evaluate

**Quick installation check** — 10 training steps, then full-test evaluation:

```bash
python train.py --implementation model --steps 10 --run-dir runs/smoke
python evaluate.py --checkpoint runs/smoke/checkpoint.pt --split test
```

This checks that the pipeline works; its score is **not** the full baseline. Each training run needs a new output directory.

**Full baseline** — 1,200 training updates, then evaluation:

```bash
python train.py --implementation model --device cpu --threads 4 --seed 17 --run-dir runs/baseline
python evaluate.py --checkpoint runs/baseline/checkpoint.pt --device cpu --precision fp32 --split test
```

The baseline has four GPT blocks, width 128, four attention heads and **1,088,256 parameters**, and achieves approximately **2.10 test BPB**. On the reference four-thread Xeon Platinum 8457C, measured training took about **311 seconds** and scoring **5.92 seconds**, excluding installation and loading. These are reference measurements, not laptop guarantees or a fixed time allowance.

**Your model** — edit `student.py` and supporting files, then:

```bash
python train.py --implementation student --seed 17 --eval-every 300 --run-dir runs/my-model
python evaluate.py --checkpoint runs/my-model/checkpoint.pt --split validation
# Freeze the final method before testing:
python evaluate.py --checkpoint runs/my-model/checkpoint.pt --split test
```

Training writes `checkpoint.pt` and `metrics.json`. Evaluation writes `test_cpu_fp32.json` (or the corresponding device/split name) and per-window losses. Submit the **bpb** value from the complete-test JSON, not token perplexity or validation BPB. Default evaluation is FP32. Add `--device cuda` for GPU runs; training can use BF16, but ranked evaluation must use FP32 and remain reproducible on CPU. The supplied CUDA runner caps PyTorch allocation at 20 GB; driver overhead is additional.

**Frozen final model** — select the 4,800-step checkpoint using validation,
then report its full-test CPU FP32 result. This run uses residual dropout
`0.10`, no EMA, AdamW with a `0.001` peak learning rate, 200 warmup steps,
cosine decay to 10% of the peak, and batch size `32`. It processes
39,321,600 training targets. The selected checkpoint is
`runs/lr-verify-warmup200-cosine-dropout010-s4800-b32-s17/checkpoint.pt`.

| Frozen result | Value |
|---|---:|
| Validation BPB (selection) | 1.5751837398682695 |
| Full-test BPB (final report) | **1.5992733523443317** |
| CPU FP32 test scoring time | 9.624154124991037 seconds |
| Measured peak evaluation RSS | 2.112 GiB |

Checkpoint SHA-256:
`04f8d2c00a70d528b2a3b31de3ad3717cca93c18df616550f69cb59645bacbfa`.
The matching `student.py` SHA-256 is
`47477124058e73a234d60d75af31af1d80e12d3a89d4cb1c1f831e7ae0ee27e5`.
Preserve this implementation and checkpoint together for reproduction.

Reproduce training from scratch in a new run directory:

```bash
python train.py \
  --implementation student \
  --config configs/student_swiglu_rope_width192_head32_theta10000_qknorm_dropout010.json \
  --device cpu \
  --threads 4 \
  --seed 17 \
  --steps 4800 \
  --eval-every 600 \
  --save-every 600 \
  --run-dir runs/reproduce-final-dropout010-s4800-s17
```

Evaluate the frozen checkpoint, without retraining:

```bash
python evaluate.py \
  --checkpoint runs/lr-verify-warmup200-cosine-dropout010-s4800-b32-s17/checkpoint.pt \
  --device cpu --threads 4 --precision fp32 --split test
```

Other learning-rate, EMA, and dropout runs are retained in `run_log.csv`,
saved run directories, and snapshots as historical evidence. Some older
command lines do not describe this frozen recipe.

**Historical RoPE ablation** — this run used the earlier training recipe and
changed only the position encoding.

```bash
python train.py \
  --implementation student \
  --config configs/student_swiglu_rope.json \
  --device cpu \
  --threads 4 \
  --seed 17 \
  --steps 2400 \
  --batch-size 16 \
  --eval-every 600 \
  --run-dir runs/swiglu-rope-b16-s17
python evaluate.py --checkpoint runs/swiglu-rope-b16-s17/checkpoint.pt --device cpu --precision fp32 --split validation
```

Compare this validation result with the learned-position raw checkpoint from
the equal-target run. Do not use the test split for this selection.

**Depth-6 RoPE ablation** — this experiment changes only the Transformer
depth from four to six blocks. It keeps the same 9,830,400 processed targets
as the depth-4 RoPE control.

```bash
python train.py \
  --implementation student \
  --config configs/student_swiglu_rope_depth6.json \
  --device cpu \
  --threads 4 \
  --seed 17 \
  --steps 2400 \
  --batch-size 16 \
  --eval-every 600 \
  --run-dir runs/swiglu-rope-depth6-b16-s2400-s17
python evaluate.py --checkpoint runs/swiglu-rope-depth6-b16-s2400-s17/checkpoint.pt --device cpu --precision fp32 --split validation
```

Use the depth-4 RoPE validation BPB `1.7837319638959395` as the control.

**QK-Norm ablation** — keep the selected RoPE configuration fixed and toggle
only `qk_norm` for the comparison. Historical theta-search artifacts remain
in `configs/` and `run_log.csv` solely to preserve reproducibility; no further
theta sensitivity runs are part of the experiment plan.

**Width expansion after QK-Norm** — the selected final candidate uses
`width=224`, `heads=7`, `depth=6`, and `ffn_hidden=600` with RoPE theta 1000
and QK-Norm:

```bash
python train.py --implementation student \
  --config configs/student_swiglu_rope_width224_qknorm.json \
  --device cpu --threads 4 --seed 17 --steps 2400 --batch-size 16 \
  --eval-every 600 --run-dir runs/swiglu-rope-width224-qknorm-b16-s2400-s17
python evaluate.py --checkpoint runs/swiglu-rope-width224-qknorm-b16-s2400-s17/checkpoint.pt \
  --device cpu --precision fp32 --split validation
```

This run has 4,094,144 parameters and validation BPB `1.6779608112533306`.

**Sweet-spot sweep** — after the width-224 run, the intermediate capacity
and attention-head experiments were trained with the same 9,830,400 targets:

| Experiment | Parameters | Validation BPB |
|---|---:|---:|
| width 160, head_dim 32 | 2,195,008 | 1.7121827833360663 |
| width 192, head_dim 32 | 3,057,792 | 1.688755912495611 |
| width 192, head_dim 16 | 3,057,408 | 1.6967530788715341 |
| width 192, head_dim 48 | 3,058,176 | 1.689923380639908 |
| width 192, head_dim 64 | 3,058,560 | 1.6886063080857077 |

The long-run candidate uses width 192, head_dim 64, and the same RoPE/QK-Norm
recipe. It uses a 7,200-step schedule with validation every 1,200 steps and
selects the earliest checkpoint within 0.005 BPB of the best observed value:

| Steps | Validation BPB |
|---:|---:|
| 1,200 | 1.808342005653997 |
| 2,400 | 1.715233316902453 |
| 3,600 | 1.664689095016607 |
| 4,800 | 1.639670684354649 |
| 6,000 | 1.626928924657661 |
| 7,200 | 1.625597416028266 |

The selected sweet-spot checkpoint is
`runs/swiglu-rope-width192-head64-qknorm-b16-s7200-s17/checkpoint_step_6000.pt`.
The 7,200-step checkpoint improves only 0.001332 BPB over 6,000 steps.

Reproduce the long run with:

```bash
python train.py --implementation student \
  --config configs/student_swiglu_rope_width192_head64_qknorm.json \
  --device cpu --threads 4 --seed 17 --steps 7200 --batch-size 16 \
  --eval-every 1200 --save-every 600 \
  --run-dir runs/swiglu-rope-width192-head64-qknorm-b16-s7200-s17
python evaluate.py \
  --checkpoint runs/swiglu-rope-width192-head64-qknorm-b16-s7200-s17/checkpoint_step_6000.pt \
  --device cpu --precision fp32 --split validation
```

**Head-dimension=32 confirmation** — a separate 4,800-step run uses the
selected RoPE configuration and `head_dim=32` (`width=192`, `heads=6`). Its
validation BPB is `1.6253206826113495`.

```bash
python train.py --implementation student \
  --config configs/student_swiglu_rope_width192_head32_theta10000_qknorm.json \
  --device cpu --threads 4 --seed 17 --steps 4800 --batch-size 16 \
  --eval-every 600 \
  --run-dir runs/swiglu-rope-width192-head32-theta10000-qknorm-b16-s4800-s17
```

## 3. Files and model interface

| Files | Use |
|---|---|
| `model.py`, `configs/baseline.json` | Runnable baseline; preserve for comparisons. |
| `student.py`, `train.py` | Your model factory and training recipe; add supporting code as needed. |
| `common.py`, `evaluate.py` | Fixed data checks, windows and scorer; keep unchanged. |
| `data/` | Supplied splits, tokenizer and dataset hashes; keep unchanged. |
| `tests/test_contract.py` | Checks your model's causality, normalization, independence and gradients. |
| `RUN_LOG_TEMPLATE.csv` | Optional experiment-log template. |
| `PACKAGE_MANIFEST.json` | Release hashes; paths are relative to the package root containing code/ and guide/. |

- `build_model(config)` returns a PyTorch model with `context=256`.
- The supplied trainer calls `forward(ids)` for unnormalized logits; the scorer calls `predict_log_probs(ids)` for finite, normalized natural-log probabilities. Both outputs have shape `[batch, time, 2048]`.
- A prediction at position t may use only the observed prefix through t. Reset temporary state between independent windows, examples and scoring passes. Compact training-derived assets may be reused across windows; evaluation-prefix state may not.
- Checkpoints record the implementation module and configuration. Include that module and every required asset so the evaluator can reconstruct the submitted predictor. No optimizer state is required for direct evaluation.
- Training length, architecture, optimizer, regularization, self-trained weight averaging and ensembles may change within the guide's constraints. Log all seeds, processed training targets, checkpoint ancestry and search costs; reusing a checkpoint does not erase its training cost. No particular seed or score improvement is mandated.

## 4. Benchmark and resource measurements

**Fixed score.** Protocol `7506-mp1-wt2-v2`: WikiText-2 raw text, train-fitted BPE-2048, independent windows of 256 targets, including the final short window. Every target except the first token of each split is scored once. Input windows share a boundary token but carry no state. BPB is summed negative log-base-2 next-token probability divided by the split's entire raw UTF-8 byte length, including the first token's bytes.

| Split | Scored targets | UTF-8 bytes |
|---|---:|---:|
| Validation | 376,599 | 1,148,007 |
| Test | 428,405 | 1,292,013 |

Use validation for all development and checkpoint/mixture selection. Weights, statistics and retrieval entries must derive only from training text. The public test text enables reproduction; it must not be used to tune the method. Once frozen, the same predictor may be evaluated repeatedly for timing or reproduction. Token perplexity is not directly comparable with published word-level perplexity.

Measure all three limits for the same frozen predictor:

- **CPU time ≤5× baseline:**
- **Peak RAM ≤4 GiB:**
- **Inference assets ≤64 MiB uncompressed:** 

For a reproducible CPU measurement, use the same evaluator command for the
baseline and candidate and record the JSON `seconds` field. On macOS,
`/usr/bin/time -l` reports peak RSS in bytes; divide it by `2**30` for GiB.
The inference-asset total should include the selected checkpoint, the model
implementation, configuration, tokenizer, and other files required to load
the predictor, but not optimizer state or training logs.

## 5. Prepare your submission and reproduce a peer

The [guide](../GUIDE.md) specifies the deadline and website workflow. Include the following in your immutable code repository:

- **Report, at most 10 pages including figures, tables and references** 
- **Reproduction instructions**

Your final website submission must link to this code and the matching complete checkpoint bundle. The website generates the Issue JSON automatically. Keep all inference assets downloadable for verification.

To check a peer, obtain their exact code version and checkpoint, follow their installation instructions, and run their frozen model with the supplied evaluator:

```bash
python evaluate.py --checkpoint /path/to/peer-checkpoint.pt --device cpu --precision fp32 --split test --output peer-test.json
```

Compare reproduced BPB with the reported score. Submit **Peer Review Report** with the reproduced score; optionally include the command, environment, difference and evidence/log link.  The instructor adjudicates discrepancies. Confirmed discrepancies during the seven-day review earn bonus credit under the announced marking policy.

## 6. Data attribution

WikiText-2 was introduced by Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher in [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843). The text is by Wikipedia contributors. The [upstream dataset](https://huggingface.co/datasets/Salesforce/wikitext) identifies [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and the [GNU Free Documentation License](https://www.gnu.org/licenses/fdl-1.3.html); retain these notices when redistributing the data.

The supplied `wikitext-2-raw-v1` splits preserve revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Rows are joined with newlines and encoded as UTF-8; the tokenizer is fitted only to training text. Dataset hashes are in `data/manifest.json`. These dataset notices do not assign a new license to the surrounding classroom code.
