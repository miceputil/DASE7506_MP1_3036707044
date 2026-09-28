"""Build the three report figures from recorded experiment results (stdlib only)."""

import csv
import json
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
with (ROOT / "code/run_log.csv").open(newline="") as file:
    RUNS = {row["run_id"]: row for row in csv.DictReader(file)}


def tag(name, **attributes):
    attrs = " ".join(f'{key.replace("_", "-")}="{escape(str(value))}"' for key, value in attributes.items())
    return f"<{name} {attrs}/>"


def label(x, y, value, size=15, anchor="middle", fill="#243047", weight=400):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{escape(str(value))}</text>'
    )


def save(name, title, body, width=900, height=470):
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">'
        f'<rect width="100%" height="100%" fill="#ffffff"/>'
        f'{label(width / 2, 34, title, 20, weight=700)}'
        + "".join(body)
        + "</svg>\n"
    )
    (OUT / name).write_text(svg, encoding="utf-8")


def axis(body, x0, y0, x1, y1, yticks, ymin, ymax):
    body.append(tag("line", x1=x0, y1=y0, x2=x0, y2=y1, stroke="#6b7788"))
    body.append(tag("line", x1=x0, y1=y1, x2=x1, y2=y1, stroke="#6b7788"))
    for tick in yticks:
        y = y1 - (tick - ymin) / (ymax - ymin) * (y1 - y0)
        body.append(tag("line", x1=x0, y1=y, x2=x1, y2=y, stroke="#dce2e9"))
        body.append(label(x0 - 12, y + 5, f"{tick:.2f}", 12, "end"))


# Figure 1: all points use the same 9,830,400 training targets; the baseline
# reaches that budget with a different step/batch combination.
milestones = [
    ("baseline", "Baseline"),
    ("swiglu-s17", "SwiGLU"),
    ("swiglu-rope-b16-s17", "RoPE"),
    ("rope-depth6", "Depth 6"),
    ("rope-theta1000", "theta 1000"),
    ("rope-theta1000-qknorm", "QK-Norm"),
    ("width192", "Width 192"),
]
body = []
x0, x1, y0, y1 = 80, 860, 80, 360
ymin, ymax = 1.60, 2.12
axis(body, x0, y0, x1, y1, [1.60, 1.70, 1.80, 1.90, 2.00, 2.10], ymin, ymax)
points = []
for i, (run_id, name) in enumerate(milestones):
    x = x0 + (x1 - x0) * i / (len(milestones) - 1)
    score = float(RUNS[run_id]["validation_bpb"])
    y = y1 - (score - ymin) / (ymax - ymin) * (y1 - y0)
    points.append((x, y, score, name))
body.append(tag("polyline", points=" ".join(f"{x},{y}" for x, y, _, _ in points), fill="none", stroke="#2374ab", stroke_width=3))
for x, y, score, name in points:
    body.append(tag("circle", cx=x, cy=y, r=5.5, fill="#2374ab"))
    body.append(label(x, y - 12, f"{score:.3f}", 12, weight=600))
    body.append(label(x, 384, name, 11))
body.append(label(470, 433, "Equal training targets; baseline uses a different step/batch schedule", 13))
save("fig1_iteration_path.svg", "Representative architecture iterations", body, height=450)


# Figure 2: a controlled capacity sweep at 2,400 steps, batch 16, seed 17.
body = []
x0, x1, y0, y1 = 100, 820, 75, 355
xmin, xmax, ymin, ymax = 1.8, 4.4, 1.66, 1.73
axis(body, x0, y0, x1, y1, [1.66, 1.68, 1.70, 1.72], ymin, ymax)
for run_id, color in [("width160", "#26866c"), ("width192", "#2374ab"), ("width224", "#a55b30")]:
    row = RUNS[run_id]
    params = int(row["parameters"]) / 1_000_000
    score = float(row["validation_bpb"])
    x = x0 + (params - xmin) / (xmax - xmin) * (x1 - x0)
    y = y1 - (score - ymin) / (ymax - ymin) * (y1 - y0)
    body.append(tag("circle", cx=x, cy=y, r=9, fill=color))
    body.append(label(x, y - 17, f'width {run_id[5:]}: {score:.3f}', 15, weight=600))
    body.append(label(x, y + 29, f'{row["train_seconds"][:3]} s train', 11))
for tick in [2, 3, 4]:
    x = x0 + (tick - xmin) / (xmax - xmin) * (x1 - x0)
    body.append(label(x, 382, str(tick), 12))
body.append(label(460, 417, "Parameters (millions)", 14))
save("fig2_capacity_tradeoff.svg", "Capacity versus validation quality", body, height=440)


# Figure 3: comparison of the two recorded 4,800-step trajectories.
paths = [
    ("Final, dropout 0.10", "lr-verify-warmup200-cosine-dropout010-s4800-b32-s17", "#2374ab"),
    ("No dropout (historical code)", "swiglu-rope-width192-head32-theta10000-qknorm-b32-s4800-s17", "#a55b30"),
]
body = []
x0, x1, y0, y1 = 80, 830, 70, 350
ymin, ymax = 1.54, 1.90
axis(body, x0, y0, x1, y1, [1.55, 1.65, 1.75, 1.85], ymin, ymax)
for name, run_id, color in paths:
    data = json.loads((ROOT / "code/runs" / run_id / "metrics.json").read_text())
    points = []
    for item in data["validation_history"]:
        x = x0 + (item["step"] - 600) / 4200 * (x1 - x0)
        y = y1 - (item["bpb"] - ymin) / (ymax - ymin) * (y1 - y0)
        points.append((x, y))
    body.append(tag("polyline", points=" ".join(f"{x},{y}" for x, y in points), fill="none", stroke=color, stroke_width=3))
    for x, y in points:
        body.append(tag("circle", cx=x, cy=y, r=4, fill=color))
    body.append(label(points[-1][0] - 6, points[-1][1] - 12, f'{data["validation_history"][-1]["bpb"]:.3f}', 14, "end", color, 700))
for step in range(600, 4801, 600):
    x = x0 + (step - 600) / 4200 * (x1 - x0)
    body.append(label(x, 377, step, 12))
body.append(tag("line", x1=160, y1=412, x2=192, y2=412, stroke="#2374ab", stroke_width=3))
body.append(label(200, 417, paths[0][0], 13, "start"))
body.append(tag("line", x1=470, y1=412, x2=502, y2=412, stroke="#a55b30", stroke_width=3))
body.append(label(510, 417, paths[1][0], 13, "start"))
save("fig3_final_curve.svg", "Validation BPB over 4,800 steps", body, height=450)
