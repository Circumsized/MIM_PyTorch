"""Generate the SVG diagrams used in README.md and paper_analysis.md.

Each figure is laid out with plain SVG primitives and fixed text grids, so
the output renders the same in GitHub, VS Code and browsers.
"""

import os

OUT_DIR = "assets"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------- palette
BG = "#ffffff"
INK = "#1f2933"
MUTED = "#52606d"
PANEL = "#f5f8fc"
PANEL_ALT = "#eef4ff"
PANEL_WARM = "#fff7ed"
BORDER = "#c9d4e3"
ARROW = "#94a3b8"
ACCENT = "#2563eb"
ACCENT2 = "#0ea5e9"

FONT = "'Segoe UI','Microsoft YaHei','PingFang SC','Noto Sans SC',sans-serif"

TITLE_SIZE = 20
HEAD_SIZE = 14
BODY_SIZE = 12.5
LINE_H = 24  # generous vertical rhythm between body lines


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg_open(w, h):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="{FONT}">'
        f"<defs>"
        f'<marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{ARROW}"/></marker>'
        f'<marker id="arA" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{ACCENT}"/></marker>'
        f'<marker id="arB" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{ACCENT2}"/></marker>'
        f"</defs>"
        f'<rect width="{w}" height="{h}" fill="{BG}"/>'
    )


def title(text, cx, y):
    return (
        f'<text x="{cx}" y="{y}" font-size="{TITLE_SIZE}" font-weight="700" '
        f'fill="{INK}" text-anchor="middle">{esc(text)}</text>'
    )


def rect(x, y, w, h, fill=PANEL):
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="12" fill="{fill}" stroke="{BORDER}" stroke-width="1.4"/>'
    )


def text(cx, y, s, size=BODY_SIZE, fill=MUTED, weight="400", anchor="middle"):
    return (
        f'<text x="{cx:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
        f'font-weight="{weight}" text-anchor="{anchor}">{esc(s)}</text>'
    )


def vcenter(x, y, w, h, head, lines):
    """Head centred, body lines centred below, evenly spaced inside the box."""
    out = []
    block_h = HEAD_SIZE + 8 + LINE_H * len(lines)
    start = y + (h - block_h) / 2 + HEAD_SIZE
    out.append(text(x + w / 2, start, head, size=HEAD_SIZE, fill=INK, weight="700"))
    cy = start + 10
    for ln in lines:
        cy += LINE_H
        out.append(text(x + w / 2, cy, ln, size=BODY_SIZE, fill=MUTED))
    return "".join(out)


def arrow(x1, y1, x2, y2, marker="ar"):
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{ARROW}" stroke-width="2" marker-end="url(#{marker})"/>'
    )


def elbow_v_to_h(x1, y1, x2, y2, marker="arA"):
    """Start at (x1,y1), go down to y2, then horizontally to x2."""
    return (
        f'<path d="M {x1:.1f} {y1:.1f} V {y2 - 24:.1f} H {x2:.1f} '
        f'V {y2:.1f}" fill="none" stroke="{ARROW}" stroke-width="2" '
        f'marker-end="url(#{marker})"/>'
    )


def write(name, body):
    with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
        f.write(body + "</svg>")


# ================================================================ Figure 1
def fig_architecture():
    W, H = 1280, 600
    s = [svg_open(W, H), title("MIM_PyTorch 架构概览", W / 2, 46)]

    cols = [
        (
            "数据集层",
            "dataset.py",
            ["MovingMNIST", "RadarEcho", "_normalize_to_unit", "get_dataloader"],
        ),
        (
            "训练引擎",
            "train.py",
            [
                "_validate_args",
                "train_one_epoch",
                "evaluate",
                "AsyncCheckpointSaver",
                "_load_resume_checkpoint",
            ],
        ),
        (
            "模型核心",
            "mim.py",
            [
                "TensorLayerNorm",
                "SpatioTemporalLSTMCell",
                "MIMS",
                "MIMBlock",
                "MIMN",
                "MIM",
            ],
        ),
        (
            "推理引擎",
            "inference.py",
            [
                "infer_architecture",
                "load_model",
                "predict",
                "CUDAGraphRunner",
                "benchmark",
            ],
        ),
        (
            "评估与可视化",
            "metrics + visualization",
            ["5 项评估指标", "TBLogger", "EpochProgress", "make_video_grid"],
        ),
    ]

    margin, gap = 40, 22
    bw = (W - 2 * margin - 4 * gap) / 5
    by, bh = 96, 330

    for i, (head, sub, items) in enumerate(cols):
        x = margin + i * (bw + gap)
        s.append(rect(x, by, bw, bh))
        out = []
        top = by + 32
        out.append(text(x + bw / 2, top, head, size=HEAD_SIZE, fill=INK, weight="700"))
        out.append(text(x + bw / 2, top + 20, sub, size=BODY_SIZE - 1, fill=ACCENT))
        cy = top + 20 + 26
        for ln in items:
            out.append(text(x + bw / 2, cy, ln, size=BODY_SIZE, fill=MUTED))
            cy += LINE_H
        s.append("".join(out))
        if i < len(cols) - 1:
            aY = by + bh / 2
            s.append(arrow(x + bw + 3, aY, x + bw + gap - 3, aY))

    bw2, bh2 = 460, 88
    bx2, by2 = (W - bw2) / 2, 470
    s.append(rect(bx2, by2, bw2, bh2, fill=PANEL_ALT))
    s.append(
        vcenter(
            bx2,
            by2,
            bw2,
            bh2,
            "依赖与配置",
            ["requirements.txt / CLI args / checkpoint"],
        )
    )
    s.append(arrow(W / 2, by + bh + 3, W / 2, by2 - 3, marker="arA"))

    write("readme_architecture_overview.svg", "".join(s))


# ================================================================ Figure 2
def fig_dataflow():
    W, H = 1280, 660
    s = [svg_open(W, H), title("数据流与状态机", W / 2, 46)]

    steps = [
        ("原始数据", [".npy / .npz"]),
        ("dataset.py", ["_normalize_to_unit", "mmap / DataLoader"]),
        ("模型前向", ["model(frames, ss_bool)"]),
        ("训练路径", ["loss → backward", "→ checkpoint"]),
    ]
    margin, gap = 48, 32
    bw = (W - 2 * margin - 3 * gap) / 4
    by, bh = 100, 170

    for i, (head, items) in enumerate(steps):
        x = margin + i * (bw + gap)
        s.append(rect(x, by, bw, bh))
        s.append(vcenter(x, by, bw, bh, head, items))
        if i < len(steps) - 1:
            aY = by + bh / 2
            s.append(arrow(x + bw + 4, aY, x + bw + gap - 4, aY, marker="arA"))

    # inference branch under the 3rd column
    ib_x = margin + 2 * (bw + gap)
    ib_y, ib_h = 360, 130
    s.append(rect(ib_x, ib_y, bw, ib_h, fill=PANEL_WARM))
    s.append(vcenter(ib_x, ib_y, bw, ib_h, "推理路径", ["predict() → .npy 输出"]))
    s.append(arrow(ib_x + bw / 2, by + bh + 4, ib_x + bw / 2, ib_y - 4, marker="arB"))

    # state machine strip
    sx, sy, sw, sh = 110, 540, W - 220, 96
    s.append(rect(sx, sy, sw, sh, fill=PANEL_ALT))
    s.append(
        vcenter(
            sx,
            sy,
            sw,
            sh,
            "状态机",
            [
                "[新建] → [训练中] → [已保存 checkpoint] → [恢复训练]",
                "[Eval 中] → [Resume 校验] → [架构一致性检查]",
            ],
        )
    )

    write("readme_dataflow_state_machine.svg", "".join(s))


# ================================================================ Figure 3
def fig_paper_overview():
    W, H = 1240, 600
    s = [svg_open(W, H), title("MIM 论文总览", W / 2, 46)]

    margin, gap = 44, 26
    bw = (W - 2 * margin - 3 * gap) / 4
    by, bh = 100, 150

    row1 = [
        ("研究问题", ["高阶非平稳时空预测"]),
        ("理论依据", ["Cramér / ARIMA", "difference-stationary"]),
        ("核心方法", ["MIM-N + MIM-S", "多层堆叠"]),
        ("实验验证", ["4 数据集", "+ 消融实验"]),
    ]
    for i, (head, items) in enumerate(row1):
        x = margin + i * (bw + gap)
        s.append(rect(x, by, bw, bh))
        s.append(vcenter(x, by, bw, bh, head, items))
        if i < len(row1) - 1:
            aY = by + bh / 2
            s.append(arrow(x + bw + 4, aY, x + bw + gap - 4, aY))

    bw2 = (W - 2 * margin - gap) / 2
    by2, bh2 = 360, 150
    row2 = [
        ("主要结论", ["长期预测 SOTA", "差分平稳化有效"]),
        ("关键局限", ["计算成本 / 理论假设", "对比覆盖不足"]),
    ]
    xs2 = [margin, margin + bw2 + gap]
    for (head, items), x in zip(row2, xs2):
        s.append(rect(x, by2, bw2, bh2))
        s.append(vcenter(x, by2, bw2, bh2, head, items))

    # clean vertical arrows from row1 col2 / col3 down into row2
    c2 = margin + (bw + gap) + bw / 2
    c3 = margin + 2 * (bw + gap) + bw / 2
    t2 = xs2[0] + bw2 / 2
    t3 = xs2[1] + bw2 / 2
    s.append(
        f'<path d="M {c2:.1f} {by + bh + 4} V {by2 - 44:.1f} '
        f'H {t2:.1f} V {by2 - 4:.1f}" fill="none" stroke="{ARROW}" '
        f'stroke-width="2" marker-end="url(#arA)"/>'
    )
    s.append(
        f'<path d="M {c3:.1f} {by + bh + 4} V {by2 - 44:.1f} '
        f'H {t3:.1f} V {by2 - 4:.1f}" fill="none" stroke="{ARROW}" '
        f'stroke-width="2" marker-end="url(#arA)"/>'
    )

    write("paper_analysis_overview.svg", "".join(s))


# ================================================================ Figure 4
def fig_logic_chain():
    W, H = 1400, 440
    s = [svg_open(W, H), title("密码表逻辑链与阅读路径", W / 2, 46)]

    chain = [
        ("SPL", "问题"),
        ("CPL", "批评"),
        ("GAP", "空白"),
        ("RAT", "理论"),
        ("ROF", "目标"),
        ("WTD/WTDD", "方法"),
        ("ROFD", "发现"),
        ("RCL/RTC", "定位"),
        ("POC/RFW/RPP", "局限"),
    ]
    margin, gap = 40, 14
    n = len(chain)
    bw = (W - 2 * margin - (n - 1) * gap) / n
    by, bh = 110, 130

    for i, (code, lab) in enumerate(chain):
        x = margin + i * (bw + gap)
        fill = PANEL if i < 7 else PANEL_ALT
        s.append(rect(x, by, bw, bh, fill=fill))
        cx = x + bw / 2
        s.append(text(cx, by + 52, code, size=14, fill=INK, weight="700"))
        s.append(text(cx, by + 82, lab, size=12.5, fill=MUTED))
        if i < n - 1:
            aY = by + bh / 2
            s.append(arrow(x + bw + 2, aY, x + bw + gap - 2, aY))

    sx, sy, sw, sh = 80, 300, W - 160, 96
    s.append(rect(sx, sy, sw, sh, fill=PANEL_ALT))
    s.append(
        vcenter(
            sx,
            sy,
            sw,
            sh,
            "阅读路径建议",
            [
                "精读顺序：SPL → CPL → GAP → RAT → ROF → WTD/WTDD → ROFD "
                "→ RCL/RTC → POC/RFW/RPP"
            ],
        )
    )

    write("paper_analysis_logic_chain.svg", "".join(s))


if __name__ == "__main__":
    fig_architecture()
    fig_dataflow()
    fig_paper_overview()
    fig_logic_chain()
    print("Generated SVG:", sorted(os.listdir(OUT_DIR)))
