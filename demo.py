import re
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go


WINDOW_SIZE = 100
STEP_SIZE = 20


SAMPLE_FASTA = ">Fragile region demo sample\n" \
    "ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT\n" \
    "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC\n" \
    "CCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCG\n" \
    "ATGCGTATATATGCGTCCGCCGATATATATATCCGCGGCGGATATATATATATATATATATATATAT\n" \
    "ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT"


@dataclass
class WindowResult:
    start: int
    end: int
    sequence: str
    gc_content: float
    at_content: float
    flexibility_score: float
    repeat_density: float
    tm: int
    fragility_score: float


def parse_sequence(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    seq = "".join(line for line in lines if not line.startswith(">"))
    seq = re.sub(r"[^ACGTacgt]", "", seq).upper()
    return seq


def gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    return ((seq.count("G") + seq.count("C")) / len(seq)) * 100.0


def flexibility_score(seq: str) -> float:
    if len(seq) < 2:
        return 0.0
    pairs = len(seq) - 1
    flexible = 0
    for i in range(pairs):
        dinuc = seq[i : i + 2]
        if dinuc in {"AT", "TA"}:
            flexible += 1
    return flexible / pairs if pairs else 0.0


def _mark_tandem_repeat_bases(seq: str, motif_len: int, min_repeats: int) -> set[int]:
    marked = set()
    i = 0
    n = len(seq)
    while i <= n - motif_len * min_repeats:
        motif = seq[i : i + motif_len]
        repeats = 1
        j = i + motif_len
        while j + motif_len <= n and seq[j : j + motif_len] == motif:
            repeats += 1
            j += motif_len
        if repeats >= min_repeats:
            marked.update(range(i, j))
            i = j
        else:
            i += 1
    return marked


def repeat_density(seq: str) -> float:
    if not seq:
        return 0.0

    repeated_positions: set[int] = set()

    for match in re.finditer(r"([ACGT])\1{3,}", seq):
        repeated_positions.update(range(match.start(), match.end()))

    repeated_positions |= _mark_tandem_repeat_bases(seq, motif_len=2, min_repeats=3)
    repeated_positions |= _mark_tandem_repeat_bases(seq, motif_len=3, min_repeats=3)

    repeated_bases = len(repeated_positions)
    return repeated_bases / len(seq)


def melting_temperature(seq: str) -> int:
    a = seq.count("A")
    t = seq.count("T")
    g = seq.count("G")
    c = seq.count("C")
    return 2 * (a + t) + 4 * (g + c)


def analyze_window(seq: str, start: int, window_size: int) -> WindowResult:
    window = seq[start : start + window_size]
    gc = gc_content(window)
    at = 100.0 - gc
    flex = flexibility_score(window)
    repeat = repeat_density(window)
    tm = melting_temperature(window)
    fragility = (0.30 * (at / 100.0)) + (0.35 * flex) + (0.35 * repeat)
    return WindowResult(
        start=start + 1,
        end=start + len(window),
        sequence=window,
        gc_content=gc,
        at_content=at,
        flexibility_score=flex,
        repeat_density=repeat,
        tm=tm,
        fragility_score=fragility,
    )


def analyze_sequence(seq: str) -> pd.DataFrame:
    rows = []
    if len(seq) < WINDOW_SIZE:
        return pd.DataFrame(rows)

    for start in range(0, len(seq) - WINDOW_SIZE + 1, STEP_SIZE):
        result = analyze_window(seq, start, WINDOW_SIZE)
        rows.append(
            {
                "Start": result.start,
                "End": result.end,
                "Window": result.sequence,
                "GC Content (%)": round(result.gc_content, 2),
                "AT Content (%)": round(result.at_content, 2),
                "Flexibility Score": round(result.flexibility_score, 4),
                "Repeat Density": round(result.repeat_density, 4),
                "Tm": result.tm,
                "Fragility Score": round(result.fragility_score, 4),
            }
        )
    return pd.DataFrame(rows)


def fragility_band(score: float) -> str:
    if score >= 0.7:
        return "High"
    if score >= 0.45:
        return "Moderate"
    return "Low"


def score_color(score: float) -> str:
    if score >= 0.7:
        return "#dc2626"
    if score >= 0.45:
        return "#f59e0b"
    return "#10b981"


def render_sequence_map(df: pd.DataFrame) -> str:
    segments = []
    for _, row in df.iterrows():
        color = score_color(row["Fragility Score"])
        label = f'{int(row["Start"])}-{int(row["End"])}'
        segments.append(
            f'<span style="display:inline-block;padding:6px 8px;margin:2px 4px 2px 0;'
            f'border-radius:6px;background:{color};color:white;font-size:12px;line-height:1.2;">{label}</span>'
        )
    return "".join(segments)


def render_color_legend() -> str:
    items = [
        ("#dc2626", "High fragility"),
        ("#f59e0b", "Moderate fragility"),
        ("#10b981", "Low fragility"),
    ]
    return "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:14px;">'
        f'<span style="width:12px;height:12px;border-radius:3px;background:{color};display:inline-block;margin-right:6px;"></span>'
        f'{label}</span>'
        for color, label in items
    )


def fragility_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Start"],
            y=df["Fragility Score"],
            mode="lines+markers",
            line=dict(color="#34495e", width=2),
            marker=dict(
                color=[score_color(v) for v in df["Fragility Score"]],
                size=9,
            ),
            name="Fragility Score",
        )
    )
    fig.update_layout(
        title="Fragility Score vs. Sequence Position",
        xaxis_title="Window Start Position",
        yaxis_title="Fragility Score",
        template="plotly_white",
        height=420,
        margin=dict(l=30, r=30, t=60, b=30),
        yaxis=dict(range=[0, max(1.0, df["Fragility Score"].max() + 0.1)]),
    )
    return fig
