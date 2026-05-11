import re
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


WINDOW_SIZE = 100
STEP_SIZE = 20


SAMPLE_FASTA = ">Fragile region demo sample\n" \
    "ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT\n" \
    "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC\n" \
    "CCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCG\n" \
    "ATGCGTATATATGCGTCCGCCGATATATATATCCGCGGCGGATATATATATATATATATATATATAT"


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
        return "#c0392b"
    if score >= 0.45:
        return "#e67e22"
    return "#2c7fb8"


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


st.set_page_config(page_title="Fragile Region Demo", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .metric-box {
        background: #f7f8fa;
        border: 1px solid #d9dde3;
        border-radius: 10px;
        padding: 0.9rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Identifying Fragile Regions in the Human Genome")
st.caption("Sliding-window computational biology demo based on sequence properties from the presentation slides.")

left, right = st.columns([1, 1])

with left:
    st.subheader("Input Sequence")
    mode = st.radio("Choose input source", ["Paste DNA sequence", "Load sample FASTA"], horizontal=True)
    if mode == "Load sample FASTA":
        sequence_text = st.text_area("DNA sequence", value=SAMPLE_FASTA, height=220)
    else:
        sequence_text = st.text_area(
            "DNA sequence",
            value="ATATATATATGCGCGCGCATATATATATATCCGCCGCCGATATATATATATATGCGTATATATATAT",
            height=220,
            help="Paste a DNA sequence or FASTA text. Non-ACGT characters are ignored.",
        )

    st.info(f"Window size = {WINDOW_SIZE} bp, step size = {STEP_SIZE} bp")

seq = parse_sequence(sequence_text)

with right:
    st.subheader("Method")
    st.markdown(
        """
        1. Scan the sequence with a 100 bp sliding window.
        2. Compute GC content, AT content, AT/TA flexibility, repeat density, and Wallace Tm.
        3. Normalize values and combine them into a weighted fragility score.
        4. Highlight windows with the highest scores as potentially fragile regions.
        """
    )

if not seq:
    st.warning("Enter a valid DNA sequence to run the analysis.")
    st.stop()

if len(seq) < WINDOW_SIZE:
    st.error(f"Sequence must be at least {WINDOW_SIZE} bp long.")
    st.stop()

results = analyze_sequence(seq)

top_row = results.loc[results["Fragility Score"].idxmax()]

st.subheader("Window Summary")
summary_cols = st.columns(5)
summary_cols[0].metric("Sequence Length", len(seq))
summary_cols[1].metric("Windows Analysed", len(results))
summary_cols[2].metric("Highest Fragility", f'{top_row["Fragility Score"]:.4f}')
summary_cols[3].metric("Top Window", f'{int(top_row["Start"])}-{int(top_row["End"])}')
summary_cols[4].metric("Classification", fragility_band(float(top_row["Fragility Score"])))

plot_col, table_col = st.columns([1.1, 0.9])

with plot_col:
    st.plotly_chart(fragility_figure(results), use_container_width=True)

with table_col:
    st.subheader("Current Window")
    window_options = results.index.tolist()
    selected_window = st.selectbox(
        "Choose a window to inspect",
        window_options,
        index=int(results["Fragility Score"].idxmax()),
        format_func=lambda i: f'{int(results.loc[i, "Start"])}-{int(results.loc[i, "End"])} | score {results.loc[i, "Fragility Score"]:.4f}',
    )
    current_row = results.loc[selected_window]
    st.caption("Selected window details")
    detail_cols = st.columns(2)
    detail_cols[0].metric("Position", f'{int(current_row["Start"])}-{int(current_row["End"])}')
    detail_cols[1].metric("Fragility Score", f'{current_row["Fragility Score"]:.4f}')

    with st.container(border=True):
        mini_cols = st.columns(2)
        mini_cols[0].metric("GC Content", f'{current_row["GC Content (%)"]:.2f}%')
        mini_cols[1].metric("AT Content", f'{current_row["AT Content (%)"]:.2f}%')
        mini_cols[0].metric("Flexibility Score", f'{current_row["Flexibility Score"]:.4f}')
        mini_cols[1].metric("Repeat Density", f'{current_row["Repeat Density"]:.4f}')
        mini_cols[0].metric("Tm", int(current_row["Tm"]))
        mini_cols[1].metric("Class", fragility_band(float(current_row["Fragility Score"])))

    st.dataframe(
        results[["Start", "End", "GC Content (%)", "AT Content (%)", "Flexibility Score", "Repeat Density", "Tm", "Fragility Score"]]
        .sort_values("Fragility Score", ascending=False)
        .head(8),
        use_container_width=True,
        hide_index=True,
    )

st.subheader("Fragile Window Map")
st.markdown(
    "<div style='padding:0.5rem 0 1rem 0; line-height: 1.8;'>" + render_sequence_map(results) + "</div>",
    unsafe_allow_html=True,
)

st.subheader("All Window Scores")
display_df = results.copy()
display_df["Fragility Class"] = display_df["Fragility Score"].map(fragility_band)
st.dataframe(display_df, use_container_width=True, hide_index=True)

st.caption("High-risk windows are shown in red, moderate in orange, and lower-scoring windows in blue.")
