import re
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


WINDOW_SIZE = 100
STEP_SIZE = 20


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
    fragility = (0.30 * (at / 100.0)) + (0.35 * flex) + (0.35 * repeat) + 0.15 * (1 - (tm / 400))
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
    if df.empty:
        return ""

    count = len(df)
    label_every = max(1, count // 12)
    tile_width = 16 if count > 120 else 22
    tiles = []
    labels = []

    for idx, row in enumerate(df.iterrows()):
        _, row = row
        color = score_color(row["Fragility Score"])
        start = int(row["Start"])
        end = int(row["End"])
        coord = f"{start}-{end}"
        display_label = str(start) if idx % label_every == 0 or idx == count - 1 else ""
        tiles.append(
            f'<div title="{coord}" style="width:{tile_width}px;height:22px;flex:0 0 {tile_width}px;'
            f'border-radius:6px;background:{color};box-shadow:inset 0 0 0 1px rgba(255,255,255,0.08);"></div>'
        )
        labels.append(
            f'<div title="{coord}" style="width:{tile_width}px;flex:0 0 {tile_width}px;text-align:center;'
            f'font-size:10px;line-height:1.1;color:rgba(255,255,255,0.68);min-height:12px;">{display_label}</div>'
        )

    return (
        '<div style="overflow-x:auto;padding-bottom:0.1rem;">'
        f'<div style="display:flex;flex-direction:column;gap:0.35rem;min-width:max-content;">'
        f'<div style="display:flex;gap:4px;align-items:center;">{"".join(tiles)}</div>'
        f'<div style="display:flex;gap:4px;align-items:flex-start;">{"".join(labels)}</div>'
        '</div></div>'
    )


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
    starts = df["Start"]
    scores = df["Fragility Score"]
    large_sequence = len(df) > 350
    marker_size = 3.5 if large_sequence else 6.5
    line_width = 1.4 if large_sequence else 2.2

    fig = go.Figure(
        go.Scattergl(
            x=starts,
            y=scores,
            mode="lines+markers",
            line=dict(color="#60a5fa", width=line_width),
            marker=dict(
                color=[score_color(v) for v in scores],
                size=marker_size,
                opacity=0.9,
                line=dict(color="rgba(255,255,255,0.20)", width=0.4),
            ),
            hovertemplate="Start %{x}<br>Fragility %{y:.4f}<extra></extra>",
            name="Fragility Score",
        )
    )

    fig.update_layout(
        title=dict(text="Fragility Score vs. Sequence Position", x=0.02, xanchor="left"),
        template="plotly_dark",
        height=470,
        margin=dict(l=30, r=30, t=60, b=30),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0.01, bgcolor="rgba(0,0,0,0)", title_text=""),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(color="#e5e7eb"),
    )
    fig.update_xaxes(
        title="Window Start Position",
        tickformat="~s" if len(df) > 100 else None,
        nticks=8 if large_sequence else 10,
        gridcolor="rgba(148, 163, 184, 0.16)",
        zeroline=False,
        showline=False,
        showspikes=False,
    )
    fig.update_yaxes(
        title="Fragility Score",
        range=[0, max(1.0, df["Fragility Score"].max() + 0.1)],
        gridcolor="rgba(148, 163, 184, 0.16)",
        zeroline=False,
        showline=False,
        showspikes=False,
    )
    return fig


st.set_page_config(page_title="Fragile Region Demo", page_icon="🧬", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1280px; }
    .panel-title {
        font-size: 1.02rem;
        font-weight: 650;
        margin-bottom: 0.35rem;
        letter-spacing: 0.01em;
    }
    .panel-copy {
        color: rgba(255, 255, 255, 0.70);
        line-height: 1.5;
        margin-bottom: 1rem;
    }
    .small-note {
        color: rgba(255, 255, 255, 0.68);
        font-size: 0.92rem;
        margin-top: 0.7rem;
    }
    .empty-state {
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        background: rgba(255, 255, 255, 0.03);
        margin-top: 1rem;
    }
    .empty-state-title {
        font-size: 1rem;
        font-weight: 650;
        margin-bottom: 0.35rem;
    }
    .empty-state-copy {
        color: rgba(255, 255, 255, 0.72);
        line-height: 1.5;
    }
    div[data-testid="stFileUploader"] section {
        border-radius: 14px;
    }
    div[data-testid="stTextArea"] textarea {
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Identifying Fragile Regions in the Human Genome")
st.caption("Upload or paste a DNA sequence, and the demo will flag windows that look unusually fragile.")

input_col, info_col = st.columns([1.15, 0.85], gap="large")

with input_col:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Sequence Input</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-copy">Upload a FASTA/FNA/TXT file or paste a raw DNA sequence. FASTA headers are removed automatically and non-ACGT characters are ignored.</div>',
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Choose a FASTA, FNA, or TXT file",
            type=["fasta", "fa", "fna", "txt"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            st.session_state.sequence_text = uploaded_file.read().decode("utf-8")

        if "sequence_text" not in st.session_state:
            st.session_state.sequence_text = ""

        sequence_text = st.text_area(
            "DNA sequence",
            key="sequence_text",
            height=240,
            label_visibility="collapsed",
            placeholder=">example\nATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
        )

        st.markdown(
            f'<div class="small-note">Window size: {WINDOW_SIZE} bp &nbsp;•&nbsp; Step size: {STEP_SIZE} bp &nbsp;•&nbsp; Sequence length: {len(parse_sequence(sequence_text))} bp</div>',
            unsafe_allow_html=True,
        )

with info_col:
    with st.container(border=True):
        st.markdown('<div class="panel-title">What the demo measures</div>', unsafe_allow_html=True)
        st.markdown(
            """
            - **GC / AT balance**
            - **AT/TA flexibility**
            - **Tandem repeat density**
            - **Wallace melting temperature**

            Windows with higher combined scores are treated as more fragile.
            """
        )

seq = parse_sequence(sequence_text)

if not seq:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-state-title">No sequence loaded yet</div>
            <div class="empty-state-copy">Upload a file or paste DNA above to generate the fragility plot and summary tables.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

if len(seq) < WINDOW_SIZE:
    st.error(f"Sequence must be at least {WINDOW_SIZE} bp long.")
    st.stop()

results = analyze_sequence(seq)

if results.empty:
    st.warning("No windows could be analyzed. Sequence may be too short.")
    st.stop()

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
    selected_window = st.selectbox(
        "Choose a window to inspect",
        results.index.tolist(),
        index=int(results["Fragility Score"].idxmax()),
        format_func=lambda i: f'{int(results.loc[i, "Start"])}-{int(results.loc[i, "End"])} | score {results.loc[i, "Fragility Score"]:.4f}', #type: ignore
    )
    current_row = results.loc[selected_window]

    with st.container(border=True):
        st.markdown(f'**Position:** {int(current_row["Start"])}-{int(current_row["End"])}')
        c1, c2, c3 = st.columns(3)
        c1.metric("GC Content", f'{current_row["GC Content (%)"]:.2f}%')
        c2.metric("AT Content", f'{current_row["AT Content (%)"]:.2f}%')
        c3.metric("Flexibility Score", f'{current_row["Flexibility Score"]:.4f}')
        c1, c2, c3 = st.columns(3)
        c1.metric("Repeat Density", f'{current_row["Repeat Density"]:.4f}')
        c2.metric("Tm", int(current_row["Tm"]))
        c3.metric("Fragility Score", f'{current_row["Fragility Score"]:.4f}')

    st.dataframe(
        results[["Start", "End", "GC Content (%)", "AT Content (%)", "Flexibility Score", "Repeat Density", "Tm", "Fragility Score"]]
        .sort_values("Fragility Score", ascending=False)
        .head(8),
        use_container_width=True,
        hide_index=True,
    )

st.subheader("Fragile Window Map")
st.markdown(
    "<div style='padding:0.25rem 0 0.6rem 0; line-height: 1.8;'>" + render_color_legend() + "</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div style='padding:0 0 0.5rem 0; overflow-x:auto;'>" + render_sequence_map(results) + "</div>",
    unsafe_allow_html=True,
)

st.subheader("All Window Scores")
display_df = results.copy()
display_df["Fragility Class"] = display_df["Fragility Score"].map(fragility_band)
st.dataframe(display_df, use_container_width=True, hide_index=True)

st.caption("Red = high fragility, orange = moderate fragility, blue = low fragility.")
