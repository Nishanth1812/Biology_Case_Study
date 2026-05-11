# Biology_Case_Study

Simple computational biology demo for identifying fragile regions in a DNA sequence.

## Run

```bash
uv sync
uv run streamlit run app.py
```

## Method

- Sliding window: 100 bp
- Step size: 20 bp
- Metrics: GC content, AT content, AT/TA flexibility, repeat density, Wallace Tm
- Fragility score: weighted combination of normalized AT content, flexibility, and repeat density
