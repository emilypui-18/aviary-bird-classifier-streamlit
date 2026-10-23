# Aviary — Streamlit Bird Classifier

A single-process Streamlit application using the **same trained EfficientNetB0 model** and preprocessing as the existing React/FastAPI Aviary website. No retraining and no separate API or Node.js server is required.

## Run in VS Code (Mac)

Open this folder in VS Code and run in its terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Streamlit will normally open at **http://localhost:8501**. Use Python 3.11 or 3.12 if you run into machine-specific PyTorch installation issues.

**Windows**: activate with `.venv\Scripts\activate` instead of `source .venv/bin/activate`.

## What this includes

- `streamlit_app.py`: one-page application with photograph upload, ready/analysing/result states, Top-5 species ranking, low-confidence note and example gallery.
- `model_utils.py`: the identical EfficientNetB0 model construction and evaluation preprocessing from your existing project.
- `assets/final.pt`: your trained checkpoint (unchanged).
- `assets/model_config.json`: exact mapping of the 200 bird classes (unchanged).
- `assets/samples/`: four sample images for demonstration.
- `.streamlit/config.toml`: light UI theme and 10 MB upload limit.

The final reported results (Top-1 79.57%, average per-class 79.03%, Top-5 94.52%) come from the official held-out test evaluation. The demo's predictions are **real model inference**. Top-1 softmax scores are not calibrated probabilities or open-set identification.

## How it differs from the React version

- Only **one command** runs the website (`streamlit run streamlit_app.py`).
- No separate FastAPI or React server is required.
- Visual styling resembles the previous Aviary design, but native Streamlit controls will not be pixel-identical to React.
- The loading state holds for **at least one second** for a visible analysis transition; this does not affect model predictions.

## Before submission

The original assignment brief may specify Gradio/Hugging Face. Confirm that Streamlit is an accepted substitute. For public deployment, the checkpoint must be included in the deployment repository or retrieved from secure hosting, and hosting needs enough memory for PyTorch. Keep the original evaluation and code artifacts with your submission.
