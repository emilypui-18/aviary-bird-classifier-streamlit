"""Aviary — a compact Streamlit interface for the saved EfficientNetB0 bird classifier.

The trained checkpoint, class order and evaluation transformations are unchanged.
The visual preview is cropped separately from the image sent to the model.
"""

from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import time
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from model_utils import build_model, get_transforms


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
MIN_LOADING_SECONDS = 0.9

EXAMPLES = [
    ("Cardinal", "cardinal.jpg"),
    ("Blue Jay", "blue-jay.jpg"),
    ("Cedar Waxwing", "cedar-waxwing.jpg"),
    ("Scarlet Tanager", "scarlet-tanager.jpg"),
]

st.set_page_config(
    page_title="Aviary · Bird identification",
    page_icon="🪶",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner="Loading EfficientNetB0 classifier…")
def load_classifier():
    """Load the original PyTorch checkpoint once per Streamlit server process."""
    config = json.loads((ASSETS / "model_config.json").read_text(encoding="utf-8"))
    if config["model"] != "efficientnet_b0" or len(config["class_names"]) != config["num_classes"]:
        raise ValueError("The model configuration or class mapping is invalid.")

    model, _ = build_model(
        config["model"], config["num_classes"], pretrained=False,
        dropout=config.get("dropout", 0.4),
    )
    weights = torch.load(ASSETS / "final.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    model.eval()
    _, transform = get_transforms(int(config["img_size"]), augment=False)
    return model, transform, config


def read_photo(raw: bytes) -> Image.Image:
    """Read and validate image bytes without modifying the model input pipeline."""
    if not raw:
        raise ValueError("Choose a photograph first.")
    if len(raw) > MAX_BYTES:
        raise ValueError("Your photograph must be smaller than 10 MB.")
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            if opened.format not in {"JPEG", "PNG"}:
                raise ValueError("Only JPG and PNG photographs are supported.")
            if opened.width * opened.height > MAX_PIXELS:
                raise ValueError("This photograph's resolution is too large.")
            return opened.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The photograph could not be opened.") from exc


def classify_photo(raw: bytes) -> dict:
    """Apply the saved evaluation transform and return the real Top-5 scores."""
    image = read_photo(raw)
    model, transform, config = load_classifier()
    began = time.perf_counter()
    batch = transform(image).unsqueeze(0)
    with torch.inference_mode():
        probabilities = torch.softmax(model(batch)[0], dim=0)
        values, indices = torch.topk(probabilities, k=5)
    top5 = [
        {"species": config["class_names"][int(i)], "confidence": float(v)}
        for v, i in zip(values.tolist(), indices.tolist())
    ]
    return {
        "predicted_species": top5[0]["species"],
        "confidence": top5[0]["confidence"],
        "top5": top5,
        "num_classes": config["num_classes"],
        "input_size": config["img_size"],
        "inference_ms": round((time.perf_counter() - began) * 1000, 1),
    }


@st.cache_data(show_spinner=False, max_entries=30)
def photo_preview_uri(raw: bytes) -> str:
    """One identical 4:3 visual preview for ready, loading and result states.

    Preview cropping changes display only, never the bytes sent to EfficientNetB0.
    """
    image = read_photo(raw)
    display = ImageOps.fit(image, (960, 720), method=Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    display.save(buffer, format="JPEG", quality=87, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def initialise_state():
    for key, default in {
        "upload_version": 0,
        "example": None,
        "result": None,
        "last_image_hash": None,
        "prediction_error": "",
        "scroll_to_identify": False,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default


def reset():
    st.session_state.upload_version += 1
    st.session_state.example = None
    st.session_state.result = None
    st.session_state.last_image_hash = None
    st.session_state.prediction_error = ""
    st.session_state.scroll_to_identify = True


def choose_example(filename: str):
    st.session_state.upload_version += 1
    st.session_state.example = filename
    st.session_state.result = None
    st.session_state.last_image_hash = None
    st.session_state.prediction_error = ""
    st.session_state.scroll_to_identify = True


def photo_frame(uri: str, label: str = "Selected bird photograph"):
    st.markdown(
        '<div class="aviary-photo-frame">'
        f'<img src="{uri}" alt="{html.escape(label, quote=True)}" />'
        '</div>',
        unsafe_allow_html=True,
    )


def stepper(current: int):
    parts = []
    for number, name in enumerate(("Photograph", "Identify", "Result"), 1):
        cls = "done" if number < current else "current" if number == current else "waiting"
        character = "✓" if number < current else str(number)
        parts.append(
            f'<div class="aviary-step {cls}">'
            f'<span class="aviary-step-number">{character}</span>'
            f'<span>{name}</span></div>'
        )
        if number < 3:
            parts.append('<span class="aviary-step-divider"></span>')
    st.markdown('<div class="aviary-steps">' + "".join(parts) + '</div>', unsafe_allow_html=True)


def loading_copy():
    st.markdown(
        '<div class="aviary-detail aviary-fade" role="status" aria-live="polite">'
        '<div class="aviary-kicker">IDENTIFYING SPECIES</div>'
        '<div class="aviary-loading-dot" aria-hidden="true"></div>'
        '<h2>Reading the plumage...</h2>'
        '<p>Analysing the photograph and ranking the closest matches.</p>'
        '<div class="aviary-loading-steps"><span>✓ Image received</span>'
        '<span><span class="aviary-pulse"></span> Comparing visual features</span>'
        '<span class="muted">○ Preparing Top-5 results</span></div>'
        '</div>', unsafe_allow_html=True,
    )


def ready_copy(filename: str, size: int) -> bool:
    st.markdown(
        '<div class="aviary-detail aviary-fade">'
        '<div class="aviary-kicker">PHOTOGRAPH READY</div>'
        '<h2>Ready to identify.</h2>'
        '<p>The model will compare your photograph with 200 bird species and show its five closest matches.</p>'
        f'<div class="aviary-file-info">{html.escape(filename)} · {size / (1024 * 1024):.1f} MB</div>'
        '</div>', unsafe_allow_html=True,
    )
    return st.button("Identify this bird  →", key="classify", type="primary", width="stretch")


def result_copy(prediction: dict):
    score = prediction["confidence"]
    name = html.escape(prediction["predicted_species"])
    st.markdown(
        '<div class="aviary-detail aviary-fade">'
        '<div class="aviary-kicker">MOST LIKELY SPECIES</div>'
        f'<h2>{name}</h2>'
        f'<p>Best match among {prediction["num_classes"]} known species.</p>'
        '<div class="aviary-score-head"><span>MODEL SCORE</span>'
        f'<strong>{score:.1%}</strong></div>'
        '<div class="aviary-score-track">'
        f'<div style="width:{max(0.0, min(100.0, score * 100)):.2f}%"></div></div>'
        '<p class="aviary-score-note">Model scores are not guarantees. Similar-looking or unsupported species may be misidentified.</p>'
        '</div>', unsafe_allow_html=True,
    )
    if score < 0.4:
        st.warning(
            "Low-confidence prediction. This bird might not be among the 200 supported species. "
            "This is an advisory, not a validated unknown-species detector."
        )


def result_matches(prediction: dict):
    rows = []
    for rank, item in enumerate(prediction["top5"], 1):
        percent = item["confidence"] * 100
        rows.append(
            f'<div class="aviary-match-row {"leading" if rank == 1 else ""}">'
            f'<span class="aviary-match-rank">{rank:02d}</span>'
            f'<span class="aviary-match-name">{html.escape(item["species"])}</span>'
            '<span class="aviary-match-bar"><span '
            f'style="width:{max(0.0, min(100.0, percent)):.2f}%"></span></span>'
            f'<span class="aviary-match-confidence">{percent:.1f}%</span>'
            '</div>'
        )
    st.markdown(
        '<div class="aviary-topfive aviary-fade">'
        '<div class="aviary-list-heading"><span>TOP-5 PREDICTIONS</span><span>MODEL CONFIDENCE</span></div>'
        + "".join(rows) + '</div>', unsafe_allow_html=True,
    )


# CSS is deliberately separated from inference and comes from a trusted local file.
st.markdown('<style>' + (ROOT / 'assets' / 'styles.css').read_text(encoding='utf-8') + '</style>', unsafe_allow_html=True)
initialise_state()

st.markdown(
    '<div class="aviary-nav"><div class="aviary-brand"><span class="aviary-brand-mark">✳</span>'
    '<span>Aviary</span></div><div class="aviary-navlinks">'
    '<a href="#identify">Identify</a><a href="#the-model">The model</a>'
    '<a href="#species">Species</a></div></div>', unsafe_allow_html=True,
)
st.markdown(
    '<div id="identify" class="aviary-hero">'
    '<div class="aviary-kicker">BIRD IDENTIFICATION · EFFICIENTNETB0</div>'
    '<h1>A name for every <em>feather.</em></h1>'
    '<p>Upload a bird photograph and explore the five species your model considers most likely.</p>'
    '</div>', unsafe_allow_html=True,
)

if st.session_state.scroll_to_identify:
    st.session_state.scroll_to_identify = False
    # Supported by Streamlit >= 1.55. This is static, trusted JS only.
    st.html(
        "<script>setTimeout(() => document.getElementById('identify')?.scrollIntoView("
        "{behavior:'smooth',block:'start'}), 120);</script>",
        unsafe_allow_javascript=True,
    )

# Keep the same container and photo geometry as the page switches between states.
with st.container(border=True, key="identify_card"):
    has_previous = (st.session_state.example is not None or st.session_state.last_image_hash is not None)
    stepper_slot = st.empty()

    with st.expander(
        "Upload or replace a photograph" if has_previous else "Choose your bird photograph",
        expanded=not has_previous,
    ):
        st.markdown(
            '<div class="aviary-upload-hint">JPG or PNG · Maximum 10 MB · Drag and drop or browse files</div>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Upload bird photo", type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key=f"bird_upload_{st.session_state.upload_version}",
            help="Upload an image up to 10 MB.",
        )

    raw = None
    filename = ""
    if uploaded_file is not None:
        raw = uploaded_file.getvalue()
        filename = uploaded_file.name
    elif st.session_state.example:
        filename = st.session_state.example
        raw = (ASSETS / "samples" / filename).read_bytes()

    if raw is not None:
        image_hash = hashlib.sha256(raw).hexdigest()
        if image_hash != st.session_state.last_image_hash:
            st.session_state.last_image_hash = image_hash
            st.session_state.result = None
            st.session_state.prediction_error = ""
        # Validate before emitting a preview or attempting inference.
        try:
            uri = photo_preview_uri(raw)
        except ValueError as exc:
            st.error(str(exc))
            uri = None
    else:
        uri = None

    with stepper_slot.container():
        stepper(3 if st.session_state.result is not None else 2 if uri is not None else 1)

    if uri is None and raw is None:
        st.markdown(
            '<div class="aviary-empty-state">'
            '<span class="aviary-empty-glyph">↟</span>'
            '<h2>Start with a photograph.</h2>'
            '<p>Choose a clear image of a bird, or try one of the examples below.</p>'
            '</div>', unsafe_allow_html=True,
        )
    elif uri is not None:
        left, right = st.columns([1.05, 1], gap="large")
        with left:
            photo_frame(uri)
        with right:
            details = st.empty()
            if st.session_state.result is None:
                with details.container():
                    clicked = ready_copy(filename, len(raw))
                if clicked:
                    details.empty()
                    with details.container():
                        loading_copy()
                    started = time.perf_counter()
                    try:
                        output = classify_photo(raw)
                        remain = MIN_LOADING_SECONDS - (time.perf_counter() - started)
                        if remain > 0:
                            time.sleep(remain)
                        st.session_state.result = output
                        st.session_state.prediction_error = ""
                    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
                        st.session_state.prediction_error = str(exc)
                    st.rerun()
                if st.session_state.prediction_error:
                    st.error(st.session_state.prediction_error)
            else:
                with details.container():
                    result_copy(st.session_state.result)

        if st.session_state.result is not None:
            result_matches(st.session_state.result)
            action, info = st.columns([1, 2.2], gap="medium", vertical_alignment="center")
            with action:
                if st.button("Identify another bird", key="reset_result", type="primary", width="stretch"):
                    reset()
                    st.rerun()
            with info:
                p = st.session_state.result
                st.caption(f"EfficientNetB0 · Inference {p['inference_ms']:.0f} ms · Input {p['input_size']} × {p['input_size']}")
        else:
            if st.button("Start over", key="reset_ready", width="content"):
                reset()
                st.rerun()

st.markdown(
    '<div id="the-model" class="aviary-section-header">'
    '<div class="aviary-kicker">THE MODEL</div><h2>How Aviary works.</h2>'
    '<p>An ImageNet-pretrained EfficientNetB0 network was fine-tuned to classify 200 bird species. '
    'The results below are from the official held-out test set.</p></div>', unsafe_allow_html=True,
)
st.markdown(
    '<div class="aviary-metrics">'
    '<div><span>ARCHITECTURE</span><strong>EfficientNetB0</strong><small>PyTorch · transfer learning</small></div>'
    '<div><span>KNOWN SPECIES</span><strong>200</strong><small>Closed-set classification</small></div>'
    '<div><span>TOP-1 ACCURACY</span><strong>79.57%</strong><small>Official test photographs</small></div>'
    '<div><span>AVERAGE PER CLASS</span><strong>79.03%</strong><small>Mean class-level accuracy</small></div>'
    '<div><span>TOP-5 ACCURACY</span><strong>94.52%</strong><small>Correct species in five guesses</small></div>'
    '<div><span>INPUT</span><strong>224 × 224</strong><small>ImageNet normalisation</small></div>'
    '</div>', unsafe_allow_html=True,
)

st.markdown(
    '<div id="species" class="aviary-section-header">'
    '<div class="aviary-kicker">TRY AN EXAMPLE</div><h2>A few familiar feathers.</h2>'
    '<p>Use one of these photographs to try the classifier without uploading your own image.</p></div>',
    unsafe_allow_html=True,
)
example_cols = st.columns(4, gap="medium")
for col, (name, example_filename) in zip(example_cols, EXAMPLES):
    with col:
        sample_uri = photo_preview_uri((ASSETS / 'samples' / example_filename).read_bytes())
        st.markdown(
            '<div class="aviary-gallery-photo">'
            f'<img src="{sample_uri}" alt="{html.escape(name, quote=True)} photograph" />'
            '</div>', unsafe_allow_html=True,
        )
        if st.button(f"{name}  ↗", key=f"example_{example_filename}", width="stretch"):
            choose_example(example_filename)
            st.rerun()

st.markdown(
    '<div class="aviary-footer">'
    '<span>Aviary · Bird species classification</span>'
    '<span>COS30082 · Computer vision</span></div>',
    unsafe_allow_html=True,
)
