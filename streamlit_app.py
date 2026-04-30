"""
SynAgent — Streamlit UI for Data Quality Audit System (FIXED VERSION)

Run with:
streamlit run streamlit_app.py
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st
from datasets import Dataset

# ── suppress warnings ─────────────────────────
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# ── project root ─────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Streamlit config ─────────────────────────
st.set_page_config(
    page_title="SynAgent — Data Quality Audit",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# SIMPLE LOGGER (FIXED)
# ─────────────────────────────────────────────
def log(msg: str):
    if "log_lines" not in st.session_state:
        st.session_state["log_lines"] = []
    st.session_state["log_lines"].append(str(msg))
    print(msg)


# ─────────────────────────────────────────────
# PIPELINE RUNNER (NO THREADING)
# ─────────────────────────────────────────────
def run_pipeline(dataset, user_hint):
    try:
        from graph import build_graph

        log("🚀 Starting pipeline...")

        app = build_graph()

        initial_state = {
            "dataset": dataset,
            "user_hint": user_hint if user_hint else None,
            "errors": [],
        }

        log("⚙️ Running graph execution...")

        final_state = app.invoke(initial_state)

        log("✅ Pipeline completed successfully")

        return final_state, None

    except Exception as e:
        import traceback
        err = f"{str(e)}\n\n{traceback.format_exc()}"
        log("❌ Pipeline failed")
        log(err)
        return None, err


# ─────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────
def load_csv(file):
    df = pd.read_csv(file)
    return Dataset.from_pandas(df), df


def load_txt(file):
    text = file.read().decode("utf-8", errors="ignore")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    df = pd.DataFrame({"text": lines})
    return Dataset.from_pandas(df), df


def load_docx(file):
    from docx import Document
    doc = Document(file)
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    df = pd.DataFrame({"paragraph": paras})
    return Dataset.from_pandas(df), df


def load_hf(name, split):
    from datasets import load_dataset
    ds = load_dataset(name, split=split)
    df = ds.to_pandas()
    return ds, df


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
for k, v in {
    "running": False,
    "done": False,
    "state": None,
    "dataset": None,
    "df": None,
    "log_lines": [],
    "err": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────
# SIDEBAR UI
# ─────────────────────────────────────────────
st.sidebar.title("🔍 SynAgent")

source = st.sidebar.radio("Data Source", ["Upload", "HuggingFace"])

dataset = None
df = None

if source == "Upload":
    file = st.sidebar.file_uploader("Upload file", type=["csv", "txt", "docx"])

    if file:
        ext = Path(file.name).suffix

        try:
            if ext == ".csv":
                dataset, df = load_csv(file)
            elif ext == ".txt":
                dataset, df = load_txt(file)
            elif ext == ".docx":
                dataset, df = load_docx(file)

            st.session_state["dataset"] = dataset
            st.session_state["df"] = df

            st.sidebar.success(f"Loaded {len(df)} rows")

        except Exception as e:
            st.sidebar.error(str(e))

else:
    snippet = st.sidebar.text_input("HF Dataset (name)")

    if snippet:
        try:
            dataset, df = load_hf(snippet, "train[:100]")
            st.session_state["dataset"] = dataset
            st.session_state["df"] = df

            st.sidebar.success("Loaded HF dataset")

        except Exception as e:
            st.sidebar.error(str(e))


user_ctx = st.sidebar.text_area("User Context")


run_btn = st.sidebar.button(
    "▶ Run Audit",
    disabled=st.session_state["dataset"] is None
)


# ─────────────────────────────────────────────
# RUN PIPELINE
# ─────────────────────────────────────────────
if run_btn:
    st.session_state["running"] = True
    st.session_state["done"] = False
    st.session_state["log_lines"] = []
    st.session_state["err"] = None
    st.session_state["state"] = None

    with st.status("Running pipeline...", expanded=True) as status:

        log("📦 Dataset ready")

        final_state, err = run_pipeline(
            st.session_state["dataset"],
            user_ctx
        )

        if err:
            st.session_state["err"] = err
            status.update(label="❌ Failed", state="error")
        else:
            st.session_state["state"] = final_state
            status.update(label="✅ Done", state="complete")

        st.session_state["running"] = False
        st.session_state["done"] = True


# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────

st.title("🔍 SynAgent Dashboard")

if st.session_state["dataset"] is None:
    st.info("Upload a dataset to begin")
    st.stop()


# dataset preview
df = st.session_state["df"]
st.subheader("📊 Dataset Preview")
st.dataframe(df.head(20))


# running state
if st.session_state["running"]:
    st.info("Running pipeline...")

    st.markdown("### Logs")
    st.code("\n".join(st.session_state["log_lines"]))


# final output
elif st.session_state["done"]:

    if st.session_state["err"]:
        st.error(st.session_state["err"])

    state = st.session_state["state"] or {}

    st.success("Pipeline Complete")

    st.markdown("### 🔥 Final Output")

    st.json({
        "thinker_output": state.get("thinker_output"),
        "evaluator_output": state.get("evaluator_output"),
        "researcher_output": state.get("researcher_output"),
    })


    st.markdown("### 📜 Logs")
    st.code("\n".join(st.session_state["log_lines"]))