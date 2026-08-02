import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from functionforDownloadButtons import download_button


# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------

st.set_page_config(
    layout="centered",
    page_title="POS Correction",
    page_icon="✍️",
)

CSS_PATH = Path("./main.css")
if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("✍️ POS Correction")
st.write("")


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

REQUIRED_COLUMNS = {"SID", "TID", "TOKEN", "POS1", "POS2"}
EDITOR_COLUMNS = ("TID", "TOKEN", "POS1", "X", "POS2", "Y", "GOLDPOS")
DISABLED_COLUMNS = ["TID", "TOKEN", "POS1", "POS2"]


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def validate_input_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        st.error(
            "Uploaded TSV is missing required columns: "
            + ", ".join(sorted(missing))
        )
        st.stop()


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize uploaded dataframe for the app.

    - ensures that the required columns exist
    - ensures that GOLDPOS exists
    - keeps GOLDPOS as object dtype (not categorical)
    """
    df = df.copy()
    validate_input_columns(df)

    if "GOLDPOS" not in df.columns:
        df.insert(0, "GOLDPOS", None)
    else:
        df["GOLDPOS"] = df["GOLDPOS"].where(df["GOLDPOS"].notna(), None)

    df["GOLDPOS"] = df["GOLDPOS"].astype(object)
    return df


def build_file_key(uploaded_file) -> str:
    return f"{uploaded_file.name}_{uploaded_file.size}"


def load_uploaded_dataframe(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file, sep="\t", header=0)
    return prepare_dataframe(df)


def get_sentence_ids(df: pd.DataFrame) -> list:
    sids = df["SID"].dropna().unique().tolist()
    try:
        return sorted(sids)
    except TypeError:
        # fallback if SID contains mixed types that can't be sorted directly
        return sorted(sids, key=lambda x: str(x))


def add_editor_columns(sentence_df: pd.DataFrame) -> pd.DataFrame:
    sentence_df = sentence_df.copy()
    sentence_df["X"] = False
    sentence_df["Y"] = False
    return sentence_df


def fill_default_goldpos(sentence_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill GOLDPOS only where it is currently empty:
    - "_" if POS1 == POS2
    - "MISMATCH" if POS1 != POS2
    """
    sentence_df = sentence_df.copy()

    empty_gold = sentence_df["GOLDPOS"].isna() | (sentence_df["GOLDPOS"] == "")
    same_pos = sentence_df["POS1"] == sentence_df["POS2"]
    diff_pos = sentence_df["POS1"] != sentence_df["POS2"]

    sentence_df.loc[empty_gold & same_pos, "GOLDPOS"] = "_"
    sentence_df.loc[empty_gold & diff_pos, "GOLDPOS"] = "MISMATCH"

    return sentence_df


def prepare_sentence_for_editor(df: pd.DataFrame, sid) -> pd.DataFrame:
    sentence_df = df[df["SID"] == sid].copy()
    sentence_df = fill_default_goldpos(sentence_df)
    sentence_df = add_editor_columns(sentence_df)
    return sentence_df


def apply_checkbox_shortcuts(edited_df: pd.DataFrame) -> pd.DataFrame:
    """
    If X is checked, copy POS1 -> GOLDPOS.
    If Y is checked, copy POS2 -> GOLDPOS.
    Then reset the checkboxes.
    """
    edited_df = edited_df.copy()

    selected_x = edited_df["X"].fillna(False)
    if selected_x.any():
        edited_df.loc[selected_x, "GOLDPOS"] = edited_df.loc[selected_x, "POS1"]

    selected_y = edited_df["Y"].fillna(False)
    if selected_y.any():
        edited_df.loc[selected_y, "GOLDPOS"] = edited_df.loc[selected_y, "POS2"]

    edited_df["X"] = False
    edited_df["Y"] = False

    return edited_df


def persist_sentence_edits(master_df: pd.DataFrame, edited_sentence_df: pd.DataFrame) -> pd.DataFrame:
    """
    Write edited GOLDPOS values back to the full dataframe using SID + TID.
    """
    master_df = master_df.copy()
    edited_sentence_df = edited_sentence_df.copy()

    update_cols = ["SID", "TID", "GOLDPOS"]
    updates = edited_sentence_df[update_cols].drop_duplicates(subset=["SID", "TID"])

    # Use a multi-index update for clean alignment
    master_indexed = master_df.set_index(["SID", "TID"])
    updates_indexed = updates.set_index(["SID", "TID"])

    master_indexed.loc[updates_indexed.index, "GOLDPOS"] = updates_indexed["GOLDPOS"]
    master_df = master_indexed.reset_index()

    # Preserve original column order as much as possible
    original_columns = list(master_df.columns)
    if "GOLDPOS" in original_columns:
        cols = ["GOLDPOS"] + [c for c in original_columns if c != "GOLDPOS"]
        master_df = master_df[cols]

    return master_df


def init_app_state(uploaded_file) -> None:
    """
    Initialise session state when a new file is uploaded.
    """
    df = load_uploaded_dataframe(uploaded_file)
    sid_list = get_sentence_ids(df)

    if not sid_list:
        st.error("No valid sentence IDs found in the SID column.")
        st.stop()

    st.session_state["df"] = df
    st.session_state["sid_list"] = sid_list
    st.session_state["sid_index"] = 0
    st.session_state["uploaded_file_key"] = build_file_key(uploaded_file)


def ensure_file_loaded(uploaded_file) -> None:
    """
    Load or reload data if the uploaded file is new.
    """
    file_key = build_file_key(uploaded_file)

    if (
        "uploaded_file_key" not in st.session_state
        or st.session_state["uploaded_file_key"] != file_key
    ):
        init_app_state(uploaded_file)


def current_sid():
    return st.session_state["sid_list"][st.session_state["sid_index"]]


def set_sid_index(index: int) -> None:
    sid_count = len(st.session_state["sid_list"])
    st.session_state["sid_index"] = max(0, min(index, sid_count - 1))


def go_previous() -> None:
    set_sid_index(st.session_state["sid_index"] - 1)


def go_next() -> None:
    set_sid_index(st.session_state["sid_index"] + 1)


def go_to_sid(selected_sid) -> None:
    sid_list = st.session_state["sid_list"]
    st.session_state["sid_index"] = sid_list.index(selected_sid)


def make_output_filename(uploaded_filename: str) -> str:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if uploaded_filename.endswith(".tsv"):
        return uploaded_filename.replace(".tsv", f"_{timestamp}.tsv")
    return f"{uploaded_filename}_{timestamp}.tsv"


# -----------------------------------------------------------------------------
# Sidebar: upload
# -----------------------------------------------------------------------------

st.sidebar.header("File upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload TSV",
    type=["tsv"],
    label_visibility="collapsed",
    help="Upload a TSV file containing SID, TID, TOKEN, POS1, and POS2.",
)

if uploaded_file is None:
    st.info("👆 Please upload a .tsv file first.")
    st.stop()

ensure_file_loaded(uploaded_file)


# -----------------------------------------------------------------------------
# Main data/state
# -----------------------------------------------------------------------------

df = st.session_state["df"]
sid_list = st.session_state["sid_list"]
sid_index = st.session_state["sid_index"]
sid = current_sid()


# -----------------------------------------------------------------------------
# Navigation
# -----------------------------------------------------------------------------

nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])

with nav_col1:
    st.button(
        "⬅ Previous",
        on_click=go_previous,
        disabled=(sid_index == 0),
        use_container_width=True,
    )

with nav_col2:
    st.info(f"Sentence {sid_index + 1}/{len(sid_list)}")

with nav_col3:
    st.button(
        "Next ➡",
        on_click=go_next,
        disabled=(sid_index == len(sid_list) - 1),
        use_container_width=True,
    )

selected_sid = st.selectbox(
    "Jump to sentence",
    options=sid_list,
    index=sid_index,
)

if selected_sid != sid:
    go_to_sid(selected_sid)
    st.rerun()


# -----------------------------------------------------------------------------
# Sentence editor
# -----------------------------------------------------------------------------

sentence_df = prepare_sentence_for_editor(df, sid)

edited_df = st.data_editor(
    sentence_df,
    hide_index=True,
    column_order=EDITOR_COLUMNS,
    disabled=DISABLED_COLUMNS,
    use_container_width=True,
    key=f"editor_sid_{sid}",
)

edited_df = apply_checkbox_shortcuts(edited_df)

# Compare only persistent columns, not temporary checkbox columns
persistent_cols = [c for c in sentence_df.columns if c not in ("X", "Y")]
original_compare = sentence_df[persistent_cols].reset_index(drop=True)
edited_compare = edited_df[persistent_cols].reset_index(drop=True)

if not edited_compare.equals(original_compare):
    st.session_state["df"] = persist_sentence_edits(st.session_state["df"], edited_df)
    st.rerun()


# -----------------------------------------------------------------------------
# Download
# -----------------------------------------------------------------------------

st.write("")
outfile = make_output_filename(uploaded_file.name)

download_button(
    st.session_state["df"],
    outfile,
    "⬇️ Save",
)
