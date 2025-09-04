from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from i18n import get_translations
from api_utils import get_token, search_similar

# ===== i18n =====
def _tr() -> Dict[str, str]:
    lang = st.session_state.get("lang", "ru")
    return get_translations(lang)

# ===== state =====
def _init_state() -> None:
    st.session_state.setdefault("sim_subject", "")
    st.session_state.setdefault("sim_description", "")
    st.session_state.setdefault("sim_limit", 10)
    st.session_state.setdefault("sim_results", [])

# ===== inputs =====
def _preset_section(_: Dict[str, str]) -> Tuple[str, str]:
    subject, description = st.session_state["sim_subject"], st.session_state["sim_description"]
    with st.expander(_["similar.title"], expanded=False):
        try:
            from config import SIMILAR_EXAMPLES  # [{"subject":..., "description":...}, ...]
        except Exception:
            SIMILAR_EXAMPLES = []
        if SIMILAR_EXAMPLES:
            idx = st.selectbox(
                _["similar.select_label"],
                options=range(len(SIMILAR_EXAMPLES)),
                format_func=lambda i: SIMILAR_EXAMPLES[i].get("subject") or f"#{i+1}",
            )
            if st.button(_["similar.use_default"]):
                ex = SIMILAR_EXAMPLES[idx]
                subject, description = ex.get("subject", ""), ex.get("description", "")
    return subject, description

def _input_form(_: Dict[str, str], subject: str, description: str) -> Tuple[str, str, int, bool]:
    st.subheader(_["similar.form.title"])
    subject = st.text_input(_["similar.form.subject"], value=subject, key="sim_subject")
    description = st.text_area(_["similar.form.description"], value=description, key="sim_description", height=140)
    cols = st.columns([3, 1])
    with cols[1]:
        limit = st.number_input(_["similar.form.limit"], min_value=1, max_value=50, value=int(st.session_state["sim_limit"]))
    run = st.button(_["similar.form.search_btn"], type="primary", use_container_width=True)
    st.session_state["sim_limit"] = int(limit)
    return subject, description, int(limit), run

# ===== api =====
def _run_search(api_url: str, username: str | None, password: str | None, limit: int, _: Dict[str, str]) -> List[Dict[str, Any]]:
    if not username or not password:
        st.error(_["auth.failed"]); return []
    with st.spinner(_["similar.spinner.token"] if "similar.spinner.token" in _ else _["classification.spinner.token"]):
        token = get_token(api_url, username, password)
    if not token:
        st.error(_["auth.failed"]); return []
    subject = st.session_state.get("sim_subject") or ""
    description = st.session_state.get("sim_description") or ""
    if not subject.strip() and not description.strip():
        st.warning(_["similar.warn.empty"]); return []
    with st.spinner(_["similar.spinner.search"]):
        resp = search_similar(subject, description, token, api_url, limit=limit)
    items = []
    if isinstance(resp, dict):
        items = resp.get("results") or resp.get("items") or []
    return items if isinstance(items, list) else []

# ===== parsing =====
def _normalize_rows(items: List[Dict[str, Any]], _: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for it in items:
        rid = it.get("id") or it.get("request_id") or it.get("source_id") or ""
        cls = it.get("class_name") or it.get("label") or ""
        score = float(it.get("score", 0.0) or 0.0)
        rows.append({_["similar.table.request_id"]: str(rid), _["similar.table.class"]: str(cls), _["similar.table.score_short"]: score})
    return pd.DataFrame(rows)

def _class_distribution(df: pd.DataFrame, _: Dict[str, str]) -> pd.DataFrame:
    col = _["similar.table.class"]
    cnt = Counter(df[col].tolist())
    return pd.DataFrame({_["similar.plot.class_dist.xlabel"]: list(cnt.keys()), _["similar.plot.class_dist.ylabel"]: list(cnt.values())})

# ===== render =====
def _render_query(_: Dict[str, str], subject: str, description: str) -> None:
    with st.expander(_["similar.query.title"], expanded=True):
        st.write(f"**{_['common.subject']}**: {subject or '—'}")
        st.write(f"**{_['common.description']}**: {description or '—'}")

def _render_results(_: Dict[str, str], df: pd.DataFrame) -> None:
    n = len(df)
    st.success(_["similar.found"].format(n=n))
    if n == 0:
        st.info(_["similar.none"]); return
    st.subheader(_["similar.top3.title"])
    st.dataframe(df.head(3), use_container_width=True)
    st.subheader(_["similar.all.title"])
    st.dataframe(df, use_container_width=True)

def _render_charts(_: Dict[str, str], df: pd.DataFrame) -> None:
    try:
        dist = _class_distribution(df, _)
        fig1, ax1 = plt.subplots(figsize=(7, 3.2))
        ax1.bar(dist.iloc[:, 0], dist.iloc[:, 1])
        ax1.set_title(_["similar.plot.class_dist.title"])
        ax1.set_xlabel(_["similar.plot.class_dist.xlabel"])
        ax1.set_ylabel(_["similar.plot.class_dist.ylabel"])
        ax1.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        st.pyplot(fig1)
    except Exception:
        pass
    try:
        counts = df[_["similar.table.class"]].value_counts()
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        ax2.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
        ax2.set_title(_["similar.plot.class_pie.title"])
        ax2.axis("equal")
        st.pyplot(fig2)
    except Exception:
        pass

# ===== entry =====
def render_similar_docs_tab(api_url: str, username: str | None, password: str | None) -> None:
    _ = _tr(); _init_state()
    st.title(_["similar.title"])
    subj, desc = _preset_section(_)
    subj, desc, limit, run = _input_form(_, subj, desc)
    if run:
        items = _run_search(api_url, username, password, limit, _)
        st.session_state["sim_results"] = items
    _render_query(_, subj, desc)
    df = _normalize_rows(st.session_state.get("sim_results", []), _)
    _render_results(_, df)
    if not df.empty:
        _render_charts(_, df)
