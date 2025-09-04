from __future__ import annotations

import uuid
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from i18n import get_translations
from api_utils import get_token, classify_request

# ===== i18n =====
def _tr() -> Dict[str, str]:
    lang = st.session_state.get("lang", "ru")
    return get_translations(lang)

# ===== state =====
def _init_state() -> None:
    st.session_state.setdefault("cls_subject", "")
    st.session_state.setdefault("cls_description", "")
    st.session_state.setdefault("cls_result", None)

# ===== inputs =====
def _preset_section(_: Dict[str, str]) -> Tuple[str, str]:
    subject, description = st.session_state["cls_subject"], st.session_state["cls_description"]
    with st.expander(_["classification.pick_example.title"], expanded=False):
        try:
            from config import CLASSIFICATION_EXAMPLES  # [{"subject":..., "description":...}, ...]
        except Exception:
            CLASSIFICATION_EXAMPLES = []
        if CLASSIFICATION_EXAMPLES:
            idx = st.selectbox(
                _["classification.pick_example.select_label"],
                options=range(len(CLASSIFICATION_EXAMPLES)),
                format_func=lambda i: CLASSIFICATION_EXAMPLES[i].get("subject") or f"#{i+1}",
            )
            if st.button(_["classification.pick_example.use_default"]):
                ex = CLASSIFICATION_EXAMPLES[idx]
                subject, description = ex.get("subject", ""), ex.get("description", "")
    return subject, description

def _input_form(_: Dict[str, str], subject: str, description: str) -> Tuple[str, str, bool]:
    st.subheader(_["classification.form.title"])
    subject = st.text_input(_["classification.form.subject"], value=subject, key="cls_subject")
    description = st.text_area(_["classification.form.description"], value=description, key="cls_description", height=140)
    run = st.button(_["classification.form.run"], type="primary", use_container_width=True)
    return subject, description, run

# ===== api =====
def _run_classification(api_url: str, username: str | None, password: str | None, _: Dict[str, str]) -> Dict[str, Any] | None:
    if not username or not password:
        st.error(_["auth.failed"]); return None
    with st.spinner(_["classification.spinner.token"]):
        token = get_token(api_url, username, password)
    if not token:
        st.error(_["auth.failed"]); return None
    payload_subject = st.session_state.get("cls_subject") or ""
    payload_description = st.session_state.get("cls_description") or ""
    if not payload_subject.strip() and not payload_description.strip():
        st.warning(_["classification.warn.empty"]); return None
    with st.spinner(_["classification.spinner.predict"]):
        return classify_request(payload_subject, payload_description, token, api_url)

# ===== parsing =====
def _extract_predictions(resp: Dict[str, Any]) -> Tuple[str | None, float | None, List[Tuple[str, float]]]:
    preds = resp.get("predictions")
    if preds is None:
        return None, None, []
    if isinstance(preds, dict):
        top = preds.get("class_name"); prob = float(preds.get("probability", 0.0) or 0.0)
        ranked = [(top, prob)] if top else []
        return top, prob, ranked
    if isinstance(preds, list):
        ranked = []
        for p in preds:
            if not isinstance(p, dict): continue
            c = p.get("class_name"); pr = float(p.get("probability", 0.0) or 0.0)
            if c is not None: ranked.append((c, pr))
        ranked.sort(key=lambda x: x[1], reverse=True)
        if ranked: return ranked[0][0], ranked[0][1], ranked
    return None, None, []

# ===== render =====
def _render_result(_: Dict[str, str], top: str | None, prob: float | None, ranked: List[Tuple[str, float]]) -> None:
    st.subheader(_["classification.results.title"])
    if top is None:
        st.error(_["classification.error.no_result"]); return
    st.success(_["classification.success.predicted"])
    c1, c2 = st.columns([2, 1])
    with c1:
        st.metric(_["classification.results.top_class"], top)
    with c2:
        st.metric(_["classification.results.top_proba"], f"{(prob or 0.0)*100:.2f}%")
    if ranked:
        st.caption(_["classification.results.top5.title"])
        df = pd.DataFrame(ranked[:5], columns=[_["classification.plot.x_class"], _["classification.plot.y_prob"]])
        try:
            fig, ax = plt.subplots(figsize=(6, 3.2))
            ax.bar(df.iloc[:, 0], df.iloc[:, 1])
            ax.set_xlabel(_["classification.plot.x_class"])
            ax.set_ylabel(_["classification.plot.y_prob"])
            ax.set_ylim(0, 1)
            for i, v in enumerate(df.iloc[:, 1].tolist()):
                ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception:
            st.info(_["classification.plot.unavailable"])

# ===== entry =====
def render_classification_tab(api_url: str, username: str | None, password: str | None) -> None:
    _ = _tr(); _init_state()
    st.title(_["classification.title"])
    subj, desc = _preset_section(_)
    subj, desc, run = _input_form(_, subj, desc)
    if run:
        resp = _run_classification(api_url, username, password, _)
        if resp:
            top, prob, ranked = _extract_predictions(resp)
            st.session_state["cls_result"] = {"top": top, "prob": prob, "ranked": ranked}
    res = st.session_state.get("cls_result")
    if res:
        _render_result(_, res["top"], res["prob"], res["ranked"])
