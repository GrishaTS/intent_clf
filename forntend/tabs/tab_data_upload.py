# tabs/tab_data_upload.py
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split

from api_utils import (
    get_token,
    clear_index,
    upload_data,
    compute_metrics_backend,
    get_last_metrics,
    get_classification_report_df,
    plot_confusion_matrix,
    plot_classification_metrics,
    filter_high_quality_classes,
)
from i18n import get_translations


# ================================
#   i18n / helpers
# ================================
def _tr() -> dict:
    lang = st.session_state.get("lang", "ru")
    return get_translations(lang)


def _bar(series: pd.Series, title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def _read_uploaded(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name
    ext = name.split(".")[-1].lower()
    if ext == "csv":
        return pd.read_csv(uploaded_file)
    if ext in ("xlsx", "xls"):
        engine = "openpyxl" if ext == "xlsx" else "xlrd"
        return pd.read_excel(uploaded_file, engine=engine)
    raise ValueError(f"Unsupported format: {ext}")


def _save_local_copy(uploaded_file, data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())


def _select_columns_ui(df: pd.DataFrame, _: dict) -> tuple[str, str | None, str, str]:
    st.subheader(_["data_upload.columns.title"])
    cols = df.columns.tolist()
    none_label = _["data_upload.columns.none"]

    c0, c1, c2 = st.columns(3)
    with c0:
        id_col = st.selectbox(
            _["data_upload.columns.id"],
            options=cols,
            index=cols.index("id") if "id" in cols else 0,
        )
    with c1:
        subject_options = [none_label] + cols
        subj_idx = subject_options.index("subject") if "subject" in cols else 0
        subject_choice = st.selectbox(
            _["data_upload.columns.subject_opt"],
            options=subject_options,
            index=subj_idx,
        )
        subject_col = None if subject_choice == none_label else subject_choice
    with c2:
        description_col = st.selectbox(
            _["data_upload.columns.description"],
            options=cols,
            index=cols.index("description") if "description" in cols else 0,
        )

    target_col = st.selectbox(
        _["data_upload.columns.target"],
        options=cols,
        index=cols.index("class") if "class" in cols else 0,
    )
    return id_col, subject_col, description_col, target_col


def _target_stats_ui(df: pd.DataFrame, target_col: str, _: dict) -> None:
    with st.expander(_["data_upload.target_stats.title"], expanded=False):
        vc = df[target_col].value_counts()
        st.write(_["data_upload.target_stats.caption"])
        stats_df = pd.DataFrame(
            {
                _["data_upload.common.value"]: vc.index,
                _["data_upload.common.count"]: vc.values,
                _["data_upload.common.percent"]: (vc.values / vc.sum() * 100).round(2),
            }
        )
        st.dataframe(stats_df)

        top_n = min(20, len(vc))
        fig = _bar(
            vc.head(top_n),
            _["data_upload.target_stats.chart_title"].format(top_n=top_n),
            _["data_upload.common.value"],
            _["data_upload.common.count"],
        )
        st.pyplot(fig)


def _normalize_df(
    df: pd.DataFrame,
    id_col: str,
    subject_col: str | None,
    description_col: str,
    target_col: str,
) -> pd.DataFrame:
    base = df[[id_col, description_col, target_col]].rename(
        columns={id_col: "id", description_col: "description", target_col: "class"}
    )
    base["subject"] = df[subject_col] if subject_col else "no_subject"
    return base[["id", "subject", "description", "class"]]


def _filter_ui(df_proc: pd.DataFrame, api_url: str, username: str, password: str, _: dict) -> pd.DataFrame:
    st.subheader(_["data_upload.filter.title"])
    choice = st.radio(
        _["data_upload.filter.method_label"],
        [_["data_upload.filter.by_freq"], _["data_upload.filter.by_quality"]],
    )

    if choice == _["data_upload.filter.by_freq"]:
        top_n_values = st.slider(
            _["data_upload.filter.by_freq_slider"],
            min_value=1,
            max_value=min(100, df_proc["class"].nunique()),
            value=10,
        )
        top_values = df_proc["class"].value_counts().head(top_n_values).index.tolist()
        df_proc["class"] = df_proc["class"].apply(
            lambda x: x if x in top_values else _("data_upload.common.others")
        )
        return df_proc

    min_samples = st.slider(
        _["data_upload.filter.min_samples"],
        min_value=1,
        max_value=100,
        value=10,
    )
    min_f1_score = st.slider(
        _["data_upload.filter.min_f1"],
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )
    if st.checkbox(_["data_upload.filter.confirm_params"], value=False):
        df_high_quality, _meta = filter_high_quality_classes(
            df_proc,
            min_samples=min_samples,
            min_f1_score=min_f1_score,
            api_url=api_url,
            username=username,
            password=password,
        )
        if df_high_quality is not None:
            return df_high_quality
    return df_proc


def _post_filter_stats_ui(df_proc: pd.DataFrame, _: dict) -> None:
    with st.expander(_["data_upload.after_filter_stats.title"], expanded=False):
        vc = df_proc["class"].value_counts()
        stats_df = pd.DataFrame(
            {
                _["data_upload.common.value"]: vc.index,
                _["data_upload.common.count"]: vc.values,
                _["data_upload.common.percent"]: (vc.values / vc.sum() * 100).round(2),
            }
        )
        st.dataframe(stats_df)
        st.pyplot(
            _bar(
                vc,
                _["data_upload.after_filter_stats.chart_title"],
                _["data_upload.common.value"],
                _["data_upload.common.count"],
            )
        )
    with st.expander(_["data_upload.after_filter_data.title"], expanded=False):
        st.dataframe(df_proc)


def _run_backend_pipeline(
    df_proc: pd.DataFrame,
    clear_flag: bool,
    api_url: str,
    username: str,
    password: str,
    _: dict,
) -> dict | None:
    token = get_token(api_url, username, password)
    if not token:
        st.error(_["auth.failed"])
        return None

    train_df, test_df = train_test_split(
        df_proc, test_size=0.1, random_state=42, stratify=df_proc["class"]
    )

    with st.spinner(_["data_upload.spinner.upload"]):
        if clear_flag:
            st.info(_["data_upload.info.clearing"])
            clear_index(token, api_url)
        if not upload_data(train_df, token, api_url):
            st.error(_["data_upload.error.train_upload_failed"])
            return None

    with st.spinner(_["data_upload.spinner.predict"]):
        metrics = compute_metrics_backend(test_df, token, api_url)
    if not metrics:
        st.error(_["data_upload.error.processing"].format(error="compute_metrics_backend failed"))
        return None

    st.success(_["data_upload.success.upload_done"])

    # опциональная загрузка тестовой части
    with st.expander(_["data_upload.upload_test.title"], expanded=False):
        upload_test = st.checkbox(_["data_upload.upload_test.checkbox"], value=True)
        if upload_test:
            with st.spinner(_["data_upload.upload_test.spinner"]):
                token2 = get_token(api_url, username, password)
                ok = upload_data(test_df, token2, api_url) if token2 else False
                if ok:
                    st.success(_["data_upload.upload_test.success"].format(n=len(test_df)))
                else:
                    st.error(_["data_upload.upload_test.error"])

    return metrics


def _metrics_ui(metrics: dict, _: dict) -> None:
    st.subheader(_["data_upload.metrics.title"])
    st.write(_["data_upload.metrics.accuracy"].format(acc=metrics.get("accuracy", 0.0)))
    st.write(
        _["data_upload.metrics.valid_total"].format(
            valid=metrics.get("n_valid", 0),
            total=metrics.get("n_total", 0),
        )
    )

    with st.expander(_["data_upload.metrics.report_table"], expanded=True):
        report_obj = metrics.get("classification_report_dict", {})
        if report_obj:
            st.dataframe(get_classification_report_df(report_obj))
        else:
            st.text(metrics.get("classification_report_text", ""))

    with st.expander(_["data_upload.metrics.by_class_plot"], expanded=False):
        report_obj = metrics.get("classification_report_dict", {})
        if report_obj:
            fig = plot_classification_metrics(report_obj)
            st.pyplot(fig)
        else:
            st.info(_["data_upload.metrics.no_metrics"])

    with st.expander(_["data_upload.metrics.cm_title"], expanded=False):
        cm = metrics.get("confusion_matrix")
        classes = metrics.get("classes")
        if cm and classes:
            st.pyplot(plot_confusion_matrix(cm, classes))
        else:
            st.info(_["data_upload.metrics.no_cm"])


def _download_processed_ui(df_proc: pd.DataFrame, uploaded_name: str, _: dict) -> None:
    csv_data = df_proc.to_csv(index=False).encode("utf-8")
    base = uploaded_name.rsplit(".", 1)[0]
    st.download_button(
        _["data_upload.download.button"],
        csv_data,
        f"processed_{base}.csv",
        "text/csv",
    )


def _last_metrics_ui(api_url: str, username: str, password: str, _: dict) -> None:
    with st.expander(_["data_upload.last_metrics.title"], expanded=True):
        token = get_token(api_url, username, password)
        if not token:
            st.info(_["data_upload.last_metrics.none"])
            return

        metrics = get_last_metrics(token, api_url)
        if not metrics:
            st.info(_["data_upload.last_metrics.none"])
            return

        st.write(_["data_upload.last_metrics.file"].format(filename=metrics.get("filename", "")))
        st.write(_["data_upload.last_metrics.date"].format(ts=metrics.get("timestamp", "")))
        st.write(_["data_upload.metrics.accuracy"].format(acc=metrics.get("accuracy", 0.0)))

        report_obj = metrics.get("classification_report_dict", {})
        if report_obj:
            st.subheader(_["data_upload.metrics.report_table"])
            st.dataframe(get_classification_report_df(report_obj))
        else:
            st.text(metrics.get("classification_report_text", ""))

        cm = metrics.get("confusion_matrix")
        classes = metrics.get("classes")
        if cm and classes:
            st.pyplot(plot_confusion_matrix(cm, classes))
        if report_obj:
            st.pyplot(plot_classification_metrics(report_obj))


# ================================
#   Public entry
# ================================
def render_data_upload_tab(api_url: str, username: str, password: str):
    _ = _tr()
    st.title(_["data_upload.title"])

    uploaded_file = st.file_uploader(
        _["data_upload.file_uploader"],
        type=["csv", "xlsx", "xls"],
    )

    if not uploaded_file:
        _last_metrics_ui(api_url, username, password, _)
        return

    try:
        df = _read_uploaded(uploaded_file)
    except Exception as e:
        st.error(_["data_upload.unsupported_format"] + f": {str(e)}")
        _last_metrics_ui(api_url, username, password, _)
        return

    st.success(_["data_upload.file_loaded"].format(filename=uploaded_file.name))
    _save_local_copy(uploaded_file, data_dir="uploaded_data")

    with st.expander(_["data_upload.preview"], expanded=False):
        st.dataframe(df)

    id_col, subject_col, description_col, target_col = _select_columns_ui(df, _)

    # drop пустые таргеты
    df = df.dropna(subset=[target_col])

    _target_stats_ui(df, target_col, _)

    df_proc = _normalize_df(df, id_col, subject_col, description_col, target_col)
    df_proc = _filter_ui(df_proc, api_url, username, password, _)
    _post_filter_stats_ui(df_proc, _)

    clear_flag = st.checkbox(_["data_upload.clear_index_checkbox"], value=True)

    if st.button(_["data_upload.run_button"]):
        metrics = _run_backend_pipeline(df_proc, clear_flag, api_url, username, password, _)
        if metrics:
            _metrics_ui(metrics, _)
            _download_processed_ui(df_proc, uploaded_file.name, _)

    _last_metrics_ui(api_url, username, password, _)
