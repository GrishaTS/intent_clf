# tab_data_upload.py
import os
import json
from datetime import datetime
from io import BytesIO
import base64

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
#   Streamlit-экран
# ================================
def render_data_upload_tab(api_url, username, password):
    lang = st.session_state.get("lang", "ru")
    TR = get_translations(lang)
    _ = lambda k: TR[k]

    st.title(_("data_upload.title"))

    data_dir = "uploaded_data"
    os.makedirs(data_dir, exist_ok=True)

    # Загрузка файла
    uploaded_file = st.file_uploader(
        _("data_upload.file_uploader"),
        type=["csv", "xlsx", "xls"],
    )

    if uploaded_file:
        file_extension = uploaded_file.name.split(".")[-1].lower()

        try:
            if file_extension == "csv":
                df = pd.read_csv(uploaded_file)
            elif file_extension in ("xlsx", "xls"):
                engine = "openpyxl" if file_extension == "xlsx" else "xlrd"
                df = pd.read_excel(uploaded_file, engine=engine)
            else:
                st.error(_("data_upload.unsupported_format") + f": {file_extension}")
                return

            st.success(_("data_upload.file_loaded").format(filename=uploaded_file.name))

            # сохраняем исходный файл локально
            with open(os.path.join(data_dir, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.expander(_("data_upload.preview"), expanded=False):
                st.dataframe(df)

            # --- Выбор колонок ---
            st.subheader(_("data_upload.columns.title"))
            all_columns = df.columns.tolist()

            NONE_LABEL = _("data_upload.columns.none")
            col0, col1, col2 = st.columns(3)
            with col0:
                id_col = st.selectbox(
                    _("data_upload.columns.id"),
                    options=all_columns,
                    index=all_columns.index("id") if "id" in all_columns else 0,
                )
            with col1:
                subject_options = [NONE_LABEL] + all_columns
                subj_default_idx = (
                    subject_options.index("subject") if "subject" in all_columns else 0
                )
                subject_choice = st.selectbox(
                    _("data_upload.columns.subject_opt"),
                    options=subject_options,
                    index=subj_default_idx,
                )
                subject_col = None if subject_choice == NONE_LABEL else subject_choice
            with col2:
                description_col = st.selectbox(
                    _("data_upload.columns.description"),
                    options=all_columns,
                    index=all_columns.index("description") if "description" in all_columns else 0,
                )

            target_col = st.selectbox(
                _("data_upload.columns.target"),
                options=all_columns,
                index=all_columns.index("class") if "class" in all_columns else 0,
            )

            # drop пустые таргеты
            df = df.dropna(subset=[target_col])

            # --- Статистика по таргету ---
            with st.expander(_("data_upload.target_stats.title"), expanded=False):
                target_counts = df[target_col].value_counts()
                st.write(_("data_upload.target_stats.caption"))

                target_stats_df = pd.DataFrame({
                    _("data_upload.common.value"): target_counts.index,
                    _("data_upload.common.count"): target_counts.values,
                    _("data_upload.common.percent"): (target_counts.values / target_counts.sum() * 100).round(2),
                })
                st.dataframe(target_stats_df)

                fig, ax = plt.subplots(figsize=(10, 6))
                top_n = min(20, len(target_counts))
                target_counts.head(top_n).plot(kind="bar", ax=ax)
                ax.set_title(_("data_upload.target_stats.chart_title").format(top_n=top_n))
                ax.set_xlabel(_("data_upload.common.value"))
                ax.set_ylabel(_("data_upload.common.count"))
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig)

            # --- Фильтрация ---
            st.subheader(_("data_upload.filter.title"))

            filter_method = st.radio(
                _("data_upload.filter.method_label"),
                [_("data_upload.filter.by_freq"), _("data_upload.filter.by_quality")],
            )

            # нормализуем колонки
            base_cols = [id_col, description_col, target_col]
            df_processed = df[base_cols].rename(
                columns={id_col: "id", description_col: "description", target_col: "class"}
            )
            if subject_col:
                df_processed["subject"] = df[subject_col]
            else:
                df_processed["subject"] = "no_subject"
            df_processed = df_processed[["id", "subject", "description", "class"]]

            if filter_method == _("data_upload.filter.by_freq"):
                top_n_values = st.slider(
                    _("data_upload.filter.by_freq_slider"),
                    min_value=1,
                    max_value=min(100, df_processed["class"].nunique()),
                    value=10,
                )
                top_values = df_processed["class"].value_counts().head(top_n_values).index.tolist()
                df_processed["class"] = df_processed["class"].apply(
                    lambda x: x if x in top_values else _("data_upload.common.others")
                )
            else:
                min_samples = st.slider(
                    _("data_upload.filter.min_samples"),
                    min_value=1,
                    max_value=100,
                    value=10,
                )
                min_f1_score = st.slider(
                    _("data_upload.filter.min_f1"),
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                )
                if st.checkbox(_("data_upload.filter.confirm_params"), value=False):
                    df_high_quality, _ = filter_high_quality_classes(
                        df_processed,
                        min_samples=min_samples,
                        min_f1_score=min_f1_score,
                        api_url=api_url,
                        username=username,
                        password=password,
                    )
                    if df_high_quality is not None:
                        df_processed = df_high_quality

            # --- Статистика после обработки ---
            with st.expander(_("data_upload.after_filter_stats.title"), expanded=False):
                new_counts = df_processed["class"].value_counts()
                new_stats_df = pd.DataFrame({
                    _("data_upload.common.value"): new_counts.index,
                    _("data_upload.common.count"): new_counts.values,
                    _("data_upload.common.percent"): (new_counts.values / new_counts.sum() * 100).round(2),
                })
                st.dataframe(new_stats_df)

                fig2, ax2 = plt.subplots(figsize=(10, 6))
                new_counts.plot(kind="bar", ax=ax2)
                ax2.set_title(_("data_upload.after_filter_stats.chart_title"))
                ax2.set_xlabel(_("data_upload.common.value"))
                ax2.set_ylabel(_("data_upload.common.count"))
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig2)

            with st.expander(_("data_upload.after_filter_data.title"), expanded=False):
                st.dataframe(df_processed)

            # --- Очистка индекса ---
            clear_index_flag = st.checkbox(
                _("data_upload.clear_index_checkbox"),
                value=True,
            )

            # --- Запуск вычислений ---
            if st.button(_("data_upload.run_button")):
                # split
                train_df, test_df = train_test_split(
                    df_processed, test_size=0.1, random_state=42, stratify=df_processed["class"]
                )

                with st.spinner(_("data_upload.spinner.upload")):
                    token = get_token(api_url, username, password)
                    if not token:
                        st.error("Auth failed")
                        return
                    if clear_index_flag:
                        st.info(_("data_upload.info.clearing"))
                        clear_index(token, api_url)
                    if not upload_data(train_df, token, api_url):
                        st.error("Train upload failed")
                        return

                # считаем метрики на backend
                with st.spinner(_("data_upload.spinner.predict")):
                    metrics = compute_metrics_backend(test_df, token, api_url)

                if not metrics:
                    st.error(_("data_upload.error.processing").format(error="compute_metrics_backend failed"))
                    return

                st.success(_("data_upload.success.upload_done"))

                # === Метрики ===
                st.subheader(_("data_upload.metrics.title"))
                st.write(_("data_upload.metrics.accuracy").format(acc=metrics.get("accuracy", 0.0)))
                st.write(f"valid: {metrics.get('n_valid', 0)} / {metrics.get('n_total', 0)}")

                # Табличный отчёт
                with st.expander(_("data_upload.metrics.report_table"), expanded=True):
                    report_obj = metrics.get("classification_report_dict", {})
                    if report_obj:
                        report_df = get_classification_report_df(report_obj)
                        st.dataframe(report_df)
                    else:
                        st.text(metrics.get("classification_report_text", ""))

                # Визуализации (строим на фронте)
                with st.expander(_("data_upload.metrics.by_class_plot"), expanded=False):
                    report_obj = metrics.get("classification_report_dict", {})
                    if report_obj:
                        fig = plot_classification_metrics(report_obj)
                        st.pyplot(fig)
                    else:
                        st.info("No metrics to plot")

                with st.expander(_("data_upload.metrics.cm_title"), expanded=False):
                    cm = metrics.get("confusion_matrix")
                    classes = metrics.get("classes")
                    if cm and classes:
                        fig = plot_confusion_matrix(cm, classes)
                        st.pyplot(fig)
                    else:
                        st.info("No confusion matrix")

                # Загрузка тестовой выборки (опционально)
                with st.expander(_("data_upload.upload_test.title"), expanded=False):
                    upload_test_data = st.checkbox(_("data_upload.upload_test.checkbox"), value=True)
                    if upload_test_data:
                        with st.spinner(_("data_upload.upload_test.spinner")):
                            token2 = get_token(api_url, username, password)
                            ok = upload_data(test_df, token2, api_url) if token2 else False
                            if ok:
                                st.success(_("data_upload.upload_test.success").format(n=len(test_df)))
                            else:
                                st.error(_("data_upload.upload_test.error"))

                # Скачать обработанные данные
                csv_data = df_processed.to_csv(index=False).encode("utf-8")
                st.download_button(
                    _("data_upload.download.button"),
                    csv_data,
                    f"processed_{uploaded_file.name.split('.')[0]}.csv",
                    "text/csv",
                )

        except Exception as e:
            st.error(_("data_upload.error.processing").format(error=str(e)))

    # --- Последние метрики с бэкенда ---
    with st.expander(_("data_upload.last_metrics.title"), expanded=True):
        token = get_token(api_url, username, password)
        if not token:
            st.info(_("data_upload.last_metrics.none"))
            return

        metrics = get_last_metrics(token, api_url)
        if not metrics:
            st.info(_("data_upload.last_metrics.none"))
        else:
            st.write(_("data_upload.last_metrics.file").format(filename=metrics.get("filename", "")))
            st.write(_("data_upload.last_metrics.date").format(ts=metrics.get("timestamp", "")))
            st.write(_("data_upload.metrics.accuracy").format(acc=metrics.get("accuracy", 0.0)))

            report_obj = metrics.get("classification_report_dict", {})
            if report_obj:
                st.subheader(_("data_upload.metrics.report_table"))
                report_df = get_classification_report_df(report_obj)
                st.dataframe(report_df)
            else:
                st.text(metrics.get("classification_report_text", ""))

            cm = metrics.get("confusion_matrix")
            classes = metrics.get("classes")
            if cm and classes:
                st.pyplot(plot_confusion_matrix(cm, classes))
            if report_obj:
                st.pyplot(plot_classification_metrics(report_obj))
