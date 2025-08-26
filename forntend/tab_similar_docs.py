# tab_similar_docs.py
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from collections import Counter

from api_utils import get_token, search_similar
from config import DEFAULT_EXAMPLES
from i18n import get_translations

def render_similar_docs_tab(api_url, username, password):
    """Вкладка поиска похожих документов (RU/EN через i18n)."""
    lang = st.session_state.get("lang", "ru")
    TR = get_translations(lang)
    _ = lambda k: TR[k]

    st.title(_("similar.title"))

    # --- Пресеты ---
    use_default_search = st.checkbox(_("similar.use_default"))
    if use_default_search:
        example_index_search = st.selectbox(
            _("similar.select_label"),
            options=range(len(DEFAULT_EXAMPLES)),
            format_func=lambda i: f"{DEFAULT_EXAMPLES[i]['id']} - {DEFAULT_EXAMPLES[i]['subject']}",
            key="search_example",
        )
        selected = DEFAULT_EXAMPLES[example_index_search]
        default_search_subject = selected["subject"]
        default_search_description = selected["description"]
    else:
        default_search_subject = ""
        default_search_description = ""

    # --- Форма запроса ---
    st.subheader(_("similar.form.title"))
    search_subject = st.text_input(_("similar.form.subject"), value=default_search_subject)
    search_description = st.text_area(_("similar.form.description"), value=default_search_description, height=150)
    limit = st.slider(_("similar.form.limit"), min_value=1, max_value=20, value=10)

    if st.button(_("similar.form.search_btn")):
        if not search_subject and not search_description:
            st.warning(_("similar.warn.empty"))
        else:
            with st.spinner(_("similar.spinner.token")):
                token = get_token(api_url, username, password)

            if token:
                with st.spinner(_("similar.spinner.search")):
                    search_results = search_similar(search_subject, search_description, token, api_url, limit)

                if search_results and "results" in search_results:
                    n = len(search_results["results"])
                    st.success(_("similar.found").format(n=n))

                    # Запрос пользователя
                    st.subheader(_("similar.query.title"))
                    query_df = pd.DataFrame({
                        _("common.subject"): [search_subject],
                        _("common.description"): [search_description],
                    })
                    st.dataframe(query_df, use_container_width=True)

                    # Топ-3
                    if n > 0:
                        st.subheader(_("similar.top3.title"))
                        top_k = min(3, n)
                        top3_data = []
                        for result in search_results["results"][:top_k]:
                            top3_data.append({
                                _("similar.table.request_id"): result.get("request_id", ""),
                                _("common.subject"): result.get("subject", ""),
                                _("common.description"): result.get("description", ""),
                                _("similar.table.class"): result.get("class_name", ""),
                                _("similar.table.score"): f"{result.get('score', 0):.4f}",
                            })
                        st.dataframe(pd.DataFrame(top3_data), use_container_width=True)

                    # Все результаты
                    st.subheader(_("similar.all.title"))
                    table_rows = []
                    for r in search_results["results"]:
                        table_rows.append({
                            _("similar.table.request_id"): r.get("request_id", ""),
                            _("common.subject"): r.get("subject", ""),
                            _("common.description"): r.get("description", ""),
                            _("similar.table.class"): r.get("class_name", ""),
                            _("similar.table.score_short"): f"{r.get('score', 0):.4f}",
                        })
                    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

                    # Распределение классов
                    if search_results["results"]:
                        class_counts = Counter([r.get("class_name", "") for r in search_results["results"]])
                        if class_counts:
                            # Bar
                            classes, counts = zip(*sorted(class_counts.items(), key=lambda x: x[1], reverse=True))
                            fig, ax = plt.subplots(figsize=(10, 6))
                            bars = ax.bar(classes, counts)
                            for bar in bars:
                                h = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.1, f"{int(h)}", ha="center", va="bottom")
                            ax.set_title(_("similar.plot.class_dist.title"))
                            ax.set_xlabel(_("similar.plot.class_dist.xlabel"))
                            ax.set_ylabel(_("similar.plot.class_dist.ylabel"))
                            plt.xticks(rotation=45, ha="right")
                            plt.tight_layout()
                            st.pyplot(fig)

                            # Pie (если >1 класса)
                            if len(class_counts) > 1:
                                fig2, ax2 = plt.subplots(figsize=(8, 8))
                                ax2.pie(
                                    list(class_counts.values()),
                                    labels=list(class_counts.keys()),
                                    autopct="%1.1f%%",
                                    textprops={"fontsize": 9},
                                )
                                ax2.set_title(_("similar.plot.class_pie.title"))
                                plt.tight_layout()
                                st.pyplot(fig2)
                else:
                    st.warning(_("similar.none"))