# tab_classification.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from api_utils import get_token, classify_request
from config import DEFAULT_EXAMPLES
from i18n import get_translations

def render_classification_tab(api_url, username, password):
    """Вкладка классификации (RU/EN через i18n)."""
    lang = st.session_state.get("lang", "ru")
    TR = get_translations(lang)
    _ = lambda k: TR[k]

    st.title(_("classification.title"))

    # --- Блок выбора примера ---
    st.subheader(_("classification.pick_example.title"))
    use_default = st.checkbox(_("classification.pick_example.use_default"))

    if use_default:
        example_index = st.selectbox(
            _("classification.pick_example.select_label"),
            options=range(len(DEFAULT_EXAMPLES)),
            format_func=lambda i: f"{DEFAULT_EXAMPLES[i]['id']} - {DEFAULT_EXAMPLES[i]['subject']}",
        )
        selected_example = DEFAULT_EXAMPLES[example_index]

        st.info(
            f"**{_('classification.example.info.id')}** {selected_example['id']}\n\n"
            f"**{_('classification.example.info.subject')}** {selected_example['subject']}\n\n"
            f"**{_('classification.example.info.description')}** {selected_example['description']}\n\n"
            f"**{_('classification.example.info.class')}** {selected_example['class']}\n\n"
            f"**{_('classification.example.info.task')}** {selected_example.get('task','-')}\n"
        )

        default_subject = selected_example["subject"]
        default_description = selected_example["description"]
    else:
        default_subject = ""
        default_description = ""

    # --- Форма ---
    st.subheader(_("classification.form.title"))
    subject = st.text_input(_("classification.form.subject"), value=default_subject)
    description = st.text_area(_("classification.form.description"), value=default_description, height=200)

    if st.button(_("classification.form.run")):
        if not subject and not description:
            st.warning(_("classification.warn.empty"))
        else:
            with st.spinner(_("classification.spinner.token")):
                token = get_token(api_url, username, password)

            if token:
                with st.spinner(_("classification.spinner.predict")):
                    result = classify_request(subject, description, token, api_url)

                if result and "predictions" in result and isinstance(result["predictions"], list) and result["predictions"]:
                    st.success(_("classification.success.predicted"))

                    # Таблица предсказаний
                    predictions_df = pd.DataFrame(result["predictions"])
                    st.subheader(_("classification.results.title"))
                    st.dataframe(predictions_df, use_container_width=True)

                    # Визуализация топ-5 по вероятности
                    try:
                        top5 = predictions_df.sort_values("probability", ascending=False).head(5)
                        fig, ax = plt.subplots(figsize=(10, 5))
                        sns.barplot(x="class_name", y="probability", data=top5, ax=ax)
                        ax.set_xlabel(_("classification.plot.x_class"))
                        ax.set_ylabel(_("classification.plot.y_prob"))
                        ax.set_title(_("classification.results.top5.title"))
                        plt.xticks(rotation=45, ha="right")
                        plt.tight_layout()
                        st.pyplot(fig)
                    except Exception:
                        st.info(_("classification.plot.unavailable"))

                    # Топ-1 класс
                    try:
                        st.subheader(f"{_('classification.results.top_class')} {result['predictions'][0]['class_name']}")
                        st.subheader(f"{_('classification.results.top_proba')} {result['predictions'][0]['probability']:.2f}")
                    except Exception:
                        pass
                else:
                    st.error(_("classification.error.no_result"))