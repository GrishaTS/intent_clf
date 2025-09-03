import streamlit as st

from api_utils import get_token, get_current_user
from i18n import t, get_translations, get_lang_options, get_lang_display_name

# ===== state (инициализация при импорте) =====
st.session_state.setdefault("auth", {"ok": False, "username": None, "password": None, "token": None})
st.session_state.setdefault("lang", "ru")        # язык приложения (после входа)
st.session_state.setdefault("lang_login", "ru")  # язык экрана логина (локальный селектор)

def current_lang() -> str:
    """Язык для текущего экрана."""
    return st.session_state["lang"] if st.session_state["auth"]["ok"] else st.session_state.get("lang_login", "ru")

def tr() -> dict:
    return get_translations(current_lang())

def top_bar() -> None:
    _ = tr()
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.session_state["auth"]["ok"]:
            st.markdown(
                f"""
                <div style="
                    background-color:#808080;
                    padding:6px 12px;
                    border-radius:8px;
                    text-align:center;
                    font-weight:500;">
                    👤 {st.session_state['auth']['username']}
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button(_["auth.sign_out"], use_container_width=True):
                st.session_state["auth"] = {"ok": False, "username": None, "password": None, "token": None}
                st.rerun()

def login_screen(api_url: str) -> None:
    _ = tr()

    # селектор языка только на логине и с отдельным ключом
    col1, col2 = st.columns([3, 1])
    with col2:
        st.selectbox(
            _["common.lang_label"],
            options=get_lang_options(),
            key="lang_login",
            format_func=get_lang_display_name,
        )

    st.title(t("app.title", current_lang()))
    with st.form("login_form", clear_on_submit=False):
        u = st.text_input(_["auth.username"])
        p = st.text_input(_["auth.password"], type="password")
        if st.form_submit_button(_["auth.sign_in"]):
            tok = get_token(api_url, u, p)
            if not tok:
                st.error(_["auth.failed"])
                return

            # тянем профиль и выставляем язык приложения из бэка
            lang_from_backend = None
            user = get_current_user(api_url, tok)
            if isinstance(user, dict):
                lang_from_backend = (user.get("language") or "").strip() or None

            # важно: назначаем 'lang' (этот ключ НЕ связан с виджетом)
            st.session_state["lang"] = lang_from_backend or st.session_state["lang_login"]

            # фиксируем аутентификацию
            st.session_state["auth"] = {"ok": True, "username": u, "password": p, "token": tok}
            st.rerun()

def ensure_auth(api_url: str) -> None:
    if not st.session_state["auth"]["ok"]:
        login_screen(api_url)
        st.stop()
