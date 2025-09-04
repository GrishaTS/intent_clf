from __future__ import annotations

from typing import Any, Dict, Optional
import streamlit as st

from api_utils import get_token, get_current_user
from i18n import t, get_translations, get_lang_options, get_lang_display_name


# ===== one-time state bootstrap =====
st.session_state.setdefault("auth", {"ok": False, "username": None, "password": None, "token": None})
st.session_state.setdefault("lang", "ru")         # язык приложения (после входа)
st.session_state.setdefault("lang_login", "ru")   # язык экрана логина (локальный селектор)
st.session_state.setdefault("auth_user_key", None)  # стабильный ключ пользователя (id/username)
st.session_state.setdefault("auth_profile", None)   # кэш профиля из бэкенда


# ===== i18n helpers =====
def current_lang() -> str:
    """Язык активного экрана (на логине — lang_login, после входа — lang)."""
    return st.session_state["lang"] if st.session_state["auth"]["ok"] else st.session_state.get("lang_login", "ru")


def tr() -> dict:
    """Объект-переводчик: TR[k]."""
    return get_translations(current_lang())


# ===== state utilities =====
def _clear_streamlit_caches() -> None:
    try:
        st.cache_data.clear()
    except Exception:
        pass
    try:
        st.cache_resource.clear()
    except Exception:
        pass


def _logout_reset_state() -> None:
    """Полный сброс состояния при выходе. Сохраняем только выбранный язык логина."""
    lang_login = st.session_state.get("lang_login", "ru")
    _clear_streamlit_caches()
    st.session_state.clear()
    st.session_state["auth"] = {"ok": False, "username": None, "password": None, "token": None}
    st.session_state["lang_login"] = lang_login
    st.session_state["lang"] = "ru"
    st.session_state["auth_user_key"] = None
    st.session_state["auth_profile"] = None


def _user_key_from_profile(user: Optional[Dict[str, Any]], fallback_username: Optional[str]) -> str:
    """Стабильный ключ пользователя для неймспейса состояния."""
    if isinstance(user, dict):
        return str(user.get("id") or user.get("username") or fallback_username or "anonymous")
    return str(fallback_username or "anonymous")


def _reset_per_user_state_if_changed(new_user_key: str) -> None:
    """
    Если сменился пользователь — чистим только состояние функциональных вкладок.
    Не трогаем auth/lang и прочие общие ключи.
    """
    prev = st.session_state.get("auth_user_key")
    if prev != new_user_key:
        prefixes = ("retrain_", "classification_", "similar_", "data_upload_")
        for k in list(st.session_state.keys()):
            if any(k.startswith(p) for p in prefixes):
                del st.session_state[k]
        st.session_state["auth_user_key"] = new_user_key


def _ensure_user_namespace_consistency() -> None:
    """
    Защитная проверка: если auth_user_key не совпадает с текущим 'логином/профилем',
    обновим неймспейс (полезно в редких сценариях смены учётки без перезагрузки).
    """
    auth = st.session_state.get("auth") or {}
    profile = st.session_state.get("auth_profile")
    user_key_now = _user_key_from_profile(profile, auth.get("username"))
    if st.session_state.get("auth_user_key") != user_key_now:
        _reset_per_user_state_if_changed(user_key_now)


# ===== UI components =====
def top_bar() -> None:
    _ = tr()
    _ensure_user_namespace_consistency()

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.session_state["auth"]["ok"]:
            st.markdown(
                f"""
                <div style="
                    background-color:#999999;
                    padding:6px 12px;
                    border-radius:8px;
                    text-align:center;
                    font-weight:500;">
                    👤 {st.session_state['auth']['username']}
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(_["auth.sign_out"], use_container_width=True):
                _logout_reset_state()
                st.rerun()


def login_screen(api_url: str) -> None:
    _ = tr()

    # селектор языка на экране логина (локальный)
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

            # профиль пользователя + язык из бэкенда (если задан)
            user = get_current_user(api_url, tok)
            lang_from_backend = None
            if isinstance(user, dict):
                lang_from_backend = (user.get("language") or "").strip() or None

            # выставляем язык приложения (НЕ связан с виджетом lang_login)
            st.session_state["lang"] = lang_from_backend or st.session_state["lang_login"]

            # неймспейс на пользователя
            new_user_key = _user_key_from_profile(user, u)
            _reset_per_user_state_if_changed(new_user_key)

            # фиксируем аутентификацию и профиль
            st.session_state["auth"] = {"ok": True, "username": u, "password": p, "token": tok}
            st.session_state["auth_profile"] = user

            st.rerun()


def ensure_auth(api_url: str) -> None:
    """
    Показывает экран логина, если пользователь не аутентифицирован.
    Также гарантирует консистентность пер-юзерного состояния.
    """
    if not st.session_state["auth"]["ok"]:
        login_screen(api_url)
        st.stop()
    _ensure_user_namespace_consistency()
