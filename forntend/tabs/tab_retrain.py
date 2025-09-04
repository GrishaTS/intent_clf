from __future__ import annotations

import json
from datetime import date, time, datetime
from typing import Any, Dict, Tuple

import streamlit as st

from i18n import get_translations
from api_utils import get_token, retrain_push, retrain_pull, retrain_delete


# ================= i18n =================
def _tr() -> Dict[str, str]:
    return get_translations(st.session_state.get("lang", "ru"))


# ================= state =================
def _init_state() -> None:
    st.session_state.setdefault("retrain_loaded", False)
    st.session_state.setdefault("retrain_saved_cfg", None)
    st.session_state.setdefault("retrain_saved_meta", {})
    st.session_state.setdefault("retrain_show_delete_dialog", False)


def _reset_state() -> None:
    st.session_state["retrain_loaded"] = False
    st.session_state["retrain_saved_cfg"] = None
    st.session_state["retrain_saved_meta"] = {}
    st.session_state["retrain_show_delete_dialog"] = False
    st.session_state.pop("retrain_delete_confirm", None)


# ================= utils =================
def _pretty_json(d: dict) -> str:
    try:
        return json.dumps(d, ensure_ascii=False, indent=2)
    except Exception:
        return "{}"


def _get_from_cfg(cfg: dict | None, def_val: Any, *path: str) -> Any:
    cur = cfg
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return def_val
        cur = cur[p]
    return cur if cur is not None else def_val


# ================= load saved =================
def _load_saved_once(api_url: str, username: str | None, password: str | None) -> None:
    if st.session_state["retrain_loaded"]:
        return
    token = get_token(api_url, username or "", password or "") if username and password else None
    if token:
        data = retrain_pull(token, api_url)
        if isinstance(data, dict) and data.get("ok") and data.get("config"):
            st.session_state["retrain_saved_cfg"] = data["config"]
            st.session_state["retrain_saved_meta"] = {
                "path": data.get("path"),
                "updated_at": data.get("updated_at"),
            }
    st.session_state["retrain_loaded"] = True


def _defaults(saved_cfg: dict | None) -> Dict[str, Any]:
    # источник
    data_api_url = _get_from_cfg(saved_cfg, "", "data_api", "url")
    auth_method = _get_from_cfg(saved_cfg, "none", "data_api", "auth", "method")
    basic_user = _get_from_cfg(saved_cfg, "", "data_api", "auth", "username")
    basic_pass = _get_from_cfg(saved_cfg, "", "data_api", "auth", "password")
    bearer_token = _get_from_cfg(saved_cfg, "", "data_api", "auth", "token")
    api_key_header = _get_from_cfg(saved_cfg, "X-API-Key", "data_api", "auth", "header")
    api_key_value = _get_from_cfg(saved_cfg, "", "data_api", "auth", "value")
    headers_dict = _get_from_cfg(saved_cfg, {}, "data_api", "auth", "headers")
    headers_json = _pretty_json(headers_dict) if headers_dict else '{\n  "Authorization": "Bearer <token>"\n}'

    # колонки
    id_col = _get_from_cfg(saved_cfg, "Id", "id_col")
    subject_col = _get_from_cfg(saved_cfg, "", "subject_col") or ""
    description_col = _get_from_cfg(saved_cfg, "Body", "description_col")
    class_col = _get_from_cfg(saved_cfg, "Target", "class_col")

    # фильтрация
    filter_method = _get_from_cfg(saved_cfg, "by_freq", "filter_method")
    top_n_values = int(_get_from_cfg(saved_cfg, 3, "top_n_values"))
    min_samples = int(_get_from_cfg(saved_cfg, 10, "min_samples"))
    min_f1_score = float(_get_from_cfg(saved_cfg, 0.5, "min_f1_score"))

    # индекс
    clear_index_flag = bool(_get_from_cfg(saved_cfg, True, "clear_index_flag"))

    # расписание
    run_every_days = int(_get_from_cfg(saved_cfg, 14, "RUN_EVERY_DAYS"))
    anchor_str = _get_from_cfg(saved_cfg, f"{date.today().strftime('%Y-%m-%d')} 06:00", "ANCHOR_DATETIME_STR")
    try:
        adate = datetime.strptime(anchor_str, "%Y-%m-%d %H:%M").date()
        atime = datetime.strptime(anchor_str, "%Y-%m-%d %H:%M").time()
    except Exception:
        adate, atime = date.today(), time(6, 0)
    run_on_start = _get_from_cfg(saved_cfg, "1", "RUN_ON_START") == "1"

    return {
        # источник
        "data_api_url": data_api_url,
        "auth_method": auth_method,
        "basic_user": basic_user,
        "basic_pass": basic_pass,
        "bearer_token": bearer_token,
        "api_key_header": api_key_header,
        "api_key_value": api_key_value,
        "headers_json": headers_json,
        # колонки
        "id_col": id_col,
        "subject_col": subject_col,
        "description_col": description_col,
        "class_col": class_col,
        # фильтрация
        "filter_method": filter_method,
        "top_n_values": top_n_values,
        "min_samples": min_samples,
        "min_f1_score": min_f1_score,
        # индекс
        "clear_index_flag": clear_index_flag,
        # расписание
        "run_every_days": run_every_days,
        "adate": adate,
        "atime": atime,
        "run_on_start": run_on_start,
    }


# ================= sections =================
def _section_header_saved(_: Dict[str, str], saved_exists: bool) -> None:
    st.markdown(_["retrain.ui.loaded_saved"] if saved_exists else _["retrain.ui.no_saved_yet"])


def _section_data_source(_: Dict[str, str], d: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    st.subheader(_["retrain.data_source.title"])
    data_api_url = st.text_input(
        _["retrain.data_source.data_api_url.label"],
        value=d["data_api_url"],
        placeholder="https://example.com/export",
        help=_["retrain.data_source.data_api_url.help"],
    )

    st.subheader(_["retrain.data_source.auth.title"])
    method_options = ["none", "basic", "bearer", "api_key", "headers_json"]
    method_labels = {
        "none": _["retrain.data_source.auth.method.none"],
        "basic": _["retrain.data_source.auth.method.basic"],
        "bearer": _["retrain.data_source.auth.method.bearer"],
        "api_key": _["retrain.data_source.auth.method.api_key"],
        "headers_json": _["retrain.data_source.auth.method.headers_json"],
    }
    data_auth_method = st.radio(
        label=_["retrain.data_source.auth.method.label"],
        options=method_options,
        index=method_options.index(d["auth_method"]) if d["auth_method"] in method_options else 0,
        horizontal=True,
        format_func=lambda v: method_labels[v],
        help=_["retrain.data_source.auth.method.help"],
    )

    auth: Dict[str, Any] = {"method": data_auth_method}
    if data_auth_method == "basic":
        c_u, c_p = st.columns(2)
        with c_u:
            user = st.text_input(_["retrain.data_source.auth.user.label"], value=d["basic_user"], help=_["retrain.data_source.auth.user.help"])
        with c_p:
            pwd = st.text_input(_["retrain.data_source.auth.pass.label"], value=d["basic_pass"], type="password", help=_["retrain.data_source.auth.pass.help"])
        auth.update({"username": user, "password": pwd})
    elif data_auth_method == "bearer":
        tok = st.text_input(_["retrain.data_source.auth.token.label"], value=d["bearer_token"], type="password", help=_["retrain.data_source.auth.token.help"])
        auth.update({"token": tok})
    elif data_auth_method == "api_key":
        c_h, c_v = st.columns(2)
        with c_h:
            hdr = st.text_input(_["retrain.data_source.auth.api_key.header.label"], value=d["api_key_header"], help=_["retrain.data_source.auth.api_key.header.help"])
        with c_v:
            val = st.text_input(_["retrain.data_source.auth.api_key.value.label"], value=d["api_key_value"], type="password", help=_["retrain.data_source.auth.api_key.value.help"])
        auth.update({"header": hdr, "value": val})
    elif data_auth_method == "headers_json":
        headers_json_str = st.text_area(_["retrain.data_source.auth.headers_json.label"], value=d["headers_json"], height=120, help=_["retrain.data_source.auth.headers_json.help"])
        auth.update({"headers_json": headers_json_str})

    st.divider()
    return data_api_url, auth


def _section_columns(_: Dict[str, str], d: Dict[str, Any]) -> Tuple[str, str | None, str, str]:
    st.subheader(_["retrain.columns.title"])
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        id_col = st.text_input(_["retrain.columns.id_col.label"], value=d["id_col"], help=_["retrain.columns.id_col.help"])
    with c2:
        subject_col_raw = st.text_input(_["retrain.columns.subject_col.label"], value=d["subject_col"], placeholder=_["retrain.columns.none"], help=_["retrain.columns.subject_col.help"])
        subject_col = subject_col_raw or None
    with c3:
        description_col = st.text_input(_["retrain.columns.description_col.label"], value=d["description_col"], help=_["retrain.columns.description_col.help"])
    with c4:
        class_col = st.text_input(_["retrain.columns.class_col.label"], value=d["class_col"], help=_["retrain.columns.class_col.help"])
    st.divider()
    return id_col, subject_col, description_col, class_col


def _section_filter(_: Dict[str, str], d: Dict[str, Any]) -> Tuple[str, int, int, float]:
    st.subheader(_["retrain.filter.title"])
    filter_method = st.radio(
        label=_["retrain.filter.method_label"],
        options=["by_freq", "by_quality"],
        index=0 if d["filter_method"] == "by_freq" else 1,
        horizontal=True,
        format_func=lambda v: _["retrain.filter.opt.by_freq"] if v == "by_freq" else _["retrain.filter.opt.by_quality"],
        help=_["retrain.filter.method.help"],
    )

    if filter_method == "by_freq":
        top_n_values = st.number_input(_["retrain.filter.top_n_values.label"], min_value=1, max_value=10000, value=d["top_n_values"], step=1, help=_["retrain.filter.top_n_values.help"])
        min_samples, min_f1_score = d["min_samples"], d["min_f1_score"]
    else:
        top_n_values = d["top_n_values"]
        c_ms, c_f1 = st.columns(2)
        with c_ms:
            min_samples = st.number_input(_["retrain.filter.min_samples.label"], min_value=1, max_value=1000, value=d["min_samples"], step=1, help=_["retrain.filter.min_samples.help"])
        with c_f1:
            min_f1_score = st.number_input(_["retrain.filter.min_f1_score.label"], min_value=0.0, max_value=1.0, value=float(d["min_f1_score"]), step=0.05, help=_["retrain.filter.min_f1_score.help"])
    st.divider()
    return filter_method, int(top_n_values), int(min_samples), float(min_f1_score)


def _section_index(_: Dict[str, str], d: Dict[str, Any]) -> bool:
    st.subheader(_["retrain.index_mode.title"])
    flag = st.checkbox(_["retrain.index_mode.clear_index_flag.label"], value=d["clear_index_flag"], help=_["retrain.index_mode.clear_index_flag.help"])
    st.divider()
    return bool(flag)


def _section_schedule(_: Dict[str, str], d: Dict[str, Any]) -> Tuple[int, str, bool]:
    st.subheader(_["retrain.schedule.title"])
    run_every_days = st.number_input(_["retrain.schedule.run_every_days.label"], min_value=1, max_value=365, value=d["run_every_days"], step=1, help=_["retrain.schedule.run_every_days.help"])
    c_anchor_date, c_anchor_time, c_ros = st.columns([1, 1, 1])
    with c_anchor_date:
        anchor_date_val = st.date_input(_["retrain.schedule.anchor_date.label"], value=d["adate"], help=_["retrain.schedule.anchor_date.help"])
    with c_anchor_time:
        anchor_time_val = st.time_input(_["retrain.schedule.anchor_time.label"], value=d["atime"], help=_["retrain.schedule.anchor_time.help"])
    with c_ros:
        run_on_start = st.checkbox(_["retrain.schedule.run_on_start.label"], value=d["run_on_start"], help=_["retrain.schedule.run_on_start.help"])
    anchor_datetime_str = f"{anchor_date_val.strftime('%Y-%m-%d')} {anchor_time_val.strftime('%H:%M')}"
    st.divider()
    return int(run_every_days), anchor_datetime_str, bool(run_on_start)


# ================= validation & cfg =================
def _validate(_: Dict[str, str], cfg: Dict[str, Any]) -> list[str]:
    errors: list[str] = []

    def _require(val: Any, key: str) -> None:
        if val is None:
            errors.append(_[key]); return
        if isinstance(val, str) and val.strip() == "":
            errors.append(_[key])

    _require(cfg["data_api"]["url"], "retrain.err.data_api_url")
    _require(cfg["id_col"], "retrain.err.id_col")
    _require(cfg["description_col"], "retrain.err.description_col")
    _require(cfg["class_col"], "retrain.err.class_col")

    auth = cfg["data_api"]["auth"] or {}
    m = (auth.get("method") or "none").strip()
    if m == "basic":
        _require(auth.get("username"), "retrain.err.data_auth.username")
        _require(auth.get("password"), "retrain.err.data_auth.password")
    elif m == "bearer":
        _require(auth.get("token"), "retrain.err.data_auth.token")
    elif m == "api_key":
        _require(auth.get("header"), "retrain.err.data_auth.api_key_header")
        _require(auth.get("value"), "retrain.err.data_auth.api_key_value")
    elif m == "headers_json":
        raw = (auth.get("headers_json") or "").strip()
        if raw == "":
            errors.append(_["retrain.err.data_auth.headers_json_empty"])
        else:
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict) or not parsed:
                    errors.append(_["retrain.err.data_auth.headers_json_not_dict"])
            except Exception:
                errors.append(_["retrain.err.data_auth.headers_json_invalid"])

    if cfg["filter_method"] == "by_freq":
        if int(cfg["top_n_values"]) < 1:
            errors.append(_["retrain.err.top_n_values"])
    else:
        if int(cfg["min_samples"]) < 1:
            errors.append(_["retrain.err.min_samples"])
        f1 = float(cfg["min_f1_score"])
        if f1 < 0.0 or f1 > 1.0:
            errors.append(_["retrain.err.min_f1_score"])

    if int(cfg["RUN_EVERY_DAYS"]) < 1:
        errors.append(_["retrain.err.run_every_days"])

    return errors


def _build_cfg(
    data_api_url: str,
    data_auth: Dict[str, Any],
    id_col: str,
    subject_col: str | None,
    description_col: str,
    class_col: str,
    filter_method: str,
    top_n_values: int,
    min_samples: int,
    min_f1_score: float,
    clear_index_flag: bool,
    run_every_days: int,
    anchor_datetime_str: str,
    run_on_start: bool,
) -> Dict[str, Any]:
    return {
        "data_api": {"url": data_api_url, "auth": data_auth},
        "id_col": id_col,
        "subject_col": subject_col,
        "description_col": description_col,
        "class_col": class_col,
        "filter_method": filter_method,
        "top_n_values": int(top_n_values),
        "min_samples": int(min_samples),
        "min_f1_score": float(min_f1_score),
        "clear_index_flag": bool(clear_index_flag),
        "RUN_EVERY_DAYS": str(int(run_every_days)),
        "ANCHOR_DATETIME_STR": anchor_datetime_str,
        "RUN_ON_START": "1" if run_on_start else "0",
    }


# ================= actions =================
def _save_or_update(_: Dict[str, str], api_url: str, username: str | None, password: str | None, cfg: Dict[str, Any], is_update: bool) -> None:
    if not username or not password:
        st.error(_["auth.failed"]); return
    token = get_token(api_url, username, password)
    if not token:
        st.error(_["auth.failed"]); return
    with st.spinner(_["retrain.ui.push_saving"]):
        resp = retrain_push(cfg, token, api_url)
    if resp and resp.get("ok"):
        st.success(_["retrain.ui.push_updated"] if is_update else _["retrain.ui.push_saved"])
        _reset_state()
        st.rerun()


def _open_delete_dialog() -> None:
    st.session_state["retrain_show_delete_dialog"] = True


def _delete_dialog(_: Dict[str, str], api_url: str, username: str | None, password: str | None) -> None:
    if not st.session_state.get("retrain_show_delete_dialog"):
        return
    try:
        @st.dialog(_["retrain.ui.delete_title"])
        def _dlg():
            st.write(_["retrain.ui.delete_confirm_text"])
            c1, c2 = st.columns(2)
            with c1:
                if st.button(_["retrain.ui.delete_yes"], type="primary", use_container_width=True, key="retrain_delete_yes"):
                    if not username or not password:
                        st.error(_["auth.failed"]); return
                    token = get_token(api_url, username, password)
                    if not token:
                        st.error(_["auth.failed"]); return
                    with st.spinner(_["retrain.ui.deleting"]):
                        dresp = retrain_delete(token, api_url)
                    if dresp and dresp.get("ok"):
                        st.success(_["retrain.ui.deleted"] if dresp.get("deleted") else _["retrain.ui.nothing_to_delete"])
                        _reset_state()
                        st.rerun()
            with c2:
                if st.button(_["retrain.ui.delete_cancel"], use_container_width=True, key="retrain_delete_no"):
                    st.session_state["retrain_show_delete_dialog"] = False
                    st.rerun()
        _dlg()
    except Exception:
        # Фолбэк без st.dialog
        with st.container(border=True):
            st.write(f"**{_['retrain.ui.delete_title']}**")
            st.write(_["retrain.ui.delete_confirm_text"])
            c1, c2 = st.columns(2)
            with c1:
                if st.button(_["retrain.ui.delete_yes"], type="primary", use_container_width=True, key="retrain_delete_yes_fb"):
                    if not username or not password:
                        st.error(_["auth.failed"]); return
                    token = get_token(api_url, username, password)
                    if not token:
                        st.error(_["auth.failed"]); return
                    with st.spinner(_["retrain.ui.deleting"]):
                        dresp = retrain_delete(token, api_url)
                    if dresp and dresp.get("ok"):
                        st.success(_["retrain.ui.deleted"] if dresp.get("deleted") else _["retrain.ui.nothing_to_delete"])
                        _reset_state()
                        st.rerun()
            with c2:
                if st.button(_["retrain.ui.delete_cancel"], use_container_width=True, key="retrain_delete_no_fb"):
                    st.session_state["retrain_show_delete_dialog"] = False
                    st.rerun()


# ================= entry =================
def render_retrain_tab(api_url: str, username: str | None, password: str | None) -> None:
    _ = _tr()
    _init_state()
    st.title(_["retrain.title"])

    _load_saved_once(api_url, username, password)
    saved_cfg: dict | None = st.session_state.get("retrain_saved_cfg")
    saved_exists = bool(saved_cfg)

    d = _defaults(saved_cfg)
    _section_header_saved(_, saved_exists)

    data_api_url, data_auth = _section_data_source(_, d)
    id_col, subject_col, description_col, class_col = _section_columns(_, d)
    filter_method, top_n_values, min_samples, min_f1_score = _section_filter(_, d)
    clear_index_flag = _section_index(_, d)
    run_every_days, anchor_datetime_str, run_on_start = _section_schedule(_, d)

    cfg = _build_cfg(
        data_api_url, data_auth,
        id_col, subject_col, description_col, class_col,
        filter_method, top_n_values, min_samples, min_f1_score,
        clear_index_flag, run_every_days, anchor_datetime_str, run_on_start,
    )

    st.subheader(_["retrain.output.title"])
    errors = _validate(_, cfg)
    st.code(_pretty_json(cfg), language="json")
    if errors:
        st.error(_["retrain.output.validation_failed"])
        for e in errors:
            st.warning(f"• {e}")
        return

    st.divider()
    if not saved_exists:
        if st.button(_["retrain.ui.btn_save"], use_container_width=True):
            _save_or_update(_, api_url, username, password, cfg, is_update=False)
    else:
        col_upd, col_del = st.columns([2, 1])
        with col_upd:
            if st.button(_["retrain.ui.btn_update"], use_container_width=True):
                _save_or_update(_, api_url, username, password, cfg, is_update=True)
        with col_del:
            if st.button(_["retrain.ui.btn_delete"], use_container_width=True):
                _open_delete_dialog()

    _delete_dialog(_, api_url, username, password)
