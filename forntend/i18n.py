# i18n.py
# Словари переводов + утилиты. Доступ к текстам: TR["section.key"]

from typing import Dict

LANG_NAMES: Dict[str, str] = {
    "ru": "Русский",
    "en": "English",
}

LANGS: Dict[str, Dict[str, str]] = {
    # =========================
    # RU (UI тексты на русском)
    # Примечание: некоторые ключи по просьбе оставлены на английском
    # =========================
    "ru": {
        # --- Common / App ---
        "app.title": "Классификатор запросов",
        "common.lang_label": "Language",              # оставить на англ.
        "common.subject": "Тема",
        "common.description": "Описание",

        # --- Auth ---
        "auth.title": "Авторизация",
        "auth.username": "Имя пользователя",
        "auth.password": "Пароль",
        "auth.sign_in": "Войти",
        "auth.sign_out": "Выйти",
        "auth.failed": "Неверные учетные данные",

        # --- Tabs ---
        "tabs.classification": "Классификация",
        "tabs.similar_docs": "Похожие документы",
        "tabs.data_upload": "Загрузка данных",
        "tabs.retrain": "Ретрейн",

        # --- Classification ---
        "classification.title": "Классификация запросов",
        "classification.pick_example.title": "Выбор запроса",
        "classification.pick_example.use_default": "Использовать предустановленный запрос",
        "classification.pick_example.select_label": "Выберите пример запроса:",
        "classification.example.info.id": "ID:",
        "classification.example.info.subject": "Тема:",
        "classification.example.info.description": "Описание:",
        "classification.example.info.class": "Класс:",
        "classification.example.info.task": "Задача:",
        "classification.form.title": "Данные для классификации",
        "classification.form.subject": "Тема (subject):",
        "classification.form.description": "Описание (description):",
        "classification.form.run": "Классифицировать",
        "classification.warn.empty": "Введите тему или описание",
        "classification.spinner.token": "Получение токена...",
        "classification.spinner.predict": "Классификация запроса...",
        "classification.success.predicted": "Запрос успешно классифицирован!",
        "classification.results.title": "Результаты классификации",
        "classification.results.top5.title": "Топ-5 классов по вероятности",
        "classification.results.top_class": "Предсказанный класс:",
        "classification.results.top_proba": "Вероятность:",
        "classification.plot.x_class": "Класс",
        "classification.plot.y_prob": "Вероятность",
        "classification.plot.unavailable": "График недоступен",
        "classification.error.no_result": "Не удалось получить результаты классификации",

        # --- Similar Docs ---
        "similar.title": "Поиск похожих документов",
        "similar.use_default": "Использовать предустановленный запрос",
        "similar.select_label": "Выберите пример запроса:",
        "similar.form.title": "Данные для поиска",
        "similar.form.subject": "Тема запроса:",
        "similar.form.description": "Описание запроса:",
        "similar.form.limit": "Количество результатов",
        "similar.form.search_btn": "Искать",
        "similar.warn.empty": "Введите тему или описание для поиска",
        "similar.spinner.token": "Получение токена...",
        "similar.spinner.search": "Поиск похожих документов...",
        "similar.found": "Найдено {n} документов",
        "similar.query.title": "Ваш запрос",
        "similar.top3.title": "Топ-3 самых похожих документа",
        "similar.all.title": "Все найденные документы",
        "similar.table.request_id": "ID заявки",
        "similar.table.class": "Класс",
        "similar.table.score": "Оценка сходства",
        "similar.table.score_short": "Оценка",
        "similar.plot.class_dist.title": "Распределение документов по классам",
        "similar.plot.class_dist.xlabel": "Класс",
        "similar.plot.class_dist.ylabel": "Количество",
        "similar.plot.class_pie.title": "Процентное соотношение классов",
        "similar.none": "Похожие документы не найдены",

        # --- Data Upload ---
        "data_upload.title": "Загрузка данных и оценка качества",
        "data_upload.file_uploader": "Выберите файл с данными",
        "data_upload.unsupported_format": "Неподдерживаемый формат файла",
        "data_upload.file_loaded": "Файл {filename} успешно загружен",
        "data_upload.preview": "Предварительный просмотр данных",

        "data_upload.columns.title": "Выбор колонок для обработки",
        "data_upload.columns.id": "Колонка с ID заявки:",
        "data_upload.columns.subject_opt": "Колонка subject (необязательно):",
        "data_upload.columns.none": "— нет —",
        "data_upload.columns.description": "Колонка для поля description:",
        "data_upload.columns.target": "Колонка с целевой переменной (class):",

        "data_upload.target_stats.title": "Статистика по целевой переменной",
        "data_upload.target_stats.caption": "Распределение значений целевой переменной:",
        "data_upload.target_stats.chart_title": "Топ-{top_n} значений целевой переменной",

        "data_upload.common.value": "Значение",
        "data_upload.common.count": "Количество",
        "data_upload.common.percent": "Процент",
        "data_upload.common.others": "Others",     # оставить на англ.

        "data_upload.filter.title": "Обработка редких значений / фильтрация классов",
        "data_upload.filter.method_label": "Выберите метод фильтрации:",
        "data_upload.filter.by_freq": "По частоте встречаемости",
        "data_upload.filter.by_quality": "По качеству (F1)",
        "data_upload.filter.by_freq_slider": "Количество сохраняемых наиболее частых значений",
        "data_upload.filter.min_samples": "Минимальное количество образцов для сохранения класса",
        "data_upload.filter.min_f1": "Минимальное значение F1-score для сохранения класса",
        "data_upload.filter.confirm_params": "Установите галочку, если выбрали параметры",

        "data_upload.after_filter_stats.title": "Статистика после обработки",
        "data_upload.after_filter_stats.chart_title": "Распределение значений после обработки",
        "data_upload.after_filter_data.title": "Данные после обработки",

        "data_upload.clear_index_checkbox": "Очистить существующий индекс перед загрузкой",
        "data_upload.run_button": "Загрузить данные и рассчитать метрики",

        "data_upload.spinner.upload": "Загрузка данных в систему...",
        "data_upload.info.clearing": "Очистка существующего индекса...",
        "data_upload.success.upload_done": "Данные успешно загружены",
        "data_upload.spinner.predict": "Расчёт метрик на бэкенде...",

        "data_upload.metrics.title": "Результаты оценки",
        "data_upload.metrics.accuracy": "Accuracy: {acc:.4f}",
        "data_upload.metrics.report_table": "Отчёт о классификации",
        "data_upload.metrics.by_class_plot": "Визуализация метрик по классам",
        "data_upload.metrics.by_class_plot_title": "Метрики классификации по классам",
        "data_upload.metrics.class_label": "Класс",
        "data_upload.metrics.metric_label": "Метрика",
        "data_upload.metrics.cm_title": "Матрица ошибок",
        "data_upload.metrics.predicted": "Предсказанный класс",
        "data_upload.metrics.true": "Истинный класс",
        "data_upload.metrics.no_metrics": "Нет метрик для отображения",
        "data_upload.metrics.no_cm": "Нет матрицы ошибок",

        "data_upload.upload_test.title": "Загрузка тестовой выборки",
        "data_upload.upload_test.checkbox": "Загрузить тестовую выборку в базу данных",
        "data_upload.upload_test.spinner": "Загрузка тестовой выборки в систему...",
        "data_upload.upload_test.success": "Тестовая выборка успешно загружена ({n} записей)",
        "data_upload.upload_test.error": "Ошибка при загрузке тестовой выборки",

        "data_upload.download.button": "Скачать обработанные данные",
        "data_upload.error.processing": "Ошибка при обработке файла: {error}",

        "data_upload.last_metrics.title": "Последние метрики",
        "data_upload.last_metrics.file": "Файл: {filename}",
        "data_upload.last_metrics.date": "Дата расчёта: {ts}",
        "data_upload.last_metrics.cm_caption": "Матрица ошибок",
        "data_upload.last_metrics.metrics_caption": "Метрики классификации по классам",
        "data_upload.last_metrics.none": "Нет сохранённых метрик. Загрузите файл и рассчитайте метрики.",

        # --- Доп. ключи для quality-фильтра ---
        "data_upload.filter.quality_analysis.title": "Анализ качества классификации по классам",
        "data_upload.filter.quality_label": "Качество",
        "data_upload.quality.high": "Высокое",
        "data_upload.quality.low": "Низкое",
        "data_upload.filter.f1_threshold": "Порог F1-score: {thr}",
        "data_upload.filter.quality_stats_after": "Статистика после фильтрации по качеству",
        "data_upload.quality.summary.total_classes": "Исходное количество классов: {n}",
        "data_upload.quality.summary.after_rare": "Классов после фильтрации редких: {n}",
        "data_upload.quality.summary.high": "Классов с высоким качеством (F1 ≥ {thr}): {n}",
        "data_upload.quality.summary.rows_kept": "Строк сохранено: {kept} из {total} ({pct})",

        # --- Retrain (UI + формы) ---
        "retrain.title": "Настройка периодического переобучения",

        "retrain.ui.no_saved_yet": "Сохранённой конфигурации пока нет. Заполните поля и нажмите «Сохранить».",
        "retrain.ui.loaded_saved": "Загружена ранее сохранённая конфигурация. Измените значения и нажмите «Обновить».",
        "retrain.ui.btn_save": "Сохранить",
        "retrain.ui.btn_update": "Обновить",
        "retrain.ui.btn_delete": "Удалить",
        "retrain.ui.push_saving": "Сохранение конфигурации...",
        "retrain.ui.push_saved": "Конфигурация сохранена.",
        "retrain.ui.push_updated": "Конфигурация обновлена.",
        "retrain.ui.deleting": "Удаление конфигурации...",
        "retrain.ui.deleted": "Конфигурация удалена.",
        "retrain.ui.nothing_to_delete": "Удалять нечего.",
        "retrain.ui.delete_title": "Удалить конфигурацию?",
        "retrain.ui.delete_confirm_text": "Действительно удалить сохранённую конфигурацию ретрейна? Действие необратимо.",
        "retrain.ui.delete_yes": "Да, удалить",
        "retrain.ui.delete_cancel": "Отмена",

        "retrain.data_source.title": "Источник данных",
        "retrain.data_source.data_api_url.label": "Источник данных (URL)",
        "retrain.data_source.data_api_url.help": "HTTP(S) эндпоинт внешнего сервиса, который возвращает тренировочный датасет.",

        "retrain.data_source.auth.title": "Доступ к источнику данных",
        "retrain.data_source.auth.method.label": "Тип авторизации",
        "retrain.data_source.auth.method.help": "Способ аутентификации при обращении к источнику.",
        "retrain.data_source.auth.method.none": "Без авторизации",
        "retrain.data_source.auth.method.basic": "Basic (логин/пароль)",
        "retrain.data_source.auth.method.bearer": "Bearer-токен",
        "retrain.data_source.auth.method.api_key": "API Key (заголовок)",
        "retrain.data_source.auth.method.headers_json": "Заголовки (JSON)",

        "retrain.data_source.auth.user.label": "Логин",
        "retrain.data_source.auth.user.help": "Имя пользователя для Basic-авторизации.",
        "retrain.data_source.auth.pass.label": "Пароль",
        "retrain.data_source.auth.pass.help": "Пароль для Basic-авторизации.",
        "retrain.data_source.auth.token.label": "Bearer-токен",
        "retrain.data_source.auth.token.help": "Значение для Authorization: Bearer <token>.",
        "retrain.data_source.auth.api_key.header.label": "Имя заголовка",
        "retrain.data_source.auth.api_key.header.help": "Напр., X-API-Key или Authorization.",
        "retrain.data_source.auth.api_key.value.label": "Значение ключа",
        "retrain.data_source.auth.api_key.value.help": "Секретный API-ключ.",
        "retrain.data_source.auth.headers_json.label": "Заголовки (JSON)",
        "retrain.data_source.auth.headers_json.help": "Полный объект заголовков (ключ-значение) в JSON.",

        "retrain.columns.title": "Колонки датасета",
        "retrain.columns.id_col.label": "ID колонка",
        "retrain.columns.id_col.help": "Уникальный идентификатор записи.",
        "retrain.columns.subject_col.label": "Subject (опционально)",
        "retrain.columns.subject_col.help": "Пусто — subject не используется.",
        "retrain.columns.description_col.label": "Description",
        "retrain.columns.description_col.help": "Основной текст сообщения.",
        "retrain.columns.class_col.label": "Target",
        "retrain.columns.class_col.help": "Целевая метка класса.",
        "retrain.columns.none": "— нет —",

        "retrain.filter.title": "Фильтрация классов",
        "retrain.filter.method_label": "Метод фильтрации",
        "retrain.filter.method.help": "by_freq — топ-N частых; by_quality — min_samples и min F1.",
        "retrain.filter.opt.by_freq": "По топ-N частоте",
        "retrain.filter.opt.by_quality": "По качеству (min_samples + min F1)",
        "retrain.filter.top_n_values.label": "Top-N классов",
        "retrain.filter.top_n_values.help": "Сколько самых частых классов оставить.",
        "retrain.filter.min_samples.label": "Min samples",
        "retrain.filter.min_samples.help": "Минимум объектов на класс.",
        "retrain.filter.min_f1_score.label": "Min F1",
        "retrain.filter.min_f1_score.help": "Минимальный F1 (0..1).",

        "retrain.index_mode.title": "Режим индекса",
        "retrain.index_mode.clear_index_flag.label": "Очистить индекс перед загрузкой",
        "retrain.index_mode.clear_index_flag.help": "Вкл — полное переобучение с нуля; выкл — дообучение.",

        "retrain.schedule.title": "Расписание ретрейна",
        "retrain.schedule.run_every_days.label": "RUN_EVERY_DAYS",
        "retrain.schedule.run_every_days.help": "Период в днях (>=1).",
        "retrain.schedule.anchor_date.label": "ANCHOR_DATE_STR",
        "retrain.schedule.anchor_date.help": "Дата якоря (YYYY-MM-DD).",
        "retrain.schedule.anchor_time.label": "ANCHOR_TIME",
        "retrain.schedule.anchor_time.help": "Время якоря (HH:MM).",
        "retrain.schedule.run_on_start.label": "RUN_ON_START",
        "retrain.schedule.run_on_start.help": "Запустить сразу при старте контейнера.",

        "retrain.output.title": "Итоговая конфигурация",
        "retrain.output.validation_failed": "Заполните обязательные поля:",

        # Retrain errors (формы)
        "retrain.err.data_api_url": "Укажите URL источника данных.",
        "retrain.err.id_col": "Укажите колонку ID.",
        "retrain.err.description_col": "Укажите колонку description.",
        "retrain.err.class_col": "Укажите колонку target.",
        "retrain.err.data_auth.username": "Укажите логин для Basic-авторизации.",
        "retrain.err.data_auth.password": "Укажите пароль для Basic-авторизации.",
        "retrain.err.data_auth.token": "Укажите bearer-токен.",
        "retrain.err.data_auth.api_key_header": "Укажите имя заголовка для API Key.",
        "retrain.err.data_auth.api_key_value": "Укажите значение API Key.",
        "retrain.err.data_auth.headers_json_empty": "Укажите JSON с заголовками.",
        "retrain.err.data_auth.headers_json_not_dict": "JSON заголовков должен быть непустым объектом.",
        "retrain.err.data_auth.headers_json_invalid": "Некорректный JSON заголовков.",
        "retrain.err.top_n_values": "Top-N классов должен быть >= 1.",
        "retrain.err.min_samples": "Min samples должен быть >= 1.",
        "retrain.err.min_f1_score": "Min F1 должен быть в диапазоне [0, 1].",
        "retrain.err.run_every_days": "RUN_EVERY_DAYS должен быть >= 1.",

        # --- API-level (общие сообщения, используются в api_utils) ---
        "api.token.failed_http": "Ошибка авторизации: {status} {text}",
        "api.conn.failed": "Не удалось подключиться к API: {url}",
        "api.token.failed_generic": "Не удалось получить токен: {error}",
        "api.user.fetch_failed": "Ошибка при запросе профиля: {status} {text}",

        "api.predict.failed": "Ошибка при классификации: {status} {text}",
        "api.search.failed": "Ошибка при поиске: {status} {text}",

        "api.index.clear_failed": "Ошибка при очистке индекса: {status} {text}",
        "api.index.clear_unexpected": "Неожиданный ответ при очистке индекса: {data}",
        "api.index.cleared": "Индекс успешно очищен",

        "api.upload.unexpected_response": "Неожиданный ответ при загрузке: {data}",
        "api.upload.batch_partial": "Батч {i}/{n}: загружено {ok}/{total}, пропущено {skipped}",
        "api.upload.skipped_many": "Пропущено записей: {n}. Первые 50: {preview}",
        "api.upload.done": "Загружено {n} записей",
        "api.predict.loop_warnings": "Пустых/ошибочных ответов: {n} из {total} (empty={empty}, errors={errors})",

        "api.metrics.compute_failed": "Ошибка расчёта метрик: {status} {text}",
        "api.metrics.latest_failed": "Ошибка запроса последних метрик: {status} {text}",
        "api.metrics.compute_empty": "Сервер вернул пустые метрики.",
        "api.metrics.report_empty": "Пустой отчёт классификации.",
        "api.token.failed_for_quality": "Не удалось получить токен для анализа качества.",

        "api.retrain.err.cfg_not_dict": "cfg должен быть словарём (dict)",
        "api.retrain.err.missing_field": "Нет обязательного поля: {field}",
        "api.retrain.err.data_api_not_dict": "data_api должен быть словарём",
        "api.retrain.err.data_api_url_required": "data_api.url обязателен",
        "api.retrain.err.data_api_auth_required": "data_api.auth обязателен",
        "api.retrain.err.data_api_auth_method_required": "data_api.auth.method обязателен",
        "api.retrain.err.bad_filter_method": "filter_method должен быть 'by_freq' или 'by_quality'",
        "api.retrain.err.top_n": "top_n_values должен быть >= 1",
        "api.retrain.err.min_samples": "min_samples должен быть >= 1",
        "api.retrain.err.min_f1": "min_f1_score должен быть в [0, 1]",
        "api.retrain.err.clear_index_flag": "clear_index_flag должен быть булевым",
        "api.retrain.err.run_every_days": "RUN_EVERY_DAYS должен быть числом (в строке)",
        "api.retrain.err.anchor_dt": "ANCHOR_DATETIME_STR должен быть формата 'YYYY-MM-DD HH:MM'",
        "api.retrain.err.run_on_start": "RUN_ON_START должен быть '0' или '1'",
        "api.retrain.err.validation_generic": "Ошибка валидации конфигурации: {error}",
        "api.retrain.invalid_cfg": "Невалидный конфиг ретрейна: {err}",
        "api.retrain.push_failed": "Сохранение конфига ретрейна не удалось: {status} {text}",
        "api.retrain.saved": "Конфиг сохранён: {path} · {ts}",
        "api.retrain.push_unexpected": "Неожиданный ответ /retrain/push: {data}",
        "api.retrain.pull_failed": "Ошибка получения конфига ретрейна: {status} {text}",
        "api.retrain.delete_failed": "Ошибка удаления конфига ретрейна: {status} {text}",
        "api.retrain.delete_failed_generic": "Ошибка удаления конфига ретрейна: {error}",
    },

    # =========================
    # EN
    # =========================
    "en": {
        # --- Common / App ---
        "app.title": "Request Classifier",
        "common.lang_label": "Language",
        "common.subject": "Subject",
        "common.description": "Description",

        # --- Auth ---
        "auth.title": "Authorization",
        "auth.username": "Username",
        "auth.password": "Password",
        "auth.sign_in": "Sign in",
        "auth.sign_out": "Sign out",
        "auth.failed": "Invalid credentials",

        # --- Tabs ---
        "tabs.classification": "Classification",
        "tabs.similar_docs": "Similar Documents",
        "tabs.data_upload": "Data Upload",
        "tabs.retrain": "Retrain",

        # --- Classification ---
        "classification.title": "Request Classification",
        "classification.pick_example.title": "Pick an example",
        "classification.pick_example.use_default": "Use a preset example",
        "classification.pick_example.select_label": "Select an example:",
        "classification.example.info.id": "ID:",
        "classification.example.info.subject": "Subject:",
        "classification.example.info.description": "Description:",
        "classification.example.info.class": "Class:",
        "classification.example.info.task": "Task:",
        "classification.form.title": "Input data for classification",
        "classification.form.subject": "Subject:",
        "classification.form.description": "Description:",
        "classification.form.run": "Classify",
        "classification.warn.empty": "Enter subject or description",
        "classification.spinner.token": "Fetching token...",
        "classification.spinner.predict": "Classifying...",
        "classification.success.predicted": "Classification completed!",
        "classification.results.title": "Classification results",
        "classification.results.top5.title": "Top-5 classes by probability",
        "classification.results.top_class": "Predicted class:",
        "classification.results.top_proba": "Probability:",
        "classification.plot.x_class": "Class",
        "classification.plot.y_prob": "Probability",
        "classification.plot.unavailable": "Plot is unavailable",
        "classification.error.no_result": "Failed to get classification results",

        # --- Similar Docs ---
        "similar.title": "Similar Documents Search",
        "similar.use_default": "Use a preset query",
        "similar.select_label": "Select a query example:",
        "similar.form.title": "Query data",
        "similar.form.subject": "Query subject:",
        "similar.form.description": "Query description:",
        "similar.form.limit": "Number of results",
        "similar.form.search_btn": "Search",
        "similar.warn.empty": "Enter subject or description to search",
        "similar.spinner.token": "Fetching token...",
        "similar.spinner.search": "Searching similar documents...",
        "similar.found": "{n} documents found",
        "similar.query.title": "Your query",
        "similar.top3.title": "Top-3 most similar documents",
        "similar.all.title": "All results",
        "similar.table.request_id": "Request ID",
        "similar.table.class": "Class",
        "similar.table.score": "Similarity score",
        "similar.table.score_short": "Score",
        "similar.plot.class_dist.title": "Documents by class",
        "similar.plot.class_dist.xlabel": "Class",
        "similar.plot.class_dist.ylabel": "Count",
        "similar.plot.class_pie.title": "Class share (%)",
        "similar.none": "No similar documents found",

        # --- Data Upload ---
        "data_upload.title": "Data Upload & Evaluation",
        "data_upload.file_uploader": "Choose a data file",
        "data_upload.unsupported_format": "Unsupported file format",
        "data_upload.file_loaded": "File {filename} loaded successfully",
        "data_upload.preview": "Data preview",

        "data_upload.columns.title": "Select columns",
        "data_upload.columns.id": "Request ID column:",
        "data_upload.columns.subject_opt": "Subject column (optional):",
        "data_upload.columns.none": "— none —",
        "data_upload.columns.description": "Description column:",
        "data_upload.columns.target": "Target column (class):",

        "data_upload.target_stats.title": "Target distribution",
        "data_upload.target_stats.caption": "Target value counts:",
        "data_upload.target_stats.chart_title": "Top-{top_n} target values",

        "data_upload.common.value": "Value",
        "data_upload.common.count": "Count",
        "data_upload.common.percent": "Percent",
        "data_upload.common.others": "Others",

        "data_upload.filter.title": "Rare class handling / filtering",
        "data_upload.filter.method_label": "Choose filtering method:",
        "data_upload.filter.by_freq": "By frequency",
        "data_upload.filter.by_quality": "By quality (F1)",
        "data_upload.filter.by_freq_slider": "Top-N values to keep",
        "data_upload.filter.min_samples": "Min samples per class to keep",
        "data_upload.filter.min_f1": "Min F1-score to keep",
        "data_upload.filter.confirm_params": "Check if parameters are set",

        "data_upload.after_filter_stats.title": "Post-filter statistics",
        "data_upload.after_filter_stats.chart_title": "Distribution after filtering",
        "data_upload.after_filter_data.title": "Data after filtering",

        "data_upload.clear_index_checkbox": "Clear existing index before upload",
        "data_upload.run_button": "Upload data and compute metrics",

        "data_upload.spinner.upload": "Uploading data...",
        "data_upload.info.clearing": "Clearing existing index...",
        "data_upload.success.upload_done": "Data uploaded successfully",
        "data_upload.spinner.predict": "Computing metrics on backend...",

        "data_upload.metrics.title": "Evaluation results",
        "data_upload.metrics.accuracy": "Accuracy: {acc:.4f}",
        "data_upload.metrics.report_table": "Classification report",
        "data_upload.metrics.by_class_plot": "Per-class metrics",
        "data_upload.metrics.by_class_plot_title": "Metrics by class",
        "data_upload.metrics.class_label": "Class",
        "data_upload.metrics.metric_label": "Metric",
        "data_upload.metrics.cm_title": "Confusion matrix",
        "data_upload.metrics.predicted": "Predicted class",
        "data_upload.metrics.true": "True class",
        "data_upload.metrics.no_metrics": "No metrics to plot",
        "data_upload.metrics.no_cm": "No confusion matrix",

        "data_upload.upload_test.title": "Upload test split",
        "data_upload.upload_test.checkbox": "Upload test split into database",
        "data_upload.upload_test.spinner": "Uploading test split...",
        "data_upload.upload_test.success": "Test split uploaded ({n} rows)",
        "data_upload.upload_test.error": "Failed to upload test split",

        "data_upload.download.button": "Download processed data",
        "data_upload.error.processing": "Error while processing file: {error}",

        "data_upload.last_metrics.title": "Last metrics",
        "data_upload.last_metrics.file": "File: {filename}",
        "data_upload.last_metrics.date": "Computed at: {ts}",
        "data_upload.last_metrics.cm_caption": "Confusion matrix",
        "data_upload.last_metrics.metrics_caption": "Per-class metrics",
        "data_upload.last_metrics.none": "No saved metrics yet. Upload a file and compute metrics.",

        # Extra for quality flow
        "data_upload.filter.quality_analysis.title": "Quality analysis per class",
        "data_upload.filter.quality_label": "Quality",
        "data_upload.quality.high": "High",
        "data_upload.quality.low": "Low",
        "data_upload.filter.f1_threshold": "F1-score threshold: {thr}",
        "data_upload.filter.quality_stats_after": "Stats after quality filtering",
        "data_upload.quality.summary.total_classes": "Initial classes: {n}",
        "data_upload.quality.summary.after_rare": "Classes after rare filtering: {n}",
        "data_upload.quality.summary.high": "High-quality classes (F1 ≥ {thr}): {n}",
        "data_upload.quality.summary.rows_kept": "Rows kept: {kept} of {total} ({pct})",

        # --- Retrain ---
        "retrain.title": "Periodic Retrain Settings",

        "retrain.ui.no_saved_yet": "No saved configuration yet. Fill the fields and click “Save”.",
        "retrain.ui.loaded_saved": "Loaded previously saved configuration. Change values and click “Update”.",
        "retrain.ui.btn_save": "Save",
        "retrain.ui.btn_update": "Update",
        "retrain.ui.btn_delete": "Delete",
        "retrain.ui.push_saving": "Saving configuration...",
        "retrain.ui.push_saved": "Configuration saved.",
        "retrain.ui.push_updated": "Configuration updated.",
        "retrain.ui.deleting": "Deleting configuration...",
        "retrain.ui.deleted": "Configuration deleted.",
        "retrain.ui.nothing_to_delete": "Nothing to delete.",
        "retrain.ui.delete_title": "Delete configuration?",
        "retrain.ui.delete_confirm_text": "Delete the saved retrain configuration? This action cannot be undone.",
        "retrain.ui.delete_yes": "Yes, delete",
        "retrain.ui.delete_cancel": "Cancel",

        "retrain.data_source.title": "Data Source",
        "retrain.data_source.data_api_url.label": "Data source (URL)",
        "retrain.data_source.data_api_url.help": "HTTP(S) endpoint that returns the training dataset.",

        "retrain.data_source.auth.title": "Access to Data Source",
        "retrain.data_source.auth.method.label": "Auth method",
        "retrain.data_source.auth.method.help": "Authentication type for data source requests.",
        "retrain.data_source.auth.method.none": "No auth",
        "retrain.data_source.auth.method.basic": "Basic (username/password)",
        "retrain.data_source.auth.method.bearer": "Bearer token",
        "retrain.data_source.auth.method.api_key": "API Key (header)",
        "retrain.data_source.auth.method.headers_json": "Headers (JSON)",

        "retrain.data_source.auth.user.label": "Username",
        "retrain.data_source.auth.user.help": "Username for Basic auth.",
        "retrain.data_source.auth.pass.label": "Password",
        "retrain.data_source.auth.pass.help": "Password for Basic auth.",
        "retrain.data_source.auth.token.label": "Bearer token",
        "retrain.data_source.auth.token.help": "Value for Authorization: Bearer <token>.",
        "retrain.data_source.auth.api_key.header.label": "Header name",
        "retrain.data_source.auth.api_key.header.help": "E.g., X-API-Key or Authorization.",
        "retrain.data_source.auth.api_key.value.label": "Key value",
        "retrain.data_source.auth.api_key.value.help": "Secret API key.",
        "retrain.data_source.auth.headers_json.label": "Headers (JSON)",
        "retrain.data_source.auth.headers_json.help": "Full headers object (key-value) as JSON.",

        "retrain.columns.title": "Dataset Columns",
        "retrain.columns.id_col.label": "ID column",
        "retrain.columns.id_col.help": "Unique record identifier.",
        "retrain.columns.subject_col.label": "Subject (optional)",
        "retrain.columns.subject_col.help": "Leave empty to skip subject.",
        "retrain.columns.description_col.label": "Description",
        "retrain.columns.description_col.help": "Main message text.",
        "retrain.columns.class_col.label": "Target",
        "retrain.columns.class_col.help": "Target class label.",
        "retrain.columns.none": "— none —",

        "retrain.filter.title": "Class Filtering",
        "retrain.filter.method_label": "Filtering method",
        "retrain.filter.method.help": "by_freq — top-N frequent; by_quality — min_samples and min F1.",
        "retrain.filter.opt.by_freq": "Top-N by frequency",
        "retrain.filter.opt.by_quality": "By quality (min_samples + min F1)",
        "retrain.filter.top_n_values.label": "Top-N classes",
        "retrain.filter.top_n_values.help": "How many most frequent classes to keep.",
        "retrain.filter.min_samples.label": "Min samples",
        "retrain.filter.min_samples.help": "Minimum samples per class.",
        "retrain.filter.min_f1_score.label": "Min F1",
        "retrain.filter.min_f1_score.help": "Minimum F1 (0..1).",

        "retrain.index_mode.title": "Index Mode",
        "retrain.index_mode.clear_index_flag.label": "Clear index before upload",
        "retrain.index_mode.clear_index_flag.help": "On — full retrain; Off — incremental.",

        "retrain.schedule.title": "Retrain Schedule",
        "retrain.schedule.run_every_days.label": "RUN_EVERY_DAYS",
        "retrain.schedule.run_every_days.help": "Period in days (>=1).",
        "retrain.schedule.anchor_date.label": "ANCHOR_DATE_STR",
        "retrain.schedule.anchor_date.help": "Anchor date (YYYY-MM-DD).",
        "retrain.schedule.anchor_time.label": "ANCHOR_TIME",
        "retrain.schedule.anchor_time.help": "Anchor time (HH:MM).",
        "retrain.schedule.run_on_start.label": "RUN_ON_START",
        "retrain.schedule.run_on_start.help": "Run immediately on container start.",

        "retrain.output.title": "Final configuration",
        "retrain.output.validation_failed": "Please fill required fields:",

        # Retrain errors (form)
        "retrain.err.data_api_url": "Specify data source URL.",
        "retrain.err.id_col": "Specify ID column.",
        "retrain.err.description_col": "Specify description column.",
        "retrain.err.class_col": "Specify target column.",
        "retrain.err.data_auth.username": "Enter username for Basic auth.",
        "retrain.err.data_auth.password": "Enter password for Basic auth.",
        "retrain.err.data_auth.token": "Enter bearer token.",
        "retrain.err.data_auth.api_key_header": "Enter header name for API Key.",
        "retrain.err.data_auth.api_key_value": "Enter API Key value.",
        "retrain.err.data_auth.headers_json_empty": "Provide headers JSON.",
        "retrain.err.data_auth.headers_json_not_dict": "Headers JSON must be a non-empty object.",
        "retrain.err.data_auth.headers_json_invalid": "Invalid headers JSON.",
        "retrain.err.top_n_values": "Top-N classes must be >= 1.",
        "retrain.err.min_samples": "Min samples must be >= 1.",
        "retrain.err.min_f1_score": "Min F1 must be within [0, 1].",
        "retrain.err.run_every_days": "RUN_EVERY_DAYS must be >= 1.",

        # --- API-level ---
        "api.token.failed_http": "Auth error: {status} {text}",
        "api.conn.failed": "Cannot connect to API: {url}",
        "api.token.failed_generic": "Token request failed: {error}",
        "api.user.fetch_failed": "Failed to fetch user: {status} {text}",

        "api.predict.failed": "Prediction failed: {status} {text}",
        "api.search.failed": "Search failed: {status} {text}",

        "api.index.clear_failed": "Index clear failed: {status} {text}",
        "api.index.clear_unexpected": "Unexpected response while clearing index: {data}",
        "api.index.cleared": "Index cleared",

        "api.upload.unexpected_response": "Unexpected upload response: {data}",
        "api.upload.batch_partial": "Batch {i}/{n}: uploaded {ok}/{total}, skipped {skipped}",
        "api.upload.skipped_many": "Skipped records: {n}. First 50: {preview}",
        "api.upload.done": "Uploaded {n} records",
        "api.predict.loop_warnings": "Empty/error responses: {n} of {total} (empty={empty}, errors={errors})",

        "api.metrics.compute_failed": "Metrics compute failed: {status} {text}",
        "api.metrics.latest_failed": "Latest metrics fetch failed: {status} {text}",
        "api.metrics.compute_empty": "Backend returned empty metrics.",
        "api.metrics.report_empty": "Classification report is empty.",
        "api.token.failed_for_quality": "Cannot obtain API token for quality analysis.",

        "api.retrain.err.cfg_not_dict": "cfg must be a dict",
        "api.retrain.err.missing_field": "Missing required field: {field}",
        "api.retrain.err.data_api_not_dict": "data_api must be a dict",
        "api.retrain.err.data_api_url_required": "data_api.url is required",
        "api.retrain.err.data_api_auth_required": "data_api.auth is required",
        "api.retrain.err.data_api_auth_method_required": "data_api.auth.method is required",
        "api.retrain.err.bad_filter_method": "filter_method must be 'by_freq' or 'by_quality'",
        "api.retrain.err.top_n": "top_n_values must be >= 1",
        "api.retrain.err.min_samples": "min_samples must be >= 1",
        "api.retrain.err.min_f1": "min_f1_score must be in [0, 1]",
        "api.retrain.err.clear_index_flag": "clear_index_flag must be a boolean",
        "api.retrain.err.run_every_days": "RUN_EVERY_DAYS must be numeric string",
        "api.retrain.err.anchor_dt": "ANCHOR_DATETIME_STR must be 'YYYY-MM-DD HH:MM'",
        "api.retrain.err.run_on_start": "RUN_ON_START must be '0' or '1'",
        "api.retrain.err.validation_generic": "Config validation error: {error}",
        "api.retrain.invalid_cfg": "Invalid retrain config: {err}",
        "api.retrain.push_failed": "Retrain config save failed: {status} {text}",
        "api.retrain.saved": "Config saved: {path} · {ts}",
        "api.retrain.push_unexpected": "Unexpected /retrain/push response: {data}",
        "api.retrain.pull_failed": "Retrain config fetch failed: {status} {text}",
        "api.retrain.delete_failed": "Retrain delete failed: {status} {text}",
        "api.retrain.delete_failed_generic": "Retrain delete error: {error}",
    },
}


def t(key: str, lang: str = "ru") -> str:
    return LANGS.get(lang, LANGS["ru"]).get(key, key)


class _Translator:
    def __init__(self, lang: str):
        self.lang = lang

    def __getitem__(self, key: str) -> str:
        return t(key, self.lang)


def get_translations(lang: str) -> _Translator:
    return _Translator(lang)


def get_lang_display_name(code: str) -> str:
    return LANG_NAMES.get(code, code)


def get_lang_options():
    return ["ru", "en"]
