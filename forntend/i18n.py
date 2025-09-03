# i18n.py
# Словари переводов + утилиты.
# Доступ к текстам: TR["page.key"]  (где TR = get_translations(lang))

from typing import Dict

# Читаемые имена языков для селектора
LANG_NAMES: Dict[str, str] = {
    "ru": "Русский",
    "en": "English",
}

# Плоские ключи: section.subsection.element
LANGS: Dict[str, Dict[str, str]] = {
    "ru": {
        # === App / Common ===
        "app.title": "Классификатор запросов",
        "common.lang_label": "Language",
        "common.subject": "Тема",
        "common.description": "Описание",

        # === Auth ===
        "auth.title": "Авторизация",
        "auth.username": "Имя пользователя",
        "auth.password": "Пароль",
        "auth.sign_in": "Войти",
        "auth.sign_out": "Выйти",
        "auth.failed": "Неверные учетные данные",

        # === Tabs ===
        "tabs.classification": "Классификация",
        "tabs.similar_docs": "Похожие документы",
        "tabs.data_upload": "Загрузка данных",

        # === Classification ===
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

        # === Similar Docs ===
        "similar.title": "Поиск похожих документов",
        "similar.use_default": "Использовать предустановленный запрос для поиска",
        "similar.select_label": "Выберите пример запроса для поиска:",
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

        # === Data Upload ===
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
        "data_upload.common.others": "Others",

        "data_upload.filter.title": "Обработка редких значений / фильтрация классов",
        "data_upload.filter.method_label": "Выберите метод фильтрации:",
        "data_upload.filter.by_freq": "По частоте встречаемости",
        "data_upload.filter.by_quality": "По качеству классификации",
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
        "data_upload.spinner.predict": "Расчёт метрик на бэкенде...",  # <— обновлено

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
        "data_upload.metrics.no_metrics": "Нет метрик для отображения",   # <— добавлено
        "data_upload.metrics.no_cm": "Нет матрицы ошибок",                 # <— добавлено

        "data_upload.upload_test.title": "Загрузка тестовой выборки",
        "data_upload.upload_test.checkbox": "Загрузить тестовую выборку в базу данных",
        "data_upload.upload_test.spinner": "Загрузка тестовой выборки в систему...",
        "data_upload.upload_test.success": "Тестовая выборка успешно загружена ({n} записей)",
        "data_upload.upload_test.error": "Ошибка при загрузке тестовой выборки",

        "data_upload.download.button": "Скачать обработанные данные",
        "data_upload.error.processing": "Ошибка при обработке файла: {error}",

        "data_upload.last_metrics.title": "Последние метрики",
        "data_upload.last_metrics.file": "Файл: {filename}",
        "data_upload.last_metrics.date": "Дата расчёта: {ts}",  # <— обновлено
        "data_upload.last_metrics.cm_caption": "Матрица ошибок",
        "data_upload.last_metrics.metrics_caption": "Метрики классификации по классам",
        "data_upload.last_metrics.none": "Нет сохранённых метрик. Загрузите файл и рассчитайте метрики.",
    },

    "en": {
        # === App / Common ===
        "app.title": "Request Classifier",
        "common.lang_label": "Language",
        "common.subject": "Subject",
        "common.description": "Description",

        # === Auth ===
        "auth.title": "Authorization",
        "auth.username": "Username",
        "auth.password": "Password",
        "auth.sign_in": "Sign in",
        "auth.sign_out": "Sign out",
        "auth.failed": "Invalid credentials",

        # === Tabs ===
        "tabs.classification": "Classification",
        "tabs.similar_docs": "Similar Documents",
        "tabs.data_upload": "Data Upload",

        # === Classification ===
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

        # === Similar Docs ===
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

        # === Data Upload ===
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
        "data_upload.spinner.predict": "Computing metrics on backend...",  # <— updated

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
        "data_upload.metrics.no_metrics": "No metrics to plot",     # <— added
        "data_upload.metrics.no_cm": "No confusion matrix",         # <— added

        "data_upload.upload_test.title": "Upload test split",
        "data_upload.upload_test.checkbox": "Upload test split into database",
        "data_upload.upload_test.spinner": "Uploading test split...",
        "data_upload.upload_test.success": "Test split uploaded ({n} rows)",
        "data_upload.upload_test.error": "Failed to upload test split",

        "data_upload.download.button": "Download processed data",
        "data_upload.error.processing": "Error while processing file: {error}",

        "data_upload.last_metrics.title": "Last metrics",
        "data_upload.last_metrics.file": "File: {filename}",
        "data_upload.last_metrics.date": "Computed at: {ts}",  # <— updated
        "data_upload.last_metrics.cm_caption": "Confusion matrix",
        "data_upload.last_metrics.metrics_caption": "Per-class metrics",
        "data_upload.last_metrics.none": "No saved metrics yet. Upload a file and compute metrics.",
    },
}

def t(key: str, lang: str = "ru") -> str:
    """Возвращает перевод по ключу, иначе сам ключ."""
    return LANGS.get(lang, LANGS["ru"]).get(key, key)

class _Translator:
    """Позволяет обращаться как к словарю: TR['section.key']"""
    def __init__(self, lang: str):
        self.lang = lang

    def __getitem__(self, key: str) -> str:
        return t(key, self.lang)

def get_translations(lang: str) -> _Translator:
    return _Translator(lang)

def get_lang_display_name(code: str) -> str:
    return LANG_NAMES.get(code, code)

def get_lang_options():
    """Возвращает список кодов языков в стабильном порядке."""
    return ["ru", "en"]
