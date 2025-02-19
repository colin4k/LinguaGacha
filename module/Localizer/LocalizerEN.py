from module.Localizer.LocalizerBase import LocalizerBase

class LocalizerEN(LocalizerBase):

    # 通用
    add: str = "Add"
    edit: str = "Edit"
    none: str = "None"
    stop: str = "Stop"
    start: str = "Start"
    close: str = "Close"
    alert: str = "Alert"
    warning: str = "Warning"
    confirm: str = "Confirm"
    cancel: str = "Cancel"
    enable: str = "Enable"
    disable: str = "Disable"
    auto: str = "Auto"
    select_file: str = "Select File"
    select_file_type: str = "JSON files (*.json);;XLSX files (*.xlsx)"

    # 主页面
    app_close_message_box: str = "Are you sure you want to exit the application...?"
    app_close_message_box_msg: str = "The main window is closed, the application will automatically exit later..."
    app_theme_btn: str = "Theme"
    app_language_btn: str = "Language"
    app_settings_page: str = "App Settings"
    app_platform_page: str = "API"
    app_project_page: str = "Project Settings"
    app_translation_page: str = "Start Translation"
    app_basic_settings_page: str = "Basic Settings"
    app_advance_Feature_page: str = "Advanced Features"
    app_glossary_page: str = "Glossary"
    app_pre_translation_replacement_page: str = "Pre-Translation Replacement"
    app_post_translation_replacement_page: str = "Post-Translation Replacement"
    app_custom_prompt_navigation_item: str = "Custom Prompts"
    app_custom_prompt_zh_page: str = "Chinese Prompts"
    app_custom_prompt_en_page: str = "English Prompts"

    # 路径
    path_bilingual: str = "bilingual"
    path_glossary_export = "export_glossary.json"
    path_pre_translation_replacement_export = "export_pre_translation_replacement.json"
    path_post_translation_replacement_export = "export_post_translation_replacement.json"
    path_result_check_code = "result_check_code_anomaly_entries.json"
    path_result_check_glossary = "result_check_glossary_issue_entries.json"
    path_result_check_untranslated = "result_check_translation_status_anomaly_entries.json"

    # 日志
    log_debug_mode: str = "Debug mode enabled ..."
    log_config_file_not_exist: str = "Configuration file not found ..."
    log_api_test_fail: str = "API test failed ... "
    log_task_fail: str = "Translation task failed ..."
    log_read_file_fail: str = "File reading failed ..."
    log_write_file_fail: str = "File writing failed ..."
    log_load_cache_file_fail: str = "Failed to load cached data from file ..."
    log_load_llama_cpp_slots_num_fail: str = "Failed to get response data from [green]llama.cpp[/] ..."
    translator_max_round: str = "Max rounds"
    translator_current_round: str = "Current round"
    translator_api_url: str = "API URL"
    translator_name: str = "API Name"
    translator_model: str = "Model Name"
    translator_proxy_url: str = "Active Network Proxy"
    translator_prompt: str = "The following prompt will be used for this task:\n{PROMPT}\n"
    translator_begin: str = "Translation task is about to start, estimated total tasks: {TASKS}, concurrent tasks: {BATCH_SIZE}. Please ensure network connection ..."
    translator_writing: str = "Writing translation data, please wait ..."
    translator_done: str = "All texts are translated, translation task finished ..."
    translator_fail: str = "Maximum translation rounds reached, some texts are still untranslated. Please check the translation results ..."
    translator_stop: str = "Translation task stopped ..."
    translator_write: str = "Translation result saved to {PATH} directory ..."
    translator_generate_task: str = "Generate translation tasks"
    translator_rule_filter: str = "Rule filtering completed, {COUNT} entries not requiring translation filtered out ..."
    translator_mtool_filter: str = "MToolOptimizer preprocessing completed, {COUNT} entries containing duplicate clauses filtered out ..."
    translator_language_filter: str = "Language filtering completed, {COUNT} entries not containing target language filtered out ..."
    translator_task_response_think: str = "Model thinking:\n"
    translator_task_response_result: str = "Model response:\n"
    translator_response_check_fail: str = "Translated text failed check, will automatically retry in the next round of translation"
    translator_response_check_fail_part: str = "Partial translated text failed check, will automatically retry in the next round of translation"
    translator_task_success: str = "Task time {TIME} seconds, {LINES} lines of text, input tokens {PT}, output tokens {CT}"
    file_checker_code: str = "Code check completed, no abnormal entries found ..."
    file_checker_code_full: str = "Code check completed, {COUNT} abnormal entries found, accounting for {PERCENT} %, results written to {TARGET} ..."
    file_checker_glossary: str = "Glossary check completed, no abnormal entries found ..."
    file_checker_glossary_full: str = "Glossary check completed, {COUNT} abnormal entries found, accounting for {PERCENT} %, results written to {TARGET} ..."
    file_checker_translation: str = "Translation status check completed, no abnormal entries found ..."
    file_checker_translation_full: str = "Translation status check completed, {COUNT} abnormal entries found, accounting for {PERCENT} %, results written to {TARGET} ..."
    file_checker_translation_alert_key: str = "____Note____"
    file_checker_translation_alert_value: str = "This file lists entries that **may** have missing translations. Please judge the actual missing translation based on the context!"
    platofrm_tester_key: str = "Testing API Key"
    platofrm_tester_proxy: str = "Network proxy enabled, proxy address: "
    platofrm_tester_messages: str = "Sending prompts"
    platofrm_tester_response_think: str = "Model thinking"
    platofrm_tester_response_result: str = "Model response"
    platofrm_tester_result: str = "Tested {COUNT} APIs in total, {SUCCESS} successful, {FAILURE} failed ..."
    response_checker_unknown: str = "Unknown"
    response_checker_fail_data: str = "Response error (data structure)"
    response_checker_fail_line: str = "Response error (number of data lines)"
    response_checker_untranslated: str = "Untranslated content found in the response"
    response_decoder_glossary_by_json: str = "Glossary data [bright_blue]->[/] deserialization, total {COUNT} entries"
    response_decoder_glossary_by_rule: str = "Glossary data [bright_blue]->[/] rule parsing after split, total {COUNT} entries"
    response_decoder_translation_by_json: str = "Translation data [bright_blue]->[/] deserialization, total {COUNT} entries"
    response_decoder_translation_by_rule: str = "Translation data [bright_blue]->[/] rule parsing after split, total {COUNT} entries"

    # 应用设置
    app_settings_page_proxy_url = "Please enter network proxy address ..."
    app_settings_page_proxy_url_title = "Network Proxy Address"
    app_settings_page_proxy_url_content = "When enabled, requests will be sent to the API using the set proxy address, e.g., http://127.0.0.1:7890"
    app_settings_page_font_hinting_title = "App Font Optimization"
    app_settings_page_font_hinting_content = "When enabled, font edge rendering will be smoother (will take effect after app restart)"
    app_settings_page_debug_title = "Debug Mode"
    app_settings_page_debug_content = "When enabled, the app will display additional debug information"
    app_settings_page_scale_factor_title = "Global Scale Factor"
    app_settings_page_scale_factor_content = "When enabled, the app interface will be scaled according to the selected ratio (will take effect after app restart)"

    # 接口管理
    platform_page_api_test_doing: str = "API test is in progress, please try again later ..."
    platform_page_api_test_result: str = "API test result: {SUCCESS} successful, {FAILURE} failed ..."
    platform_page_api_activate: str = "Activate API"
    platform_page_api_edit: str = "Edit API"
    platform_page_api_args: str = "Edit Arguments"
    platform_page_api_test: str = "Test API"
    platform_page_api_delete: str = "Delete API"
    platform_page_widget_add_title: str = "API List"
    platform_page_widget_add_content: str = "Add and manage any LLM API compatible with OpenAI and Anthropic formats here"

    # 接口编辑
    platform_edit_page_name: str = "Please enter API name ..."
    platform_edit_page_name_title: str = "API Name"
    platform_edit_page_name_content: str = "Please enter API name, only for display within the app, no practical effect"
    platform_edit_page_api_url: str = "Please enter API URL ..."
    platform_edit_page_api_url_title: str = "API URL"
    platform_edit_page_api_url_content: str = "Please enter API URL, pay attention to whether /v1 needs to be added at the end"
    platform_edit_page_api_key: str = "Please enter API Key ..."
    platform_edit_page_api_key_title: str = "API Key"
    platform_edit_page_api_key_content: str = "Please enter API Key, e.g., sk-d0daba12345678fd8eb7b8d31c123456. Multiple keys can be entered for polling, one key per line"
    platform_edit_page_api_format_title: str = "API Format"
    platform_edit_page_api_format_content: str = "Please select API format. Most platforms are compatible with OpenAI format, while Claude models on some platforms use Anthropic format"
    platform_edit_page_model: str = "Please enter Model Name ..."
    platform_edit_page_model_title: str = "Model Name"
    platform_edit_page_model_content: str = "Current model in use: {MODEL}"
    platform_edit_page_model_edit: str = "Manual Input"
    platform_edit_page_model_sync: str = "Fetch Online"

    # 参数编辑
    args_edit_page_top_p_title: str = "top_p"
    args_edit_page_top_p_content: str = "Please set with caution, incorrect values may cause abnormal results or request errors"
    args_edit_page_temperature_title: str = "temperature"
    args_edit_page_temperature_content: str = "Please set with caution, incorrect values may cause abnormal results or request errors"
    args_edit_page_presence_penalty_title: str = "presence_penalty"
    args_edit_page_presence_penalty_content: str = "Please set with caution, incorrect values may cause abnormal results or request errors"
    args_edit_page_frequency_penalty_title: str = "frequency_penalty"
    args_edit_page_frequency_penalty_content: str = "Please set with caution, incorrect values may cause abnormal results or request errors"
    args_edit_page_document_link: str = "Click to view documentation"

    # 模型列表
    model_list_page_title: str = "Available Model List"
    model_list_page_content: str = "Click to select the model to use"
    model_list_page_fail: str = "Failed to get model list, please check API configuration ..."

    # 项目设置
    project_page_source_language_title: str = "Source Language"
    project_page_source_language_content: str = "Set the language of the source text used in the current translation project"
    project_page_source_language_items: str = "Chinese,English,Japanese,Korean,Russian"
    project_page_target_language_title: str = "Target Language"
    project_page_target_language_content: str = "Set the language of the translated text used in the current translation project"
    project_page_target_language_items: str = "Chinese,English,Japanese,Korean,Russian"
    project_page_input_folder_title: str = "Input Folder"
    project_page_input_folder_content: str = "Current input folder is"
    project_page_input_folder_btn: str = "Select Folder"
    project_page_output_folder_title: str = "Output Folder (cannot be the same as input folder)"
    project_page_output_folder_content: str = "Current output folder is"
    project_page_output_folder_btn: str = "Select Folder"
    project_page_traditional_chinese_title: str = "Output Chinese in Traditional Characters"
    project_page_traditional_chinese_content: str = "When enabled, if the target language is set to Chinese, Chinese text will be output in Traditional Chinese characters"

    # 开始翻译
    translation_page_status_idle = "Idle"
    translation_page_status_api_testing = "Testing"
    translation_page_status_translating = "Translating"
    translation_page_status_stoping = "Stopping"
    translation_page_indeterminate_saving = "Saving cache file ..."
    translation_page_indeterminate_stoping = "Stopping translation task ..."
    translation_page_card_time = "Elapsed Time"
    translation_page_card_remaining_time = "Remaining Time"
    translation_page_card_line = "Translated Lines"
    translation_page_card_remaining_line = "Remaining Lines"
    translation_page_card_speed = "Average Speed"
    translation_page_card_token = "Total Tokens"
    translation_page_card_task = "Real Time Tasks"
    translation_page_alert_start = "Unfinished translation tasks will be reset. Confirm to start a new translation task ...?"
    translation_page_alert_pause = "Stopped translation tasks can be resumed at any time. Confirm to stop the task ...?"
    translation_page_continue = "Continue Translation"
    translation_page_export = "Export Translation Data"
    translation_page_export_toast = "Translation files have been generated in the output folder based on the current translation data ..."

    # 基础设置
    basic_settings_page_batch_size_title = "Concurrent Tasks"
    basic_settings_page_batch_size_content = (
        "Maximum number of translation tasks executed simultaneously."
        + "\n" + "Setting appropriately can greatly increase translation speed. Please refer to the API platform's limits for settings."
    )

    basic_settings_page_task_token_limit_title = "Task Length Threshold"
    basic_settings_page_task_token_limit_content = "Maximum text length sent to the model at once for each translation task, unit is Token."
    basic_settings_page_request_timeout_title = "Request Timeout"
    basic_settings_page_request_timeout_content = (
        "Timeout duration for a model's response to a translation request."
        + "\n" + "If the model doesn't respond in time, the translation task will fail, unit is Seconds. Not applicable to Google models."
    )
    basic_settings_page_max_round_title = "Maximum Translation Rounds"
    basic_settings_page_max_round_content = "After one translation round, if entries are still untranslated, restart translation until finished or the round limit is reached."

    # 高级功能
    advance_feature_page_auto_glossary_enable_title = "Auto Complete Glossary (Experimental feature, SakuraLLM model not supported)"
    advance_feature_page_auto_glossary_enable_content = (
        "When enabled, text will be analyzed during translation to attempt to automatically complete missing proper noun entries in the glossary."
        + "\n" + "This feature is designed only for gap-filling and cannot replace manually created glossaries. It is only effective when **Glossary feature is enabled**."
        + "\n" + "May cause **negative effects** or **translation anomalies**. and it may have positive effects only on powerful models such as DeepSeek V3/R1."
        + "\n" + "Please **judge for yourself** whether to enable it."
    )
    advance_feature_page_mtool_optimizer_enable_title = "MTool Optimizer"
    advance_feature_page_mtool_optimizer_enable_content = (
        "When enabled, when translating MTool text, it can reduce translation time and token consumption by up to 40%."
        + "\n" + "May cause issues such as **original text residue** or **incoherent sentences**, and it should only be enabled when **translating MTool text**."
        + "\n" + "Please **judge for yourself** whether to enable it"
    )

    # 术语表
    glossary_page_head_title = "Glossary"
    glossary_page_head_content = "By building a glossary in the prompt to guide model translation, unified translation and correction of personal pronouns can be achieved"
    glossary_page_table_row_01 = "Original Text"
    glossary_page_table_row_02 = "Translated Text"
    glossary_page_table_row_03 = "Description"
    glossary_page_import = "Import"
    glossary_page_import_toast = "Data imported ..."
    glossary_page_export = "Export"
    glossary_page_export_toast = "Data exported to application root directory ..."
    glossary_page_add = "Add"
    glossary_page_add_toast = "New row added ..."
    glossary_page_save = "Save"
    glossary_page_save_toast = "Data saved ..."
    glossary_page_reset = "Reset"
    glossary_page_reset_toast = "Data reset ..."
    glossary_page_reset_alert = "Confirm reset to default data ...?"
    glossary_page_kg = "One-Click Tools"
    glossary_page_wiki = "Wiki"

    # 译前替换
    pre_translation_replacement_page_head_title = "Pre-translation Replacement"
    pre_translation_replacement_page_head_content = (
        "Before translation starts, replace the matched parts in the original text with the specified text, the execution order is from top to bottom.\n"
        + "When translating RPGMaker MV/MZ games, importing Actors.json files from the data or www\\data folder can significantly improve translation quality"
    )
    pre_translation_replacement_page_table_row_01 = "Original Text"
    pre_translation_replacement_page_table_row_02 = "Replacement"
    pre_translation_replacement_page_import = "Import"
    pre_translation_replacement_page_import_toast = "Data imported ..."
    pre_translation_replacement_page_export = "Export"
    pre_translation_replacement_page_export_toast = "Data exported to application root directory ..."
    pre_translation_replacement_page_add = "Add"
    pre_translation_replacement_page_add_toast = "New row added ..."
    pre_translation_replacement_page_save = "Save"
    pre_translation_replacement_page_save_toast = "Data saved ..."
    pre_translation_replacement_page_reset = "Reset"
    pre_translation_replacement_page_reset_toast = "Data reset ..."
    pre_translation_replacement_page_reset_alert = "Confirm reset to default data ...?"
    pre_translation_replacement_page_wiki = "Wiki"

    # 译后替换
    post_translation_replacement_page_head_title = "Post-translation Replacement"
    post_translation_replacement_page_head_content = "After translation is completed, replace the matched parts in the translated text with the specified text, the execution order is from top to bottom"
    post_translation_replacement_page_table_row_01 = "Original Text"
    post_translation_replacement_page_table_row_02 = "Replacement"
    post_translation_replacement_page_import = "Import"
    post_translation_replacement_page_import_toast = "Data imported ..."
    post_translation_replacement_page_export = "Export"
    post_translation_replacement_page_export_toast = "Data exported to application root directory ..."
    post_translation_replacement_page_add = "Add"
    post_translation_replacement_page_add_toast = "New row added ..."
    post_translation_replacement_page_save = "Save"
    post_translation_replacement_page_save_toast = "Data saved ..."
    post_translation_replacement_page_reset = "Reset"
    post_translation_replacement_page_reset_toast = "Data reset ..."
    post_translation_replacement_page_reset_alert = "Confirm reset to default data ...?"
    post_translation_replacement_page_wiki = "Wiki"

    # 自定义提示词 - 中文
    custom_prompt_zh_page_head_title = "Custom prompt used when target language is set to Chinese (SakuraLLM model not supported)"
    custom_prompt_zh_page_head_content = (
        "Add extra translation requirements such as story settings and writing style through custom prompts."
        + "Note: Prefix and suffix parts are fixed and unmodifiable. Custom prompts in this page will only be used when **target language is set to Chinese**."
    )
    custom_prompt_zh_page_save = "Save"
    custom_prompt_zh_page_save_toast = "Data saved ..."
    custom_prompt_zh_page_reset = "Reset"
    custom_prompt_zh_page_reset_toast = "Data reset ..."
    custom_prompt_zh_page_reset_alert = "Confirm reset to default data ...?"

    # 自定义提示词 - 英文
    custom_prompt_en_page_head_title = "Custom prompt used when target language is set to non-Chinese languages (SakuraLLM model not supported)"
    custom_prompt_en_page_head_content = (
        "Add extra translation requirements such as story settings and writing style through custom prompts.\n"
        + "Note: Prefix and suffix parts are fixed and unmodifiable. Custom prompts in this page will only be used when **target language is set to non-Chinese languages**."
    )
    custom_prompt_en_page_save = "Save"
    custom_prompt_en_page_save_toast = "Data saved ..."
    custom_prompt_en_page_reset = "Reset"
    custom_prompt_en_page_reset_toast = "Data reset ..."
    custom_prompt_en_page_reset_alert = "Confirm reset to default data ...?"