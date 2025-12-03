import streamlit as st

class I18N:

    # Dictionary of translations
    _DB = {
        "CN": {
            "app_title": "Project Codex v2.1",
            "sidebar_dashboard": "🏠 任务看板",
            "sidebar_config": "⚙️ 模组配置",
            "sidebar_kb": "📚 知识库",
            "sidebar_workbench": "📝 审阅工作台",
            "welcome_msg": "欢迎使用 Project Codex 智能稿件本地化系统。",
            "upload_header": "1. 上传与配置",
            "monitor_header": "2. 实时监控",
            "start_btn": "🚀 开始生产",
            "stop_btn": "停止所有任务",
            "config_router": "模型路由",
            "config_provider": "供应商配置",
            "config_prompts": "提示词工程",
            "config_profiles": "配置预设",
            "kb_glossary": "术语表管理",
            "kb_chars": "角色卡片",
            "wb_title": "审阅工作台",
            "save_btn": "💾 保存修改",
            "export_btn": "📥 导出结果",
            "god_mode": "开启上帝视角 (三栏)",
            "lang_select": "Language/语言"
        },
        "EN": {
            "app_title": "Project Codex v2.1",
            "sidebar_dashboard": "🏠 Dashboard",
            "sidebar_config": "⚙️ Config Studio",
            "sidebar_kb": "📚 Knowledge Base",
            "sidebar_workbench": "📝 Workbench",
            "welcome_msg": "Welcome to Project Codex Intelligent Localization System.",
            "upload_header": "1. Upload & Configure",
            "monitor_header": "2. Live Monitor",
            "start_btn": "🚀 Start Production",
            "stop_btn": "Stop All Tasks",
            "config_router": "Model Router",
            "config_provider": "Provider Config",
            "config_prompts": "Prompt Lab",
            "config_profiles": "Profiles",
            "kb_glossary": "Glossary",
            "kb_chars": "Character Sheets",
            "wb_title": "Editor Workbench",
            "save_btn": "💾 Save Changes",
            "export_btn": "📥 Export Result",
            "god_mode": "Enable God Mode (3-Col)",
            "lang_select": "Language/语言"
        }
    }

    @staticmethod
    def get(key: str) -> str:
        # Determine language from session state
        # Default to Chinese if not set
        current_lang = st.session_state.get("language", "中文")
        lang_code = "CN" if current_lang == "中文" else "EN"

        return I18N._DB.get(lang_code, {}).get(key, key)
