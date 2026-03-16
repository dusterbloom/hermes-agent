#!/usr/bin/env python3
"""
Tools Package

This package contains all the specific tool implementations for the Hermes Agent.
Each module provides specialized functionality for different capabilities:

- web_tools: Web search, content extraction, and crawling
- terminal_tool: Command execution using mini-swe-agent (local/docker/modal/daytona backends)
- vision_tools: Image analysis and understanding
- mixture_of_agents_tool: Multi-model collaborative reasoning
- image_generation_tool: Text-to-image generation with upscaling

The tools are imported into model_tools.py which provides a unified interface
for the AI agent to access all capabilities.

Each import block is wrapped in try/except so that missing optional packages
(firecrawl, fal_client, playwright, etc.) do not prevent the rest of the
tools package from loading.
"""

# Web tools (may need firecrawl, httpx, openai)
try:
    from .web_tools import (
        web_search_tool,
        web_extract_tool,
        web_crawl_tool,
        check_firecrawl_api_key
    )
except ImportError:
    pass

# Primary terminal tool (mini-swe-agent backend: local/docker/singularity/modal/daytona)
try:
    from .terminal_tool import (
        terminal_tool,
        check_terminal_requirements,
        cleanup_vm,
        cleanup_all_environments,
        get_active_environments_info,
        register_task_env_overrides,
        clear_task_env_overrides,
        TERMINAL_TOOL_DESCRIPTION
    )
except ImportError:
    pass

# Vision tools (may need httpx, openai)
try:
    from .vision_tools import (
        vision_analyze_tool,
        check_vision_requirements
    )
except ImportError:
    pass

# Mixture-of-agents tool (may need openrouter client)
try:
    from .mixture_of_agents_tool import (
        mixture_of_agents_tool,
        check_moa_requirements
    )
except ImportError:
    pass

# Image generation (needs fal_client)
try:
    from .image_generation_tool import (
        image_generate_tool,
        check_image_generation_requirements
    )
except ImportError:
    pass

# Skills tools
try:
    from .skills_tool import (
        skills_list,
        skill_view,
        check_skills_requirements,
        SKILLS_TOOL_DESCRIPTION
    )
except ImportError:
    pass

# Skill management tool
try:
    from .skill_manager_tool import (
        skill_manage,
        check_skill_manage_requirements,
        SKILL_MANAGE_SCHEMA
    )
except ImportError:
    pass

# Browser automation tools (agent-browser + Browserbase; may need playwright/requests)
try:
    from .browser_tool import (
        browser_navigate,
        browser_snapshot,
        browser_click,
        browser_type,
        browser_scroll,
        browser_back,
        browser_press,
        browser_close,
        browser_get_images,
        browser_vision,
        cleanup_browser,
        cleanup_all_browsers,
        get_active_browser_sessions,
        check_browser_requirements,
        BROWSER_TOOL_SCHEMAS
    )
except ImportError:
    pass

# Cronjob management tools (CLI-only, hermes-cli toolset)
try:
    from .cronjob_tools import (
        cronjob,
        schedule_cronjob,
        list_cronjobs,
        remove_cronjob,
        check_cronjob_requirements,
        get_cronjob_tool_definitions,
        CRONJOB_SCHEMA,
    )
except ImportError:
    pass

# RL Training tools (Tinker-Atropos; may need yaml)
try:
    from .rl_training_tool import (
        rl_list_environments,
        rl_select_environment,
        rl_get_current_config,
        rl_edit_config,
        rl_start_training,
        rl_check_status,
        rl_stop_training,
        rl_get_results,
        rl_list_runs,
        rl_test_inference,
        check_rl_api_keys,
        get_missing_keys,
    )
except ImportError:
    pass

# File manipulation tools (read, write, patch, search)
try:
    from .file_tools import (
        read_file_tool,
        write_file_tool,
        patch_tool,
        search_tool,
        get_file_tools,
        clear_file_ops_cache,
    )
except ImportError:
    pass

# Text-to-speech tools (Edge TTS / ElevenLabs / OpenAI; may need edge_tts/elevenlabs)
try:
    from .tts_tool import (
        text_to_speech_tool,
        check_tts_requirements,
    )
except ImportError:
    pass

# Planning & task management tool
try:
    from .todo_tool import (
        todo_tool,
        check_todo_requirements,
        TODO_SCHEMA,
        TodoStore,
    )
except ImportError:
    pass

# Clarifying questions tool (interactive Q&A with the user)
try:
    from .clarify_tool import (
        clarify_tool,
        check_clarify_requirements,
        CLARIFY_SCHEMA,
    )
except ImportError:
    pass

# Code execution sandbox (programmatic tool calling)
try:
    from .code_execution_tool import (
        execute_code,
        check_sandbox_requirements,
        EXECUTE_CODE_SCHEMA,
    )
except ImportError:
    pass

# Subagent delegation (spawn child agents with isolated context)
try:
    from .delegate_tool import (
        delegate_task,
        check_delegate_requirements,
        DELEGATE_TASK_SCHEMA,
    )
except ImportError:
    pass

# File tools have no external requirements - they use the terminal backend
def check_file_requirements():
    """File tools only require terminal backend to be available."""
    from .terminal_tool import check_terminal_requirements
    return check_terminal_requirements()

__all__ = [
    # Web tools
    'web_search_tool',
    'web_extract_tool',
    'web_crawl_tool',
    'check_firecrawl_api_key',
    # Terminal tools (mini-swe-agent backend)
    'terminal_tool',
    'check_terminal_requirements',
    'cleanup_vm',
    'cleanup_all_environments',
    'get_active_environments_info',
    'register_task_env_overrides',
    'clear_task_env_overrides',
    'TERMINAL_TOOL_DESCRIPTION',
    # Vision tools
    'vision_analyze_tool',
    'check_vision_requirements',
    # MoA tools
    'mixture_of_agents_tool',
    'check_moa_requirements',
    # Image generation tools
    'image_generate_tool',
    'check_image_generation_requirements',
    # Skills tools
    'skills_list',
    'skill_view',
    'check_skills_requirements',
    'SKILLS_TOOL_DESCRIPTION',
    # Skill management
    'skill_manage',
    'check_skill_manage_requirements',
    'SKILL_MANAGE_SCHEMA',
    # Browser automation tools
    'browser_navigate',
    'browser_snapshot',
    'browser_click',
    'browser_type',
    'browser_scroll',
    'browser_back',
    'browser_press',
    'browser_close',
    'browser_get_images',
    'browser_vision',
    'cleanup_browser',
    'cleanup_all_browsers',
    'get_active_browser_sessions',
    'check_browser_requirements',
    'BROWSER_TOOL_SCHEMAS',
    # Cronjob management tools (CLI-only)
    'cronjob',
    'schedule_cronjob',
    'list_cronjobs',
    'remove_cronjob',
    'check_cronjob_requirements',
    'get_cronjob_tool_definitions',
    'CRONJOB_SCHEMA',
    # RL Training tools
    'rl_list_environments',
    'rl_select_environment',
    'rl_get_current_config',
    'rl_edit_config',
    'rl_start_training',
    'rl_check_status',
    'rl_stop_training',
    'rl_get_results',
    'rl_list_runs',
    'rl_test_inference',
    'check_rl_api_keys',
    'get_missing_keys',
    # File manipulation tools
    'read_file_tool',
    'write_file_tool',
    'patch_tool',
    'search_tool',
    'get_file_tools',
    'clear_file_ops_cache',
    'check_file_requirements',
    # Text-to-speech tools
    'text_to_speech_tool',
    'check_tts_requirements',
    # Planning & task management tool
    'todo_tool',
    'check_todo_requirements',
    'TODO_SCHEMA',
    'TodoStore',
    # Clarifying questions tool
    'clarify_tool',
    'check_clarify_requirements',
    'CLARIFY_SCHEMA',
    # Code execution sandbox
    'execute_code',
    'check_sandbox_requirements',
    'EXECUTE_CODE_SCHEMA',
    # Subagent delegation
    'delegate_task',
    'check_delegate_requirements',
    'DELEGATE_TASK_SCHEMA',
]
