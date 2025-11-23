"""
Configuration management module.
Handles all configuration loading, validation, and defaults.
"""
import os
from typing import TypedDict, Optional
from dotenv import load_dotenv

load_dotenv()


class DelaysConfig(TypedDict):
    """Configuration for various delay timings."""
    SHORT_MIN: float
    SHORT_MAX: float
    MEDIUM_MIN: float
    MEDIUM_MAX: float
    LONG_MIN: float
    LONG_MAX: float
    RELOAD_PAUSE: float
    SCAN_INTERVAL: float


class BotConfig(TypedDict):
    """Main bot configuration."""
    POST_URL: Optional[str]
    MAX_REPLIES: int
    MAX_ITERATIONS: int
    MY_NAME: str
    DELAYS: DelaysConfig
    CHROME_PROFILE: str
    MAX_RETRIES: int
    COOLDOWN_SECONDS: int
    MAX_REPLIES_PER_USER: int
    RATE_LIMIT_WINDOW: int
    MAX_REPLIES_PER_WINDOW: int
    REPLY_TO_THREADS: bool
    HEADLESS: bool
    RUN_CONTINUOUSLY: bool


class SupabaseConfig(TypedDict):
    """Supabase database configuration."""
    URL: Optional[str]
    ANON_KEY: Optional[str]


class OpenAIConfig(TypedDict):
    """OpenAI API configuration."""
    API_KEY: Optional[str]
    MODEL: str
    PROMPT: str
    TEMPERATURE: float
    MAX_TOKENS: int


def get_bot_config() -> BotConfig:
    """Get and validate bot configuration."""
    return {
        'POST_URL': os.getenv('POST_URL'),
        'MAX_REPLIES': int(os.getenv('MAX_REPLIES', '100')),
        'MAX_ITERATIONS': int(os.getenv('MAX_ITERATIONS', '10000')),
        'MY_NAME': os.getenv('FB_OUR_NAME', 'Danny Nguyen'),
        'DELAYS': {
            'SHORT_MIN': float(os.getenv('DELAY_SHORT_MIN', '0.5')),
            'SHORT_MAX': float(os.getenv('DELAY_SHORT_MAX', '2.0')),
            'MEDIUM_MIN': float(os.getenv('DELAY_MEDIUM_MIN', '2.0')),
            'MEDIUM_MAX': float(os.getenv('DELAY_MEDIUM_MAX', '5.0')),
            'LONG_MIN': float(os.getenv('DELAY_LONG_MIN', '5.0')),
            'LONG_MAX': float(os.getenv('DELAY_LONG_MAX', '20.0')),
            'RELOAD_PAUSE': float(os.getenv('RELOAD_PAUSE', '180.0')),
            'SCAN_INTERVAL': float(os.getenv('SCAN_INTERVAL', '15.0')),
        },
        'CHROME_PROFILE': os.getenv('CHROME_PROFILE', 'Default'),
        'MAX_RETRIES': int(os.getenv('MAX_RETRIES', '3')),
        'COOLDOWN_SECONDS': int(os.getenv('COOLDOWN_SECONDS', '0')),
        'MAX_REPLIES_PER_USER': int(os.getenv('MAX_REPLIES_PER_USER', '9999')),
        'RATE_LIMIT_WINDOW': int(os.getenv('RATE_LIMIT_WINDOW', '300')),
        'MAX_REPLIES_PER_WINDOW': int(os.getenv('MAX_REPLIES_PER_WINDOW', '999')),
        'REPLY_TO_THREADS': os.getenv('REPLY_TO_THREADS', 'true').lower() == 'true',
        'HEADLESS': os.getenv('HEADLESS', 'false').lower() == 'true',
        'RUN_CONTINUOUSLY': os.getenv('RUN_CONTINUOUSLY', 'true').lower() == 'true',
    }


def get_supabase_config() -> SupabaseConfig:
    """Get and validate Supabase configuration."""
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_ANON_KEY')

    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in environment")

    return {
        'URL': url,
        'ANON_KEY': key,
    }


def get_openai_config() -> OpenAIConfig:
    """Get and validate OpenAI configuration."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be set in environment")

    return {
        'API_KEY': api_key,
        'MODEL': os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
        'PROMPT': os.getenv('OPENAI_PROMPT', 'Generate a helpful, friendly reply to this comment: ') +
                  ' Do not include emojis or any introductory phrases or additional text.',
        'TEMPERATURE': float(os.getenv('OPENAI_TEMPERATURE', '0.8')),
        'MAX_TOKENS': int(os.getenv('OPENAI_MAX_TOKENS', '150')),
    }

