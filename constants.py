"""
Application constants and static configuration.
"""
from typing import Final

# Prompt templates for AI reply generation
PROMPT_TEMPLATES: Final[list[str]] = [
    'Read this Facebook post and comment carefully. Generate a relevant, helpful reply that addresses the comment in the context of the original post.\n\nOriginal Post: {post}\n\nComment: {comment}\n\nReply should be brief, friendly, and directly related to both the post and comment content.',
    'Analyze this Facebook post and the comment below it. Respond appropriately based on the comment content, keeping in mind the context of the original post.\n\nOriginal Post: {post}\n\nComment: {comment}\n\nIf it is a question, answer it. If it is a statement, acknowledge it meaningfully. Keep the reply short and relevant to both the post and comment.',
    'Generate a contextual reply to this comment, considering the original post it responds to.\n\nOriginal Post: {post}\n\nComment: {comment}\n\nMatch the tone and address the specific topic mentioned. Be genuine and concise.',
    'Respond to this comment in a way that shows you understand both the original post and the user\'s comment.\n\nOriginal Post: {post}\n\nComment: {comment}\n\nUse a friendly, conversational tone that directly relates to their message and the post context.',
    'Create a reply that is specifically tailored to this comment, considering the original post it responds to.\n\nOriginal Post: {post}\n\nComment: {comment}\n\nAddress any questions, concerns, or points raised. Be authentic and helpful.',
    # Fallback templates for when post content is not available
    'Read this Facebook comment carefully and generate a relevant, helpful reply that directly addresses what the user is asking or commenting about. Comment: {comment}. Reply should be brief, friendly, and directly related to the comment content.',
    'Analyze this comment and respond appropriately based on its content. If it is a question, answer it. If it is a statement, acknowledge it meaningfully. Comment: {comment}. Keep the reply short and relevant.',
    'Generate a contextual reply to this comment. Match the tone and address the specific topic mentioned. Comment: {comment}. Be genuine and concise.',
    'Respond to this comment in a way that shows you understand what the user is saying. Comment: {comment}. Use a friendly, conversational tone that directly relates to their message.',
    'Create a reply that is specifically tailored to this comment content. Address any questions, concerns, or points raised. Comment: {comment}. Be authentic and helpful.'
]

# Chrome launch arguments for better stealth
CHROME_LAUNCH_ARGS: Final[list[str]] = [
    '--disable-blink-features=AutomationControlled',
    '--disable-popup-blocking',
    '--disable-notifications',
    '--disable-dev-shm-usage',
    '--no-sandbox',
    '--disable-setuid-sandbox',
    '--disable-gpu',
    '--disable-software-rasterizer',
    '--disable-extensions',
    '--disable-sync',
    '--disable-background-networking',
    '--metrics-recording-only',
    '--mute-audio'
]

# Facebook selectors (updated for current Facebook structure)
FACEBOOK_SELECTORS: Final[dict[str, str]] = {
    'main_content': "//div[@role='main']",
    'comment_article': "//div[@role='article']",
    'comment_text': 'div[dir="auto"]',
    'author_name': 'span.x193iq5w[dir="auto"]',
    'reply_button': "//div[@role='button' and contains(text(), 'Reply')]",
    'reply_input': "p[@dir='auto' and contains(@class, 'xdj266r')]",
    'submit_button': "//div[@role='button' and @aria-label='Comment']",
}

# Cache limits
MAX_CACHE_SIZE: Final[int] = 100
CACHE_CLEANUP_SIZE: Final[int] = 20

# Retry configuration
DEFAULT_MAX_RETRIES: Final[int] = 3
DEFAULT_RETRY_DELAY: Final[float] = 1.0

# Timeout values (in seconds)
PAGE_LOAD_TIMEOUT: Final[int] = 30
ELEMENT_WAIT_TIMEOUT: Final[int] = 10
REPLY_BOX_WAIT_TIMEOUT: Final[int] = 5

