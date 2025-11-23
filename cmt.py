"""
Facebook AI Reply Bot - Production-Grade Implementation

This module implements a sophisticated Facebook comment reply bot with:
- Robust error handling and retry logic
- Database-backed comment tracking
- AI-powered reply generation
- Anti-detection mechanisms
- Comprehensive logging and monitoring

Author: Senior Engineering Team
"""
import os
import random
import time
import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Tuple, Any, TypedDict, Optional, cast
from contextlib import contextmanager
from playwright.sync_api import sync_playwright, Locator, Page, BrowserContext, Playwright, FloatRect
from openai import OpenAI

# Local imports
from database import SupabaseDatabase
from config import get_bot_config, get_supabase_config, get_openai_config
from constants import (
    PROMPT_TEMPLATES,
    CHROME_LAUNCH_ARGS,
    FACEBOOK_SELECTORS,
    MAX_CACHE_SIZE,
    CACHE_CLEANUP_SIZE,
    DEFAULT_MAX_RETRIES,
    PAGE_LOAD_TIMEOUT,
    ELEMENT_WAIT_TIMEOUT,
)
from utils.logger import setup_logger
from utils.retry import retry_with_backoff
from utils.validators import validate_comment_id, validate_user_id, sanitize_text, validate_url

class DelaysConfig(TypedDict):
    SHORT_MIN: float
    SHORT_MAX: float
    MEDIUM_MIN: float
    MEDIUM_MAX: float
    LONG_MIN: float
    LONG_MAX: float
    RELOAD_PAUSE: float
    SCAN_INTERVAL: float

class BotConfig(TypedDict):
    POST_URL: str | None
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

class SupabaseConfigDict(TypedDict):
    URL: str | None
    ANON_KEY: str | None

class OpenAIConfigDict(TypedDict):
    API_KEY: str | None
    MODEL: str
    PROMPT: str
    TEMPERATURE: float
    MAX_TOKENS: int

class CommentData(TypedDict):
    id: str
    user_id: str
    author: str | None
    text: str
    element: Locator
    reply_button: Locator | None
    timestamp: Optional[datetime]
    priority: int
    detected_at: datetime
    comment_type: str

class RepliedComments(TypedDict):
    replied: list[str]
    user_replies: dict[str, int]

class Statistics(TypedDict):
    total_comments: int
    unique_users: int
    successful: int
    failed: int

class CommentQueue(TypedDict):
    pending: list[CommentData]
    processing: list[CommentData]
    completed: list[str]
    failed: list[str]

class DetectionMetrics(TypedDict):
    total_scans: int
    comments_detected: int
    new_comments_found: int
    processing_time_avg: float
    last_scan_duration: float

# Initialize logger
logger = setup_logger(__name__)
class FacebookAIReplyBot:
    """
    Production-grade Facebook AI Reply Bot.

    Features:
    - Database-backed comment tracking
    - AI-powered contextual replies
    - Anti-detection mechanisms
    - Robust error handling and retries
    - Comprehensive logging and monitoring
    """

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        """
        Initialize the Facebook AI Reply Bot.

        Args:
            config: Optional configuration overrides

        Raises:
            ValueError: If required configuration is missing
        """
        # Load and merge configuration
        base_config = get_bot_config()
        if config:
            base_config.update(config)
        self.config: BotConfig = cast(BotConfig, base_config)

        # Validate critical configuration
        if not validate_url(self.config.get('POST_URL')):
            raise ValueError("POST_URL must be a valid URL")

        # Initialize browser components
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[BrowserContext] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

        # Initialize data structures
        self.comment_queue: CommentQueue = {
            'pending': [],
            'processing': [],
            'completed': [],
            'failed': []
        }

        self.detection_metrics: DetectionMetrics = {
            'total_scans': 0,
            'comments_detected': 0,
            'new_comments_found': 0,
            'processing_time_avg': 0.0,
            'last_scan_duration': 0.0
        }

        self.known_comment_elements: set[str] = set()
        self.last_comment_count: int = 0
        self.last_full_scan: datetime = datetime.now()
        self.session_processed_ids: set[str] = set()
        self.adaptive_scan_interval: float = self.config['DELAYS']['SCAN_INTERVAL']
        self.last_activity_time: datetime = datetime.now()

        # Initialize database
        try:
            supabase_config = get_supabase_config()
            self.db = SupabaseDatabase(
                supabase_url=supabase_config['URL'] or '',
                supabase_key=supabase_config['ANON_KEY'] or ''
            )
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

        # Initialize response cache with size limit
        self.response_cache: dict[str, str] = {}

        # Initialize OpenAI client
        try:
            openai_config = get_openai_config()
            if not openai_config.get('API_KEY'):
                raise ValueError("OPENAI_API_KEY must be set in environment")
            self.openai_client = OpenAI(api_key=openai_config['API_KEY'])
            self.openai_config = openai_config  # Store for later use
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

        # Log startup
        self.db.log_event('bot_startup', 'Bot initialized with Playwright + Supabase', 'info')
        logger.info("✅ Facebook AI Reply Bot initialized successfully")

        # Bot identification
        self.bot_user_id: Optional[str] = None
        self.post_content_cache: Optional[str] = None

    def is_own_comment(self, author: str | None, user_id: str | None = None) -> bool:
        """Check if a comment is from the bot itself using case-insensitive name comparison and user_id."""
        if not author:
            return False

        # Normalize author name for comparison (case-insensitive, trimmed)
        bot_name_normalized = (self.config['MY_NAME'] or '').strip().lower()
        author_normalized = author.strip().lower()

        # Check by name (case-insensitive)
        if bot_name_normalized and author_normalized == bot_name_normalized:
            return True

        # Check by user_id if we have the bot's user_id stored
        if self.bot_user_id and user_id and user_id == self.bot_user_id:
            return True

        return False

    def is_user_in_queue(self, user_id: str) -> bool:
        """Check if a user is already in the pending or processing queue."""
        # Check pending queue
        for comment in self.comment_queue['pending']:
            if comment.get('user_id') == user_id:
                return True

        # Check processing queue
        for comment in self.comment_queue['processing']:
            if comment.get('user_id') == user_id:
                return True

        return False

    def is_comment_id_in_queue(self, comment_id: str) -> bool:
        """Check if a comment_id is already in any part of the queue."""
        # Check pending queue
        for comment in self.comment_queue['pending']:
            if comment.get('id') == comment_id:
                return True

        # Check processing queue
        for comment in self.comment_queue['processing']:
            if comment.get('id') == comment_id:
                return True

        # Check completed/failed lists
        if comment_id in self.comment_queue['completed']:
            return True
        if comment_id in self.comment_queue['failed']:
            return True

        return False

    def is_browser_alive(self) -> bool:
        """Check if browser and page are still open and accessible."""
        try:
            if not self.browser or not self.page:
                return False

            # Try to check if page is still accessible
            try:
                # Quick check - try to get page URL (non-blocking check)
                _ = self.page.url
                return True
            except Exception:
                return False
        except Exception:
            return False

    def ensure_browser_alive(self) -> bool:
        """Ensure browser is alive, reopen if necessary."""
        if self.is_browser_alive():
            return True

        logger.warning("⚠️  Browser/page was closed unexpectedly, attempting to reopen...")

        try:
            # Close any stale connections
            try:
                if self.browser:
                    self.browser.close()
            except:
                pass

            try:
                if self.playwright:
                    self.playwright.stop()
            except:
                pass

            # Reinitialize browser
            self.setup_driver()

            # Reload the post page
            post_url = self.config.get('POST_URL')
            if post_url and self.page:
                logger.info(f"🔄 Reloading post page: {post_url}")
                self.page.goto(post_url, wait_until='domcontentloaded', timeout=PAGE_LOAD_TIMEOUT * 1000)
                self.random_pause(2, 4)
                self.setup_dialog_prevention()

                # Re-extract post content
                self.post_content_cache = None
                self.get_post_content()

                logger.info("✅ Browser recovered successfully")
                return True
            else:
                logger.error("❌ Cannot recover: POST_URL not configured")
                return False

        except Exception as e:
            logger.error(f"❌ Failed to recover browser: {e}")
            return False

    def setup_dialog_prevention(self) -> None:
        """Inject JavaScript to prevent beforeunload dialogs and handle them automatically."""
        if not self.page:
            return

        try:
            # Inject JavaScript to prevent and handle beforeunload dialogs
            self.page.evaluate("""
                (function() {
                    // Prevent beforeunload dialogs
                    window.addEventListener('beforeunload', function(e) {
                        // Cancel the event
                        e.preventDefault();
                        e.returnValue = '';
                        return '';
                    }, { capture: true });

                    // Override window.onbeforeunload
                    window.onbeforeunload = null;

                    // Monitor for dialog appearance and auto-click Stay
                    setInterval(function() {
                        // Look for Leave Page dialog
                        const dialogs = document.querySelectorAll('[role="dialog"], [role="alertdialog"]');
                        for (let dialog of dialogs) {
                            const dialogText = dialog.textContent || '';
                            if (dialogText.includes('Leave Page') || dialogText.includes("You haven't finished")) {
                                // Find Stay button
                                const buttons = dialog.querySelectorAll('div[role="button"], button');
                                for (let btn of buttons) {
                                    const btnText = (btn.textContent || '').trim();
                                    if (btnText.includes('Stay') || btnText.includes('Stay on Page')) {
                                        console.log('Auto-clicking Stay button');
                                        btn.click();
                                        return;
                                    }
                                }
                            }
                        }
                    }, 200);
                })();
            """)
            logger.info("✅ Dialog prevention JavaScript injected")
        except Exception as e:
            logger.warning(f"Could not inject dialog prevention: {e}")

    def handle_leave_page_dialog(self) -> bool:
        """Check for and handle the 'Leave Page?' dialog by clicking 'Stay on Page'.
        Uses multiple methods including JavaScript injection for reliability.
        Returns True if dialog was found and handled, False otherwise."""
        if not self.page:
            return False

        try:
            # Method 1: Use JavaScript to find and click the Stay button directly
            try:
                result = self.page.evaluate("""
                    (function() {
                        // Find all dialogs
                        const dialogs = document.querySelectorAll('[role="dialog"], [role="alertdialog"], div[class*="dialog"], div[class*="modal"]');
                        for (let dialog of dialogs) {
                            const text = (dialog.textContent || '').toLowerCase();
                            if (text.includes('leave page') || text.includes("you haven't finished")) {
                                // Find Stay button within this dialog
                                const buttons = dialog.querySelectorAll('div[role="button"], button, [class*="button"]');
                                for (let btn of buttons) {
                                    const btnText = (btn.textContent || '').trim().toLowerCase();
                                    if (btnText.includes('stay') && !btnText.includes('leave')) {
                                        try {
                                            btn.click();
                                            return true;
                                        } catch (e) {
                                            // Try alternative click method
                                            if (btn.dispatchEvent) {
                                                btn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
                                                return true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        return false;
                    })();
                """)
                if result:
                    logger.info("🔔 Found and clicked 'Stay on Page' button via JavaScript")
                    time.sleep(0.3)
                    return True
            except Exception as e:
                logger.debug(f"JavaScript dialog handling failed: {e}")

            # Method 2: Try multiple XPath selectors to find the "Stay on Page" button
            stay_selectors = [
                # More specific selectors for Facebook dialogs
                "//div[@role='dialog']//div[@role='button' and contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'stay')]",
                "//div[@role='alertdialog']//div[@role='button' and contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'stay')]",
                "//div[contains(@class, 'dialog')]//div[@role='button' and contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'stay')]",
                "//div[contains(text(), 'Leave Page')]//ancestor::div[@role='dialog']//div[@role='button'][contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'stay')]",
                "//div[contains(text(), 'You haven't finished')]//ancestor::div[@role='dialog']//div[@role='button'][contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'stay')]",
                # Fallback selectors
                "//div[@role='button' and contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'stay on page')]",
                "//div[@role='button' and contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'stay')]",
                "//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'stay')]",
            ]

            for selector in stay_selectors:
                try:
                    stay_button = self.page.locator(selector).first
                    if stay_button.count() > 0:
                        # Check if visible using JavaScript for more reliability
                        is_visible = stay_button.evaluate("el => el.offsetParent !== null && window.getComputedStyle(el).display !== 'none'")
                        if is_visible:
                            logger.info("🔔 Found 'Leave Page?' dialog - clicking 'Stay on Page'")
                            # Try JavaScript click first (more reliable)
                            try:
                                stay_button.evaluate("el => el.click()")
                            except:
                                stay_button.click(timeout=2000)
                            time.sleep(0.3)
                            return True
                except Exception as e:
                    logger.debug(f"Selector {selector[:50]}... failed: {e}")
                    continue

            # Method 3: Find dialog by text and then find button within it
            try:
                dialog_text_selectors = [
                    "//div[contains(text(), 'Leave Page')]",
                    "//div[contains(text(), 'You haven't finished')]",
                    "//*[contains(text(), 'Leave Page?')]",
                ]

                for text_selector in dialog_text_selectors:
                    dialog_elements = self.page.locator(text_selector).all()
                    for dialog_elem in dialog_elements:
                        try:
                            # Find the dialog container
                            dialog_container = dialog_elem.locator('xpath=./ancestor::div[@role="dialog" or @role="alertdialog" or contains(@class, "dialog")]').first
                            if dialog_container.count() > 0:
                                # Find Stay button within container
                                stay_buttons = dialog_container.locator("//div[@role='button' | //button").all()
                                for btn in stay_buttons:
                                    btn_text = (btn.text_content() or '').trim().lower()
                                    if 'stay' in btn_text and 'leave' not in btn_text:
                                        logger.info("🔔 Found Stay button in dialog container")
                                        btn.evaluate("el => el.click()")
                                        time.sleep(0.3)
                                        return True
                        except:
                            continue
            except Exception as e:
                logger.debug(f"Dialog container search failed: {e}")

        except Exception as e:
            logger.debug(f"Error checking for Leave Page dialog: {e}")

        return False

    def load_replied_comments(self):

        logger.warning('load_replied_comments is deprecated. Using Supabase database now.')
        return cast(RepliedComments, {'replied': [], 'user_replies': {}})
    def save_replied_comments(self):

        logger.debug('save_replied_comments is deprecated. Using Supabase database now.')
    def setup_driver(self):

        try:

            self.playwright = sync_playwright().start()

            chrome_path = os.getenv('CHROME_BINARY_PATH', '').strip().strip('"')

            launch_args = [
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

            headless = self.config.get('HEADLESS', False)
            if headless:
                logger.info('Running in HEADLESS mode (background)')
            else:
                logger.info('Running with visible browser')

            user_data_dir = os.path.join(os.getcwd(), 'facebook_user_data')
            os.makedirs(user_data_dir, exist_ok=True)
            logger.info(f'Using persistent user data directory: {user_data_dir}')

            if chrome_path and os.path.exists(chrome_path):
                logger.info(f'Using portable Chrome at: {chrome_path}')
                self.browser = self.playwright.chromium.launch_persistent_context(
                    user_data_dir,
                    executable_path=chrome_path,
                    headless=headless,
                    args=launch_args,
                    channel=None,
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    viewport={'width': 1920, 'height': 1080},
                    locale='en-US',
                    timezone_id='America/New_York',
                    permissions=['geolocation'],
                    extra_http_headers={
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Sec-Fetch-Dest': 'document',
                        'Sec-Fetch-Mode': 'navigate',
                        'Sec-Fetch-Site': 'none',
                        'Sec-Fetch-User': '?1',
                        'Upgrade-Insecure-Requests': '1'
                    }
                )
                self.context = self.browser
                logger.info('✓ Using persistent context - login will be saved!')
            else:
                logger.info('Using Playwright bundled Chromium')
                self.browser = self.playwright.chromium.launch_persistent_context(
                    user_data_dir,
                    headless=headless,
                    args=launch_args,
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    viewport={'width': 1920, 'height': 1080},
                    locale='en-US',
                    timezone_id='America/New_York',
                    permissions=['geolocation'],
                    extra_http_headers={
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Sec-Fetch-Dest': 'document',
                        'Sec-Fetch-Mode': 'navigate',
                        'Sec-Fetch-Site': 'none',
                        'Sec-Fetch-User': '?1',
                        'Upgrade-Insecure-Requests': '1'
                    }
                )
                self.context = self.browser
                logger.info('✓ Using persistent context - login will be saved!')

            anti_detection_script = """
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
                Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
                window.chrome = {runtime: {}};
                Object.defineProperty(navigator, 'permissions', {get: () => ({query: () => Promise.resolve({state: 'granted'})})});
            """

            if self.context:
                for page in self.context.pages:
                    page.add_init_script(script=anti_detection_script)

            if self.context and len(self.context.pages) > 0:
                self.page = self.context.pages[0]
            elif self.context:
                self.page = self.context.new_page()

            # Set up dialog handler to prevent ProtocolError
            if self.page:
                def handle_dialog(dialog):
                    """Handle JavaScript dialogs gracefully."""
                    try:
                        if dialog.type == 'beforeunload':
                            # For beforeunload, we want to stay on page
                            dialog.dismiss()
                        else:
                            # For other dialogs, accept them
                            dialog.accept()
                    except Exception as e:
                        # If dialog handling fails (e.g., session closed), just log it
                        logger.debug(f"Dialog handling failed (may be expected): {e}")

                self.page.on("dialog", handle_dialog)

            if self.page is not None:
                self.page.add_init_script(script=anti_detection_script)

            if self.page is not None:
                self.page.route("**/*", lambda route: (
                route.abort() if (

                    '/ads/' in route.request.url or
                    'doubleclick.net' in route.request.url or
                    'googlesyndication' in route.request.url or

                    (route.request.resource_type == "image" and int(route.request.headers.get('content-length', '0') or 0) > 100000) or
                    route.request.resource_type == "media" or

                    'analytics' in route.request.url or
                    'pixel' in route.request.url
                ) else route.continue_()
            ))

            logger.info('✓ Selective resource blocking enabled (faster load, Facebook still works)')
            logger.info('Playwright browser setup successfully.')
        except Exception as e:
            logger.error(f'Failed to setup Playwright browser: {e}')
            raise
    def random_pause(self, min_time: float = 1, max_time: float = 5) -> None:

        base_delay = random.uniform(min_time, max_time)

        if random.random() < 0.15:
            base_delay += random.uniform(0.1, 0.5)
        time.sleep(base_delay)
        logger.debug(f'Paused for {base_delay:.2f} seconds.')
    def human_mouse_jiggle(self, element: Locator, moves: int = 2) -> None:

        if not self.page:
            return
        try:

            box: Optional[FloatRect] = element.bounding_box()
            if box:

                center_x = box['x'] + box['width'] / 2
                center_y = box['y'] + box['height'] / 2
                self.page.mouse.move(center_x, center_y)

                for _ in range(moves):
                    x_offset = random.randint(-15, 15)
                    y_offset = random.randint(-15, 15)
                    self.page.mouse.move(center_x + x_offset, center_y + y_offset)
                    self.random_pause(0.3, 1)

                self.page.mouse.move(center_x, center_y)
                self.random_pause(0.3, 1)
                logger.debug(f'Performed mouse jiggle with {moves} moves.')
        except Exception as e:
            logger.debug(f'Mouse jiggle failed: {e}')
    def human_type(self, element: Locator, text: str) -> None:

        element.click()
        time.sleep(0.2)

        words = text.split()
        for w_i, word in enumerate(words):

            if random.random() < 0.05:
                fake_word = random.choice(['aaa', 'zzz', 'hmm'])
                for c in fake_word:
                    element.type(c, delay=random.uniform(80, 350))
                for _ in fake_word:
                    element.press('Backspace')
                    time.sleep(random.uniform(0.06, 0.25))

            for char in word:
                if random.random() < 0.05:
                    wrong_char = random.choice('abcdefghijklmnopqrstuvwxyz')
                    element.type(wrong_char, delay=random.uniform(80, 350))
                    time.sleep(random.uniform(0.06, 0.15))
                    element.press('Backspace')
                    time.sleep(random.uniform(0.06, 0.25))
                element.type(char, delay=random.uniform(80, 350))

            if w_i < len(words) - 1:
                element.type(' ', delay=random.uniform(80, 300))

            if random.random() < 0.03:
                element.press('ArrowLeft')
                time.sleep(random.uniform(0.1, 0.3))
                element.press('ArrowRight')
                time.sleep(random.uniform(0.1, 0.3))

        self.random_pause(0.5, 1.5)
        logger.debug('Completed human-like typing.')
    def close_all_reply_boxes(self) -> None:
        """Close all open reply boxes to avoid confusion."""
        if not self.is_browser_alive() or not self.page:
            logger.debug("Browser not alive, skipping close_all_reply_boxes")
            return

        try:
            logger.info('🔒 Closing any open reply boxes...')
            # Use JavaScript to close all reply boxes
            closed_count = self.page.evaluate("""
                (function() {
                    let closed = 0;
                    // Find all visible reply boxes
                    const replyBoxes = document.querySelectorAll('p[dir="auto"][class*="xdj266r"], div[contenteditable="true"]');

                    for (const box of replyBoxes) {
                        const style = window.getComputedStyle(box);
                        if (style.display === 'none' || style.visibility === 'hidden') continue;

                        // Check if this is a reply box (not main comment box)
                        const ariaLabel = box.getAttribute('aria-label') || '';
                        if (ariaLabel.toLowerCase().includes('write a comment')) continue;

                        // Try to find and click cancel/close button
                        let current = box;
                        for (let i = 0; i < 5; i++) {
                            current = current.parentElement;
                            if (!current) break;

                            // Look for close/cancel buttons
                            const closeButtons = current.querySelectorAll('[role="button"]');
                            for (const btn of closeButtons) {
                                const btnText = (btn.textContent || '').toLowerCase();
                                if (btnText.includes('cancel') || btnText.includes('close') || btnText.includes('×')) {
                                    btn.click();
                                    closed++;
                                    break;
                                }
                            }
                        }
                    }

                    // Also press Escape to close any modals
                    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));

                    return closed;
                })();
            """)

            # Also press Escape multiple times as fallback (only if browser is still alive)
            if self.is_browser_alive() and self.page:
                try:
                    for _ in range(3):
                        if self.is_browser_alive():
                            self.page.keyboard.press('Escape')
                            time.sleep(0.2)
                        else:
                            break
                except Exception as e:
                    if 'closed' not in str(e).lower():
                        logger.debug(f"Error pressing Escape: {e}")

            if closed_count > 0:
                logger.info(f"✅ Closed {closed_count} reply boxes via JavaScript")
        except Exception as e:
            error_msg = str(e)
            if 'closed' in error_msg.lower() or 'Target' in error_msg:
                logger.warning(f"Browser closed during close_all_reply_boxes: {e}")
                # Don't raise, just return - caller should handle browser recovery
            else:
                logger.debug(f"Error in close_all_reply_boxes: {e}")
            time.sleep(0.3)
        except Exception as e:
            logger.debug(f'Error closing reply boxes: {e}')
    def random_scroll(self):

        if not self.page:
            return
        scroll_direction = random.choice(['up', 'down'])
        scroll_distance = random.randint(200, 800)

        if random.random() < 0.3:

            chunks = random.randint(2, 4)
            chunk_size = scroll_distance // chunks
            for i in range(chunks):
                if scroll_direction == 'down':
                    self.page.evaluate(f'window.scrollBy(0, {chunk_size})')
                else:
                    self.page.evaluate(f'window.scrollBy(0, -{chunk_size})')
                time.sleep(random.uniform(0.2, 0.5))
                logger.debug(f'Scrolling {scroll_direction} {chunk_size}px (chunk {i+1}/{chunks})')
        else:
            if scroll_direction == 'down':
                self.page.evaluate(f'window.scrollBy(0, {scroll_distance})')
                logger.debug(f'Scrolling down {scroll_distance} pixels.')
            else:
                self.page.evaluate(f'window.scrollBy(0, -{scroll_distance})')
                logger.debug(f'Scrolling up {scroll_distance} pixels.')

        self.random_pause(1, 3)

    def simulate_human_activity(self):

        if not self.page:
            return
        activity = random.choice(['scroll', 'mouse_move', 'pause', 'none'])

        if activity == 'scroll':
            logger.debug('Anti-detection: Performing random scroll')
            self.random_scroll()

        elif activity == 'mouse_move':
            logger.debug('Anti-detection: Performing random mouse movement')
            try:

                x = random.randint(100, 1800)
                y = random.randint(100, 900)
                self.page.mouse.move(x, y)
                time.sleep(random.uniform(0.1, 0.3))
            except:
                pass

        elif activity == 'pause':
            logger.debug('Anti-detection: Pausing (reading simulation)')
            time.sleep(random.uniform(2, 5))

    def parse_comment_timestamp(self, comment_element: Locator) -> Optional[datetime]:

        try:

            timestamp_selectors = [
                ".//a[contains(@href, 'comment_id') or contains(@aria-label, 'ago')]",
                ".//span[contains(text(), 'h') or contains(text(), 'd') or contains(text(), 'w')]",
                ".//time",
                ".//*[contains(text(), 'minute') or contains(text(), 'hour') or contains(text(), 'day')]"
            ]

            for selector in timestamp_selectors:
                elements = comment_element.locator(selector).all()
                for elem in elements:
                    text = (elem.text_content() or '').strip()

                    if self._parse_relative_time(text):
                        return self._parse_relative_time(text)

            return datetime.now()
        except Exception as e:
            logger.debug(f'Error parsing timestamp: {e}')
            return datetime.now()

    def _parse_relative_time(self, time_str: str) -> Optional[datetime]:

        try:
            import re
            now = datetime.now()

            pattern = r'(\d+)\s*([hHdDwWmM])'
            match = re.search(pattern, time_str)

            if match:
                value = int(match.group(1))
                unit = match.group(2).lower()

                if unit in ['h']:
                    return now - timedelta(hours=value)
                elif unit in ['d']:
                    return now - timedelta(days=value)
                elif unit in ['w']:
                    return now - timedelta(weeks=value)
                elif unit in ['m'] and 'minute' in time_str.lower():
                    return now - timedelta(minutes=value)

            return None
        except Exception:
            return None

    def classify_comment_type(self, comment_element: Locator, comment_text: str) -> str:

        try:

            parent_indicators = comment_element.locator(".//*[contains(text(), 'Reply to') or contains(text(), 'replied')]").all()
            if parent_indicators:
                return 'reply'

            thread_indicators = comment_element.locator(".//*[contains(text(), 'replies') or contains(text(), 'View more')]").all()
            if thread_indicators:
                return 'thread'

            return 'new'
        except Exception as e:
            logger.debug(f'Error classifying comment type: {e}')
            return 'new'

    def calculate_comment_priority(self, comment_data: CommentData) -> int:

        priority = 100

        try:

            if comment_data['timestamp']:
                hours_old = (datetime.now() - comment_data['timestamp']).total_seconds() / 3600
                if hours_old < 1:
                    priority += 50
                elif hours_old < 6:
                    priority += 30
                elif hours_old < 24:
                    priority += 15

            text_length = len(comment_data['text'])
            if text_length > 100:
                priority += 20
            elif text_length > 50:
                priority += 10

            if '?' in comment_data['text']:
                priority += 25

            if '@' in comment_data['text'] or '#' in comment_data['text']:
                priority += 15

            spam_indicators = ['buy now', 'click here', 'free money', 'www.', 'http']
            for indicator in spam_indicators:
                if indicator.lower() in comment_data['text'].lower():
                    priority -= 30
                    break

            return max(priority, 10)

        except Exception as e:
            logger.debug(f'Error calculating priority: {e}')
            return 100

    def smart_comment_detection(self) -> list[CommentData]:

        if not self.page:
            return []

        start_time = time.time()
        detected_comments: list[CommentData] = []

        try:
            logger.info("🔍 Starting smart comment detection...")

            pre_expand_comments = self.get_comment_elements()
            current_comment_count = len(pre_expand_comments)
            count_changed = current_comment_count != self.last_comment_count

            if count_changed:
                logger.info(f"📈 Comment count changed: {self.last_comment_count} → {current_comment_count}")
                self.last_comment_count = current_comment_count
            else:
                logger.debug("📊 Comment count unchanged, forcing rescan to catch edits/live updates")

            self.smart_expand_comments()

            comment_elements = self.get_comment_elements()
            logger.info(f"🔍 Found {len(comment_elements)} comment elements to analyze")

            for elem in comment_elements:
                try:

                    comment_id = self.get_comment_id(elem)

                    # Check database first - use has_replied_to_comment for successful replies
                    if self.db.has_replied_to_comment(comment_id):
                        logger.debug(f"🔄 Comment {comment_id[:8]} already replied to (successful), skipping")
                        continue

                    # Also check if processed (includes failed attempts)
                    if self.db.is_comment_processed(comment_id):
                        processed_info = self.db.get_processed_comment(comment_id)
                        if processed_info:
                            status = processed_info.get('status', 'unknown')
                            if status == 'success':
                                logger.debug(f"🔄 Comment {comment_id[:8]} already successfully replied to, skipping")
                                continue
                            else:
                                logger.debug(f"🔄 Comment {comment_id[:8]} was processed but status was '{status}', will retry")
                        else:
                            logger.debug(f"🔄 Comment {comment_id[:8]} already processed in database, skipping")
                            continue

                    if comment_id in self.session_processed_ids:
                        logger.debug(f"🔄 Comment {comment_id[:8]} already processed in this session, skipping")
                        continue

                    element_html = elem.inner_html()[:100]
                    element_signature = hashlib.sha256(element_html.encode()).hexdigest()[:16]
                    user_id = self.get_user_id(elem)
                    author = self.get_author_name(elem)
                    comment_text = self.get_comment_text(elem)
                    timestamp = self.parse_comment_timestamp(elem)

                    logger.info(f"🔍 Processing comment: ID={comment_id[:8]}, Author={author}, Text='{comment_text[:50]}...'")

                    # Early check for own comment - skip it immediately
                    if self.is_own_comment(author, user_id):
                        logger.debug(f"🚫 Skipping own comment by {author} (user_id: {user_id[:8] if user_id else 'unknown'})")
                        # Store bot's user_id if we haven't found it yet
                        if user_id and not self.bot_user_id:
                            self.bot_user_id = user_id
                            logger.info(f"✅ Detected bot's own user_id: {user_id[:8]}")
                        # Mark as processed so we don't check it again
                        self.session_processed_ids.add(comment_id)
                        self.db.add_processed_comment(
                            comment_id,
                            user_id,
                            author if author is not None else '',
                            comment_text,
                            'SKIPPED_OWN_COMMENT',
                            status='skipped',
                            retry_count=0
                        )
                        continue

                    if not comment_text or len(comment_text.strip()) < 2:
                        logger.debug(f"⚠️  Skipping comment with insufficient content: '{comment_text}'")
                        continue

                    reply_button = self.find_reply_button_advanced(elem)

                    if reply_button is None:
                        reply_button = self.find_reply_button_fallback(elem)

                    if reply_button is None:
                        logger.debug(f"⚠️  Trying final fallback for reply button detection for {author}")
                        try:

                            all_buttons = elem.locator(".//div[@role='button']").all()
                            if all_buttons:
                                reply_button = all_buttons[-1]
                                logger.debug(f"✅ Using fallback button for {author}")
                        except:
                            pass

                    if reply_button:
                        logger.debug(f"✅ Found reply button for comment by {author}")
                    else:
                        logger.debug(f"❌ No reply button found for comment by {author} - will still queue for processing")

                    comment_data: CommentData = {
                        'id': comment_id,
                        'user_id': user_id,
                        'author': author,
                        'text': comment_text,
                        'element': elem,
                        'reply_button': reply_button,
                        'timestamp': timestamp,
                        'priority': 100,
                        'detected_at': datetime.now(),
                        'comment_type': self.classify_comment_type(elem, comment_text)
                    }

                    comment_data['priority'] = self.calculate_comment_priority(comment_data)

                    detected_comments.append(comment_data)
                    self.known_comment_elements.add(element_signature)
                    self.session_processed_ids.add(comment_id)

                    logger.info(f"✅ Detected new comment by {author}: {comment_text[:50]}... (Priority: {comment_data['priority']})")

                except Exception as e:
                    logger.debug(f"⚠️  Error processing comment element: {e}")
                    continue

            detected_comments.sort(key=lambda x: x['priority'], reverse=True)

            scan_duration = time.time() - start_time
            self.detection_metrics['total_scans'] += 1
            self.detection_metrics['comments_detected'] += len(detected_comments)
            self.detection_metrics['new_comments_found'] += len(detected_comments)
            self.detection_metrics['last_scan_duration'] = scan_duration

            if self.detection_metrics['total_scans'] > 0:
                self.detection_metrics['processing_time_avg'] = (
                    (self.detection_metrics['processing_time_avg'] * (self.detection_metrics['total_scans'] - 1) + scan_duration)
                    / self.detection_metrics['total_scans']
                )

            logger.info(f"🎯 Smart detection complete: {len(detected_comments)} new comments found in {scan_duration:.2f}s")
            self.last_full_scan = datetime.now()

            return detected_comments

        except Exception as e:
            logger.error(f"❌ Error in smart comment detection: {e}")
            return []

    def find_reply_button_advanced(self, comment_element: Locator) -> Locator | None:

        try:
            logger.debug("Starting advanced reply button detection...")

            logger.debug("Strategy 1: Direct text match for 'Reply'...")
            try:

                reply_candidates = comment_element.locator("//*[text()='Reply' or text()='reply']").all()
                logger.debug(f"  Found {len(reply_candidates)} elements with 'Reply' text")

                for candidate in reply_candidates:
                    try:
                        if candidate.is_visible():

                            is_clickable = candidate.evaluate()

                            if is_clickable:
                                logger.debug(f"Found clickable Reply element via text match")
                                return candidate
                    except:
                        continue
            except Exception as e:
                logger.debug(f"  Strategy 1 error: {e}")

            logger.debug("Strategy 2: Clickable elements containing 'Reply'...")
            try:
                all_elements = comment_element.locator(".//*[contains(text(), 'Reply')]").all()
                logger.debug(f"  Found {len(all_elements)} elements containing 'Reply'")

                for elem in all_elements:
                    try:
                        if elem.is_visible():
                            text = (elem.text_content() or '').strip()

                            if text.lower() in ['reply', 'reply ', ' reply'] or (len(text) < 15 and 'reply' in text.lower()):
                                logger.debug(f"Found potential reply button: '{text}'")
                                return elem
                    except:
                        continue
            except Exception as e:
                logger.debug(f"  Strategy 2 error: {e}")

            logger.debug("Strategy 3: Traditional button role search...")
            try:
                all_buttons = comment_element.locator(".//*[@role='button']").all()
                logger.debug(f"  Found {len(all_buttons)} elements with role=button")

                for btn in all_buttons:
                    try:
                        if btn.is_visible():
                            text = (btn.text_content() or '').strip().lower()
                            if 'reply' in text and len(text) < 30:
                                logger.debug(f"Found reply button via role=button: '{text}'")
                                return btn
                    except:
                        continue
            except Exception as e:
                logger.debug(f"  Strategy 3 error: {e}")

            logger.debug("Strategy 4: Facebook clickable class search...")
            try:
                clickable = comment_element.locator(".//*[contains(@class, 'x1i10hfl')]").all()
                for elem in clickable:
                    try:
                        if elem.is_visible():
                            text = (elem.text_content() or '').strip().lower()
                            if text == 'reply' or (len(text) < 15 and 'reply' in text):
                                logger.debug(f"Found via clickable class: '{text}'")
                                return elem
                    except:
                        continue
            except Exception as e:
                logger.debug(f"  Strategy 4 error: {e}")

            logger.debug("Advanced reply button detection failed")
            return None

        except Exception as e:
            logger.debug(f"Error in advanced reply button detection: {e}")
            return None
    def find_reply_button_fallback(self, comment_element: Locator) -> Locator | None:

        try:
            logger.debug("Using aggressive fallback reply button detection...")

            logger.debug("Strategy 1: Aggressive text search...")
            try:

                all_reply_elements = comment_element.locator(".//*").all()

                for elem in all_reply_elements:
                    try:
                        text = (elem.text_content() or '').strip()

                        if text and text.lower() in ['reply', 'reply ', ' reply', 'reply...']:
                            if elem.is_visible():
                                logger.debug(f"Found Reply text, attempting to use it: '{text}'")
                                return elem
                    except:
                        continue
            except Exception as e:
                logger.debug(f"  Aggressive text search error: {e}")

            logger.debug("Strategy 2: Page-level Reply search...")
            try:
                if self.page:

                    page_reply_elements = self.page.locator("text='Reply'").all()
                    logger.debug(f"  Found {len(page_reply_elements)} 'Reply' elements on page")

                    for elem in page_reply_elements:
                        try:
                            if elem.is_visible():

                                logger.debug("Found visible Reply element on page")
                                return elem
                        except:
                            continue
            except Exception as e:
                logger.debug(f"  Page-level search error: {e}")

            logger.debug("Strategy 3: Brute force clickability test...")
            try:

                potential_buttons = comment_element.locator(".//span, .//div").all()

                for elem in potential_buttons[:50]:
                    try:
                        if elem.is_visible():
                            text = (elem.text_content() or '').strip().lower()

                            if text == 'reply':
                                logger.debug("Found element with exact 'reply' text")
                                return elem
                    except:
                        continue
            except Exception as e:
                logger.debug(f"  Brute force error: {e}")

            logger.debug("All fallback strategies failed")
            return None

        except Exception as e:
            logger.debug(f"Fallback detection error: {e}")
            return None

    def get_comment_id(self, comment_element: Locator) -> str:

        try:

            html_content = comment_element.inner_html()[:200]
            return hashlib.sha256(html_content.encode()).hexdigest()[:16]
        except Exception:
            return hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]

    def smart_expand_comments(self):
            return None

    def get_comment_id(self, comment_element: Locator) -> str:

        try:

            html_content = comment_element.inner_html()[:200]
            return hashlib.sha256(html_content.encode()).hexdigest()[:16]
        except Exception:
            return 'new'

    def smart_expand_comments(self):

        if not self.page:
            return

        try:
            logger.debug("🔄 Smart expanding comments...")

            try:
                logger.debug("📜 Scrolling to load more comments...")
                for _ in range(3):
                    self.page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                    time.sleep(0.5)

                self.page.evaluate('window.scrollTo(0, 0)')
                time.sleep(0.5)

                self.page.evaluate('window.scrollTo(0, document.body.scrollHeight / 2)')
                time.sleep(0.5)
            except Exception as e:
                logger.debug(f"Error scrolling: {e}")

            expand_selectors = [
                "//div[@role='button'][contains(., 'View') and (contains(., 'more') or contains(., 'replies'))]",
                "//div[@role='button'][contains(., 'replies')]",
                "//span[contains(text(), 'View more')]/ancestor::div[@role='button']",
                "//div[contains(@aria-label, 'View more')][@role='button']",
                "//span[contains(text(), 'Hide') and contains(text(), 'replies')]/ancestor::div[@role='button']",
                "//div[contains(text(), 'replies') and contains(text(), 'View')][@role='button']"
            ]

            expanded_count = 0
            for selector in expand_selectors:
                try:
                    buttons = self.page.locator(selector).all()
                    for btn in buttons[:5]:
                        if btn.is_visible():
                            try:
                                btn.scroll_into_view_if_needed()
                                self.random_pause(0.2, 0.5)
                                btn.click(timeout=2000)
                                expanded_count += 1
                                logger.debug(f"✅ Expanded comment section")
                                time.sleep(1)
                            except:

                                try:
                                    btn.evaluate('el => el.click()')
                                    expanded_count += 1
                                    logger.debug(f"✅ Expanded comment section (JS click)")
                                except:
                                    pass
                except Exception:
                    continue

            if expanded_count > 0:
                logger.debug(f"📖 Expanded {expanded_count} comment sections")
                self.random_pause(1, 2)

        except Exception as e:
            logger.debug(f"Error in smart expansion: {e}")

        if not self.page:
            return
        try:
            logger.info('" Expanding all comment threads (including sub-comments)...')

            max_passes = 5
            for pass_num in range(max_passes):
                logger.debug(f'Expansion pass {pass_num + 1}/{max_passes}')

                expand_selectors = [
                    "//span[contains(text(), 'View more')]",
                    "//span[contains(text(), 'replies')]",
                    "//span[contains(text(), 'more replies')]",
                    "//span[contains(text(), 'Reply') and contains(text(), '·')]",
                    "//div[@role='button' and contains(., 'View')]",
                    "//div[contains(@aria-label, 'View more')]",
                ]
                buttons_found = 0
                for selector in expand_selectors:
                    try:
                        expand_buttons: list[Locator] = self.page.locator(selector).all()
                        for btn in expand_buttons:
                            try:

                                if not btn.is_visible():
                                    continue
                                btn.scroll_into_view_if_needed()
                                self.random_pause(0.3, 0.8)

                                try:
                                    btn.click(timeout=2000)
                                    buttons_found += 1
                                    logger.debug(f"Expanded comment thread")
                                except:

                                    btn.evaluate('el => el.click()')
                                    buttons_found += 1
                                    logger.debug(f"Expanded comment thread (JS)")
                                self.random_pause(0.5, 1.0)
                            except Exception as e:
                                logger.debug(f'Could not click expand button: {e}')
                                continue
                    except Exception as e:
                        logger.debug(f'Error finding buttons with selector: {e}')
                        continue
                logger.info(f'Pass {pass_num + 1}: Expanded {buttons_found} comment threads')

                if buttons_found == 0:
                    logger.info(f"All comment threads fully expanded after {pass_num + 1} passes")
                    break

                self.page.evaluate('window.scrollBy(0, 500)')
                self.random_pause(1, 2)
            logger.info('" Comment expansion complete - all threads visible')
        except Exception as e:
            logger.debug(f'Error expanding comments: {e}')

    def get_comment_elements(self) -> list[Locator]:

        if not self.page:
            return []

        selectors = [
            "//div[@role='article']",
            "//div[contains(@data-testid, 'UFI2Comment/root_depth')]",
            "//div[contains(@aria-label, 'Comment by')]",
            "//div[@data-visualcompletion='ignore-dynamic' and .//div[@dir='auto']]",
            "//div[contains(@class, 'x1n2onr6') and .//span[text()='Reply']]",
        ]

        unique_comments: list[Locator] = []
        seen_signatures: set[str] = set()

        for selector in selectors:
            try:
                elements = self.page.locator(selector).all()
                if not elements:
                    continue

                for elem in elements:
                    try:
                        text_preview = (elem.text_content() or '')[:120]
                        data_ft = elem.get_attribute('data-ft') or ''
                        signature_source = f"{selector}|{text_preview}|{data_ft}"
                        signature = hashlib.sha256(signature_source.encode()).hexdigest()
                        if signature in seen_signatures:
                            continue
                        seen_signatures.add(signature)
                        unique_comments.append(elem)
                    except Exception:
                        unique_comments.append(elem)
            except Exception as selector_error:
                logger.debug(f"Comment selector failed ({selector}): {selector_error}")

        logger.debug(f"get_comment_elements gathered {len(unique_comments)} candidates")
        return unique_comments
    def get_comment_id(self, comment_element: Locator) -> str:

        try:
            comment_text = (comment_element.text_content() or '')[:200]
            author_elements = comment_element.locator(".//a[contains(@href, '?id=') or contains(@class, 'actor')]").all()
            author = (author_elements[0].text_content() or '') if author_elements else 'unknown'

            unique_string = f"{author}_{comment_text}_{datetime.now().strftime('%Y-%m-%d')}"
            comment_id = hashlib.sha256(unique_string.encode()).hexdigest()[:16]
            return comment_id
        except Exception as e:
            logger.debug(f'Error generating comment ID: {e}')
            return hashlib.sha256(str(random.randint(1000000, 9999999)).encode()).hexdigest()[:16]
    def get_user_id(self, comment_element: Locator) -> str:

        try:
            author_elements = comment_element.locator(
                ".//a[contains(@href, 'user') or contains(@href, 'profile.php')]"
            ).all()
            if author_elements:
                href = author_elements[0].get_attribute('href') or ''

                if 'user/' in href:
                    user_id = href.split('user/')[-1].split('?')[0].split('/')[0]
                elif 'id=' in href:
                    user_id = href.split('id=')[-1].split('&')[0]
                else:

                    author_name = self.get_author_name(comment_element)
                    user_id = hashlib.sha256((author_name or 'unknown').encode()).hexdigest()[:16]
                return user_id

            author_name = self.get_author_name(comment_element)
            return hashlib.sha256((author_name or 'unknown').encode()).hexdigest()[:16]
        except Exception as e:
            logger.debug(f'Error getting user ID: {e}')
            return hashlib.sha256(str(random.randint(1000000, 9999999)).encode()).hexdigest()[:16]
    def get_author_name(self, comment_element: Locator) -> str | None:

        try:

            facebook_author_selectors = [
                'span.x193iq5w[dir="auto"]',
                'span[dir="auto"]',
                'span.x193iq5w',
            ]

            for selector in facebook_author_selectors:
                elements = comment_element.locator(selector).all()
                for element in elements:
                    text = (element.text_content() or '').strip()

                    if (text and
                        len(text) > 0 and
                        len(text) < 100 and
                        not any(x in text.lower() for x in [
                            'like', 'reply', 'comment', 'share', 'edited', 'most relevant',
                            'minute', 'hour', 'day', 'week', 'ago', 'just now',
                            'see more', 'hide', 'translate'
                        ]) and

                        any(c.isalpha() for c in text)):
                        logger.debug(f"✅ Found author name: '{text}' using Facebook selector: {selector}")
                        return text

            legacy_selectors = [
                ".//a[@role='link']//span",
                ".//a[contains(@href, 'user')]//span",
                ".//a[contains(@href, 'profile.php')]//span",
                ".//strong//a",
                ".//h3//a",
            ]

            for selector in legacy_selectors:
                elements = comment_element.locator(selector).all()
                for element in elements:
                    text = (element.text_content() or '').strip()
                    if (text and len(text) > 0 and len(text) < 100 and
                        text not in ['Author', 'Admin', 'Like', 'Reply', 'Comment', 'Share', 'Edited', 'Most relevant']):
                        logger.debug(f"✅ Found author name: '{text}' using legacy selector: {selector}")
                        return text

            strong_elements = comment_element.locator(".//strong, .//b, .//em").all()
            for element in strong_elements:
                text = (element.text_content() or '').strip()
                if (text and len(text) > 0 and len(text) < 50 and
                    not any(x in text.lower() for x in ['like', 'reply', 'comment', 'minute', 'hour', 'day', 'week', 'ago'])):
                    logger.debug(f"✅ Found author name in strong element: '{text}'")
                    return text

            full_text = (comment_element.text_content() or '').strip()
            if full_text:
                lines = [line.strip() for line in full_text.split('\n') if line.strip()]
                if lines:
                    first_line = lines[0]

                    if (len(first_line) > 0 and len(first_line) < 50 and
                        not any(x in first_line.lower() for x in ['like', 'reply', 'comment', 'minute', 'hour', 'day', 'week'])):
                        logger.debug(f"✅ Found author name from first line: '{first_line}'")
                        return first_line

            logger.debug("⚠️  Could not extract author name from comment element")
            return "Unknown User"

        except Exception as e:
            logger.debug(f'Error getting author name: {e}')
            return "Unknown User"
    def get_post_content(self) -> str:
        """Extract the main post content from the Facebook post page."""
        if not self.page:
            return ""

        # Return cached content if available (including empty string to avoid re-extraction)
        if self.post_content_cache is not None:
            return self.post_content_cache

        try:
            logger.info("📄 Extracting post content...")

            # Try multiple selectors to find the main post content
            post_selectors = [
                # Main post content area
                "//div[@role='article']//div[@dir='auto']",
                "//div[contains(@data-pagelet, 'FeedUnit')]//div[@dir='auto']",
                "//div[@role='article']//span[@dir='auto']",
                # Post text in various structures
                "//div[@role='article']//div[contains(@class, 'x193iq5w')]//span[@dir='auto']",
                "//div[@role='article']//p[@dir='auto']",
                # Fallback: get text from the main article
                "//div[@role='article']",
            ]

            post_text = ""
            for selector in post_selectors:
                try:
                    elements = self.page.locator(selector).all()
                    if elements:
                        # Get the first element which is usually the main post
                        main_element = elements[0]
                        text = (main_element.text_content() or '').strip()

                        # Filter out common UI elements
                        if (text and
                            len(text) > 20 and  # Must be substantial
                            len(text) < 5000 and  # Not too long
                            'like' not in text.lower()[:50] and
                            'comment' not in text.lower()[:50] and
                            'share' not in text.lower()[:50] and
                            not any(time_word in text.lower()[:100] for time_word in ['minute', 'hour', 'day', 'week', 'ago', 'just now'])):

                            # Clean up the text
                            lines = [line.strip() for line in text.split('\n') if line.strip()]
                            # Filter out lines that look like UI elements
                            filtered_lines = []
                            for line in lines:
                                if (len(line) > 5 and
                                    not any(ui_word in line.lower() for ui_word in ['like', 'comment', 'share', 'reply', 'see more', 'see less']) and
                                    not any(time_word in line.lower() for time_word in ['minute', 'hour', 'day', 'week', 'ago'])):
                                    filtered_lines.append(line)

                            if filtered_lines:
                                post_text = '\n'.join(filtered_lines[:10])  # Take first 10 meaningful lines
                                break
                except Exception as e:
                    logger.debug(f"Selector {selector} failed: {e}")
                    continue

            # If we didn't find it with selectors, try getting from the main article
            if not post_text or len(post_text) < 20:
                try:
                    article = self.page.locator("//div[@role='article']").first
                    if article.count() > 0 and article.is_visible():
                        full_text = (article.text_content() or '').strip()
                        # Extract the main content (usually the first substantial paragraph)
                        paragraphs = [p.strip() for p in full_text.split('\n') if len(p.strip()) > 20]
                        if paragraphs:
                            # The first substantial paragraph is usually the post
                            post_text = paragraphs[0]
                except Exception as e:
                    logger.debug(f"Fallback post extraction failed: {e}")

            # Clean and limit the post content
            if post_text:
                post_text = post_text.strip()[:1000]  # Limit to 1000 chars
                self.post_content_cache = post_text
                logger.info(f"✅ Extracted post content ({len(post_text)} chars): {post_text[:100]}...")
            else:
                logger.warning("⚠️  Could not extract post content")
                post_text = ""
                # Cache empty string to avoid repeated extraction attempts
                self.post_content_cache = ""

            return post_text

        except Exception as e:
            logger.error(f"Error extracting post content: {e}")
            return ""

    def get_comment_text(self, comment_element: Locator) -> str:

        try:

            text_divs = comment_element.locator('div[dir="auto"]').all()

            logger.debug(f"Found {len(text_divs)} div[dir='auto'] elements")

            if text_divs:
                for idx, div in enumerate(text_divs):
                    text = (div.text_content() or '').strip()
                    logger.debug(f"  Div {idx+1}: '{text[:50]}...' (len={len(text)})")

                    if text and len(text) > 0 and text not in ['Like', 'Reply', 'Author', 'Comment', 'Share']:

                        if not any(pattern in text for pattern in ['5h', '1h', '2h', '3h', '4h', '6h', '7h', '8h', '9h',
                                                                   '1d', '2d', '3d', '4d', '5d', '6d', '7d',
                                                                   '1w', '2w', '3w', '4w', 'minute', 'hour', 'day', 'week']):
                            logger.info(f"✓ Found comment text: '{text[:50]}'")
                            return text

            full_text = comment_element.text_content() or ''
            lines = [line.strip() for line in full_text.split('\n') if line.strip()]

            author_found = False
            for line in lines:

                if len(line) < 1:
                    continue

                if line in ['Like', 'Reply', 'Comment', 'Share', 'Author', 'Admin', 'Edited', 'Most relevant']:
                    continue

                if any(x in line for x in ['5h', '1h', '2h', '3h', '4h', '6h', '7h', '8h', '9h',
                                           '1d', '2d', '3d', '4d', '5d', '6d', '7d',
                                           '1w', '2w', '3w', '4w', 'minute', 'hour', 'day', 'week', 'ago']):
                    continue

                if 'edit' in line.lower() or 'delete' in line.lower():
                    continue

                if author_found:
                    logger.debug(f"Found comment text (fallback): {line[:50]}")
                    return line
                else:
                    author_found = True

            logger.warning("Could not extract comment text from element")
            return ""

        except Exception as e:
            logger.error(f'Error extracting comment text: {e}')
            import traceback
            logger.error(traceback.format_exc())
            return ""

    def add_comments_to_queue(self, comments: list[CommentData]) -> None:

        if not comments:
            logger.debug("No comments provided to add_comments_to_queue")
            return

        logger.info(f"📋 Processing {len(comments)} detected comments for queue...")

        valid_comments = []
        seen_comment_ids = set()
        seen_user_ids = set()

        for comment in comments:
            comment_id = comment.get('id', '')
            user_id = comment.get('user_id', '')
            author = comment.get('author')

            # Skip if comment_id is already in queue or processed
            if comment_id and self.is_comment_id_in_queue(comment_id):
                logger.debug(f"🔄 Skipping duplicate comment_id: {comment_id[:8]}")
                continue

            # Skip if we've already seen this comment_id in this batch
            if comment_id in seen_comment_ids:
                logger.debug(f"🔄 Skipping duplicate comment_id in batch: {comment_id[:8]}")
                continue
            seen_comment_ids.add(comment_id)

            # Check if it's the bot's own comment (improved check)
            if self.is_own_comment(author, user_id):
                logger.info(f"🚫 Skipping own comment by {author} (user_id: {user_id[:8] if user_id else 'unknown'})")
                # Store bot's user_id if we haven't found it yet
                if user_id and not self.bot_user_id:
                    self.bot_user_id = user_id
                    logger.info(f"✅ Detected bot's own user_id: {user_id[:8]}")
                continue

            if not comment['text'] or len(comment['text'].strip()) < 2:
                logger.debug(f"🚫 Skipping comment with no text by {author}")
                continue

            # Check if user is already in queue (prevent multiple comments from same user)
            if user_id and self.is_user_in_queue(user_id):
                logger.info(f"🚫 Skipping comment by {author}: user already in queue (user_id: {user_id[:8]})")
                continue

            # Check if we've already seen this user_id in this batch
            if user_id in seen_user_ids:
                logger.info(f"🚫 Skipping duplicate user in batch: {author} (user_id: {user_id[:8]})")
                continue
            seen_user_ids.add(user_id)

            try:
                can_reply, reason = self.can_reply_to_user(
                    user_id,
                    author if author is not None else ''
                )
                if not can_reply:
                    logger.info(f"🚫 Skipping comment by {author}: {reason}")
                    continue
            except Exception as e:

                logger.warning(f"⚠️  User check failed for {author}, allowing anyway: {e}")

            valid_comments.append(comment)
            logger.info(f"✅ Added comment by {author} to processing queue: '{comment['text'][:50]}...'")

        if valid_comments:
            self.comment_queue['pending'].extend(valid_comments)

            self.comment_queue['pending'].sort(key=lambda x: x['priority'], reverse=True)

            max_queue_size = 50
            if len(self.comment_queue['pending']) > max_queue_size:
                self.comment_queue['pending'] = self.comment_queue['pending'][:max_queue_size]

            logger.info(f"📥 Added {len(valid_comments)} comments to queue (Total pending: {len(self.comment_queue['pending'])})")
        else:
            logger.warning(f"⚠️  No valid comments to add to queue from {len(comments)} detected comments")

    def process_comment_queue(self) -> bool:

        if not self.comment_queue['pending']:
            logger.debug("🔍 No comments in pending queue to process")
            return False

        logger.info(f"📋 Processing {len(self.comment_queue['pending'])} comments from queue")

        processed_any = False

        batch_size = len(self.comment_queue['pending'])

        if batch_size == 0:
            return False

        logger.info(f"🔄 Starting to process {batch_size} comments from queue...")

        for batch_num in range(batch_size):

            try:
                rate_ok, rate_msg = self.check_rate_limit()
                if not rate_ok:
                    logger.warning(f"🚫 Rate limit reached: {rate_msg}")

                    logger.info("⚠️  Continuing in offline mode (rate limit check failed)")
            except Exception as e:

                logger.debug(f"Rate limit check failed (offline mode), continuing: {e}")

            if not self.comment_queue['pending']:
                break

            comment = self.comment_queue['pending'].pop(0)
            self.comment_queue['processing'].append(comment)

            logger.info(f"\n🚀 Processing high-priority comment {batch_num + 1}/{batch_size} (Priority: {comment['priority']})")
            logger.info(f"👤 Author: {comment['author']}")
            logger.info(f"📝 Text: {comment['text'][:100]}...")
            logger.info(f"⏰ Detected: {comment['detected_at'].strftime('%H:%M:%S')}")
            logger.info(f"🔘 Has reply button: {'Yes' if comment['reply_button'] else 'No (will retry)'}")

            success = self.reply_to_comment(comment, len(self.comment_queue['completed']) + 1)

            self.comment_queue['processing'].remove(comment)

            if success:
                self.comment_queue['completed'].append(comment['id'])
                processed_any = True
                self.last_activity_time = datetime.now()
                logger.info(f"✅ Successfully processed comment {comment['id'][:8]}")
            else:
                self.comment_queue['failed'].append(comment['id'])
                logger.warning(f"❌ Failed to process comment {comment['id'][:8]}")

            if batch_num < batch_size - 1:
                self.random_pause(1, 3)

        return processed_any

    def adaptive_delay_between_comments(self) -> None:

        base_delay = random.uniform(
            float(self.config['DELAYS']['MEDIUM_MIN']),
            float(self.config['DELAYS']['MEDIUM_MAX'])
        )

        queue_size = len(self.comment_queue['pending'])
        if queue_size > 10:
            base_delay *= 0.7
        elif queue_size < 3:
            base_delay *= 1.3

        total_attempts = len(self.comment_queue['completed']) + len(self.comment_queue['failed'])
        if total_attempts > 5:
            success_rate = len(self.comment_queue['completed']) / total_attempts
            if success_rate < 0.7:
                base_delay *= 1.5

        final_delay = base_delay + random.uniform(-0.5, 0.5)
        final_delay = max(1.0, final_delay)

        logger.debug(f"⏱️  Adaptive delay: {final_delay:.1f}s (base: {base_delay:.1f}s)")
        time.sleep(final_delay)

    def update_adaptive_scan_interval(self) -> None:

        base_interval = float(self.config['DELAYS']['SCAN_INTERVAL'])

        time_since_last_activity = (datetime.now() - self.last_activity_time).total_seconds()

        if len(self.comment_queue['pending']) == 0:
            if time_since_last_activity < 300:
                self.adaptive_scan_interval = base_interval * 0.5
            else:
                self.adaptive_scan_interval = base_interval * 1.5
        else:
            self.adaptive_scan_interval = base_interval * 0.8

        self.adaptive_scan_interval = max(5.0, min(60.0, self.adaptive_scan_interval))

        logger.debug(f"📊 Adaptive scan interval: {self.adaptive_scan_interval:.1f}s")

    def print_queue_status(self) -> None:

        pending = len(self.comment_queue['pending'])
        processing = len(self.comment_queue['processing'])
        completed = len(self.comment_queue['completed'])
        failed = len(self.comment_queue['failed'])

        logger.info(f"\n📊 QUEUE STATUS")
        logger.info(f"{'='*50}")
        logger.info(f"📥 Pending: {pending}")
        logger.info(f"⚙️  Processing: {processing}")
        logger.info(f"✅ Completed: {completed}")
        logger.info(f"❌ Failed: {failed}")

        if pending > 0:

            logger.info(f"\n🔜 Next in queue:")
            for i, comment in enumerate(self.comment_queue['pending'][:3]):
                logger.info(f"   {i+1}. {comment['author']} (Priority: {comment['priority']})")

        logger.info(f"\n⚡ PERFORMANCE METRICS")
        logger.info(f"{'='*50}")
        logger.info(f"🔍 Total scans: {self.detection_metrics['total_scans']}")
        logger.info(f"📈 Comments detected: {self.detection_metrics['comments_detected']}")
        logger.info(f"⏱️  Avg scan time: {self.detection_metrics['processing_time_avg']:.2f}s")
        logger.info(f"🎯 Detection rate: {self.detection_metrics['new_comments_found']/max(1, self.detection_metrics['total_scans']):.1f}/scan")
        logger.info(f"⏰ Adaptive interval: {self.adaptive_scan_interval:.1f}s")
        logger.info(f"{'='*50}\n")

    def get_all_comments(self) -> list[CommentData]:

        if not self.page:
            return []
        self.smart_expand_comments()
        comments: list[CommentData] = []
        try:
            for _ in range(3):
                self.page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                self.random_pause(2, 4)
            comment_elements: list[Locator] = self.page.locator("//div[@role='article']").all()
            logger.info(f'Found {len(comment_elements)} comment elements.')
            for elem in comment_elements:
                try:
                    comment_id = self.get_comment_id(elem)
                    user_id = self.get_user_id(elem)
                    author = self.get_author_name(elem)

                    comment_text = self.get_comment_text(elem)

                    if not comment_text or len(comment_text) < 1:
                        logger.debug(f"Skipping comment with no text content")
                        continue

                    reply_buttons: list[Locator] = elem.locator("//div[@role='button' and contains(@class, 'x1i10hfl') and (contains(text(), 'Reply') or contains(text(), 'reply'))]").all()

                    if not reply_buttons:
                        reply_buttons: list[Locator] = elem.locator(".//div[@role='button']").all()

                    reply_button = None
                    for btn in reply_buttons:
                        btn_text = (btn.text_content() or '').lower()
                        if 'reply' in btn_text or 'رد' in btn_text or 'répondre' in btn_text:
                            reply_button = btn
                            break

                    comment_data: CommentData = {
                        'id': comment_id,
                        'user_id': user_id,
                        'author': author,
                        'text': comment_text[:200],
                        'element': elem,
                        'reply_button': reply_button,
                        'timestamp': self.parse_comment_timestamp(elem),
                        'priority': 100,
                        'detected_at': datetime.now(),
                        'comment_type': self.classify_comment_type(elem, comment_text)
                    }
                    comments.append(comment_data)
                    logger.debug(f"Added comment from {author}: {comment_text[:50]}")

                except Exception as e:
                    logger.debug(f'Error processing comment element: {e}')
                    continue
            logger.info(f'Processed {len(comments)} comments.')
            return comments
        except Exception as e:
            logger.error(f'Error getting comments: {e}')
            return []
    def check_rate_limit(self) -> Tuple[bool, str]:

        window = int(self.config.get('RATE_LIMIT_WINDOW', 300))
        recent_count = self.db.get_recent_reply_count(window)
        if recent_count >= int(self.config.get('MAX_REPLIES_PER_WINDOW', 8)):
            reason = f"Rate limit exceeded: {recent_count}/{int(self.config.get('MAX_REPLIES_PER_WINDOW', 8))} replies in last {window}s"
            logger.warning(reason)
            return False, reason
        return True, ''
    def can_reply_to_user(self, user_id: str, user_name: str) -> Tuple[bool, str]:

        try:
            reply_count = self.db.get_user_reply_count(user_id)
            max_replies = int(self.config.get('MAX_REPLIES_PER_USER', 9999))
            if reply_count >= max_replies:
                reason = f'Already replied to {user_name} {reply_count} time(s)'
                return False, reason
        except Exception as e:

            logger.debug(f"User reply count check failed (offline), allowing: {e}")

        cooldown_seconds = int(self.config.get('COOLDOWN_SECONDS', 0))
        if cooldown_seconds > 0:
            try:
                last_reply = self.db.get_last_reply_time(user_id)
                if last_reply:
                    time_since_last = datetime.now() - last_reply
                    cooldown = timedelta(seconds=cooldown_seconds)
                    if time_since_last < cooldown:
                        remaining = (cooldown - time_since_last).seconds
                        reason = f'Cooldown active for {user_name}: {remaining}s remaining'
                        return False, reason
            except Exception as e:

                logger.debug(f"Cooldown check failed (offline), allowing: {e}")

        return True, ''
    def generate_reply(self, comment_text: str, post_content: str | None = None) -> str:
        try:
            # Validate and sanitize input
            if not comment_text or len(comment_text.strip()) < 2:
                logger.warning("Comment text too short, using fallback")
                return "Thanks for your comment!"

            comment_clean = sanitize_text(comment_text, max_length=500)

            # Get post content if not provided
            if post_content is None:
                post_content = self.get_post_content()
            else:
                post_content = sanitize_text(post_content, max_length=500)

            # Create cache key including both post and comment for better context
            cache_input = f"{post_content[:200]}|{comment_clean}"
            cache_key = hashlib.sha256(cache_input.encode()).hexdigest()[:16]

            # Check cache
            if cache_key in self.response_cache:
                cached = self.response_cache[cache_key]
                logger.debug(f'Using cached response for similar comment')
                return cached

            # Clean cache if too large
            if len(self.response_cache) > MAX_CACHE_SIZE:
                keys_to_remove = list(self.response_cache.keys())[:CACHE_CLEANUP_SIZE]
                for key in keys_to_remove:
                    del self.response_cache[key]
                logger.debug(f"Cleaned {CACHE_CLEANUP_SIZE} entries from response cache")

            # Build prompt with both post and comment context
            if post_content and len(post_content.strip()) > 10:
                # Use templates that support both post and comment (first 5 templates)
                template = random.choice(PROMPT_TEMPLATES[:5])
                try:
                    prompt = template.format(
                        post=post_content[:500],  # Limit post content to 500 chars
                        comment=comment_clean
                    )
                    logger.debug(f"Using post+comment context (post: {len(post_content)} chars, comment: {len(comment_clean)} chars)")
                except (KeyError, IndexError, ValueError) as e:
                    # Fallback if template doesn't have {post} or wrong template selected
                    logger.warning(f"Template error, using comment-only: {e}")
                    try:
                        template = random.choice(PROMPT_TEMPLATES[5:])
                        prompt = template.format(comment=comment_clean)
                    except (IndexError, ValueError) as fallback_error:
                        # Ultimate fallback if slicing fails
                        logger.error(f"Fallback template selection failed: {fallback_error}")
                        prompt = f"Generate a helpful reply to this comment: {comment_clean}"
            else:
                # Fallback to comment-only if post content not available (last 5 templates)
                try:
                    template = random.choice(PROMPT_TEMPLATES[5:])
                    prompt = template.format(comment=comment_clean)
                    logger.debug(f"Using comment-only context (post not available)")
                except (IndexError, ValueError) as e:
                    # Ultimate fallback if slicing fails
                    logger.error(f"Template selection failed: {e}")
                    prompt = f"Generate a helpful reply to this comment: {comment_clean}"

            system_prompt = "You are a helpful assistant replying to Facebook comments. Generate brief, relevant replies that directly address the comment content in the context of the original post. Be friendly and conversational. Do not use emojis unless absolutely necessary. Keep replies under 100 words."

            seed = random.randint(1, 10000)

            # Get OpenAI config
            openai_config = get_openai_config()

            response = self.openai_client.chat.completions.create(
                model=openai_config['MODEL'],
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                temperature=openai_config['TEMPERATURE'],
                max_tokens=openai_config['MAX_TOKENS'],
                seed=seed
            )

            try:
                reply_raw = getattr(response.choices[0].message, 'content', '')
                reply = (reply_raw or '').strip()

                if not reply or len(reply) < 3:
                    reply = f"Thanks for your comment about {comment_clean[:30]}..."

            except Exception as e:
                logger.warning(f'Error extracting reply: {e}')
                reply = f"Thanks for your comment about {comment_clean[:30]}..."

            if reply:
                self.response_cache[cache_key] = reply

            # Cache cleanup is now handled earlier in the function

            logger.info(f'Generated reply for comment "{comment_clean[:50]}...": {reply[:50]}...')
            return reply
        except Exception as e:
            logger.error(f'Failed to generate reply: {e}')

            fallbacks = [
                'Thank you for your comment! I appreciate your input.',
                'Thanks for sharing your thoughts!',
                'I appreciate you taking the time to comment!',
                'Great point! Thanks for your feedback.',
                'Thank you! Your input is valuable.'
            ]
            return random.choice(fallbacks)
    def verify_reply_box_belongs_to_comment(self, reply_box: Locator, comment_element: Locator) -> bool:
        """Verify that the reply box actually belongs to the comment we're replying to."""
        try:
            comment_bbox = comment_element.bounding_box()
            reply_bbox = reply_box.bounding_box()

            if not comment_bbox or not reply_bbox:
                return False

            # Check if reply box is within reasonable distance of comment
            comment_bottom = comment_bbox.get('y', 0) + comment_bbox.get('height', 0)
            reply_top = reply_bbox.get('y', 0)

            # Reply box should be below the comment, within 500px
            distance = reply_top - comment_bottom
            if distance < 0 or distance > 500:
                logger.warning(f"Reply box too far from comment (distance: {distance:.0f}px)")
                return False

            # Check if reply box is within the same article/comment thread
            try:
                # Get the comment's article container
                comment_article = comment_element.locator('xpath=./ancestor::div[contains(@role, "article")]').first
                if comment_article.count() > 0:
                    # Check if reply box is within the same article
                    reply_in_article = reply_box.locator('xpath=./ancestor::div[contains(@role, "article")]').first
                    if reply_in_article.count() > 0:
                        # Verify they're the same article by checking if reply box is descendant of comment article
                        article_id = comment_article.evaluate('el => el.getAttribute("data-pagelet") || el.id || ""')
                        reply_article_id = reply_in_article.evaluate('el => el.getAttribute("data-pagelet") || el.id || ""')
                        if article_id and reply_article_id and article_id == reply_article_id:
                            return True

                        # Alternative: check if reply box is a descendant of comment article
                        reply_is_descendant = reply_box.evaluate(f"""
                            (function() {{
                                const commentArticle = document.evaluate('{comment_article}', document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                                const replyBox = arguments[0];
                                if (!commentArticle || !replyBox) return false;
                                return commentArticle.contains(replyBox);
                            }})();
                        """)
                        if reply_is_descendant:
                            return True
            except Exception as e:
                logger.debug(f"Article verification failed: {e}")

            # Check if there's a "Reply to [author]" text near the reply box
            try:
                reply_container = reply_box.locator('xpath=./ancestor::div[position() <= 5]').first
                if reply_container.count() > 0:
                    container_text = (reply_container.text_content() or '').lower()
                    # Get comment author name
                    comment_author = self.get_author_name(comment_element)
                    if comment_author:
                        author_lower = comment_author.lower()
                        # Check if container mentions replying to this author
                        if f'reply to {author_lower}' in container_text or f'replying to {author_lower}' in container_text:
                            return True
            except Exception as e:
                logger.debug(f"Author verification failed: {e}")

            # If distance is reasonable, accept it
            if 0 <= distance <= 300:
                return True

            return False
        except Exception as e:
            logger.debug(f"Reply box verification error: {e}")
            return False

    def find_reply_input_box(self, comment_element: Locator) -> Locator | None:
        if not self.page:
            return None

        # Get comment info for logging
        try:
            comment_author = self.get_author_name(comment_element)
            comment_text = self.get_comment_text(comment_element)
            comment_id = self.get_comment_id(comment_element)
            logger.info(f"🔍 Looking for reply box for comment by {comment_author} (ID: {comment_id[:8]}): '{comment_text[:50]}...'")
        except:
            pass

        # First, try to find the reply box marked by JavaScript
        try:
            marked_box = self.page.locator('[data-bot-selected-reply-box="true"]').first
            if marked_box.count() > 0 and marked_box.is_visible():
                logger.info('✅ Found JavaScript-marked reply box')
                # Verify it belongs to this comment
                if self.verify_reply_box_belongs_to_comment(marked_box, comment_element):
                    # Remove the marker
                    marked_box.evaluate('el => el.removeAttribute("data-bot-selected-reply-box")')
                    return marked_box
                else:
                    # Remove invalid marker
                    marked_box.evaluate('el => el.removeAttribute("data-bot-selected-reply-box")')
        except Exception as e:
            logger.debug(f"JavaScript-marked box check failed: {e}")

        max_attempts = 5
        for attempt in range(max_attempts):
            time.sleep(0.5)
            try:
                comment_bbox: Optional[FloatRect] = comment_element.bounding_box()
                if not comment_bbox:
                    logger.warning('Could not get comment bounding box')
                    continue

                logger.info(f'Looking for reply box near comment at y={comment_bbox.get("y", 0):.0f}')

                # Method 1: Try to find reply box within comment element structure
                try:
                    reply_box_in_comment = comment_element.locator("xpath=.//following-sibling::*//p[@dir='auto' and contains(@class, 'xdj266r')] | .//following-sibling::*//div[@contenteditable='true']").first
                    if reply_box_in_comment and reply_box_in_comment.count() > 0 and reply_box_in_comment.is_visible():
                        bbox = reply_box_in_comment.bounding_box()
                        if bbox and bbox.get('height', 0) > 20:
                            if self.verify_reply_box_belongs_to_comment(reply_box_in_comment, comment_element):
                                logger.info('✅ Found and verified reply box within comment element structure')
                                return reply_box_in_comment
                            else:
                                logger.warning('⚠️  Reply box found but verification failed')
                except Exception as e:
                    logger.debug(f"Method 1 failed: {e}")

                # Method 2: Try to find reply box after comment element
                try:
                    reply_box_after = comment_element.locator("xpath=./ancestor::div[contains(@role, 'article')]//following-sibling::*[1]//p[@dir='auto' and contains(@class, 'xdj266r')] | ./ancestor::div[contains(@role, 'article')]//following-sibling::*[1]//div[@contenteditable='true']").first
                    if reply_box_after and reply_box_after.count() > 0 and reply_box_after.is_visible():
                        bbox = reply_box_after.bounding_box()
                        if bbox and bbox.get('height', 0) > 20:
                            if self.verify_reply_box_belongs_to_comment(reply_box_after, comment_element):
                                logger.info('✅ Found and verified reply box after comment element')
                                return reply_box_after
                            else:
                                logger.warning('⚠️  Reply box found but verification failed')
                except Exception as e:
                    logger.debug(f"Method 2 failed: {e}")

                # Method 3: Search all edit boxes and find the closest verified one
                all_edit_boxes: list[Locator] = self.page.locator("//p[@dir='auto' and contains(@class, 'xdj266r')]").all()
                if not all_edit_boxes:
                    all_edit_boxes: list[Locator] = self.page.locator("//div[@contenteditable='true']").all()

                if not all_edit_boxes:
                    logger.debug(f'Attempt {attempt + 1}: No editable boxes found')
                    continue

                visible_boxes: list[tuple[Locator, float]] = []
                for box in all_edit_boxes:
                    try:
                        if not box.is_visible():
                            continue
                        bbox: Optional[FloatRect] = box.bounding_box()
                        if not bbox or bbox.get('height', 0) < 20 or bbox.get('width', 0) < 100:
                            continue

                        aria_label = box.get_attribute('aria-label') or ''
                        if 'write a comment' in aria_label.lower():
                            logger.debug('Skipping main comment box')
                            continue

                        # Verify this box belongs to our comment
                        if not self.verify_reply_box_belongs_to_comment(box, comment_element):
                            continue

                        parent = box.locator('xpath=./ancestor::div[3]').first
                        parent_text = (parent.text_content() or '').lower() if parent else ''

                        box_y = bbox.get('y', 0)
                        comment_y = comment_bbox.get('y', 0)
                        comment_height = comment_bbox.get('height', 0)

                        if 'reply to' in parent_text:
                            distance = abs(box_y - (comment_y + comment_height))
                            if distance < 300:
                                logger.info(f"✅ Found verified reply box with 'Reply to' text (distance: {distance:.0f}px)")
                                visible_boxes.append((box, distance))
                        elif box_y > comment_y and box_y < comment_y + comment_height + 200:
                            distance = abs(box_y - (comment_y + comment_height))
                            logger.debug(f'Found verified box below comment (distance: {distance:.0f}px)')
                            visible_boxes.append((box, distance))
                    except Exception as e:
                        logger.debug(f'Error checking box: {e}')
                        continue

                if visible_boxes:
                    visible_boxes.sort(key=lambda x: x[1])
                    closest_box, distance = visible_boxes[0]
                    logger.info(f'✅ Found {len(visible_boxes)} verified reply boxes, using closest (distance: {distance:.0f}px)')
                    return closest_box
                else:
                    logger.debug(f'Attempt {attempt + 1}: No verified reply boxes found near comment')
            except Exception as e:
                logger.error(f'Error in attempt {attempt + 1}: {e}')
                continue
        logger.error('❌ No verified reply boxes found after all attempts')
        return None

    def _type_character_by_character(self, reply_box: Locator, reply_text: str) -> bool:
        """Type text character by character with periodic dialog checks."""

        try:
            for i, char in enumerate(reply_text):
                try:
                    # Check for dialog periodically during typing (more frequently)
                    if (i + 1) % 5 == 0:
                        if self.handle_leave_page_dialog():
                            logger.info(f"Dialog handled during typing at character {i + 1}")
                            time.sleep(0.3)
                        # Also re-setup prevention in case it was cleared
                        if (i + 1) % 20 == 0:
                            self.setup_dialog_prevention()

                    if char in [' ', ',', '.', '!', '?']:

                        delay = random.uniform(100, 250)
                    elif char.isupper():

                        delay = random.uniform(80, 180)
                    else:

                        delay = random.uniform(50, 120)

                    reply_box.type(char, delay=delay)

                    if (i + 1) % 20 == 0:
                        logger.debug(f'Typed {i + 1}/{len(reply_text)} characters')

                    if random.random() < 0.05:
                        time.sleep(random.uniform(0.3, 0.8))
                except Exception as e:
                    logger.error(f'Error typing character {i}: {e}')
                    return False

            reply_box.dispatch_event('input')
            reply_box.dispatch_event('change')
            return True
        except Exception as e:
            logger.error(f'Character-by-character typing failed: {e}')
            return False
    def reply_to_comment(self, comment_data: CommentData, reply_count: int) -> bool:

        max_retries = self.config['MAX_RETRIES']
        retry_count = 0
        while retry_count < max_retries:
            try:

                # Check if we've already successfully replied to this comment
                if self.db.has_replied_to_comment(comment_data['id']):
                    reply_text = self.db.get_reply_for_comment(comment_data['id'])
                    logger.info(f"✅ IDEMPOTENCY: Comment {comment_data['id'][:8]} already successfully replied to")
                    if reply_text:
                        logger.info(f"   Previous reply: {reply_text[:50]}...")
                    self.db.log_event('idempotency_check', f"Skipped duplicate (already replied): {comment_data['id']}", 'info')
                    return False

                # Check if processed (but maybe failed)
                if self.db.is_comment_processed(comment_data['id']):
                    processed_info = self.db.get_processed_comment(comment_data['id'])
                    if processed_info:
                        status = processed_info.get('status', 'unknown')
                        if status == 'success':
                            logger.info(f"✅ Comment {comment_data['id'][:8]} already successfully processed - skipping")
                            return False
                        else:
                            logger.info(f"🔄 Comment {comment_data['id'][:8]} was processed with status '{status}', will retry")
                    else:
                        logger.info(f"IDEMPOTENCY: Comment {comment_data['id'][:8]} already processed - skipping")
                        self.db.log_event('idempotency_check', f"Skipped duplicate: {comment_data['id']}", 'info')
                        return False
                if not comment_data['reply_button']:
                    logger.warning(f"🔍 No reply button initially found for comment {comment_data['id']}, trying to find it again...")

                    reply_button = self.find_reply_button_advanced(comment_data['element'])
                    if not reply_button:
                        reply_button = self.find_reply_button_fallback(comment_data['element'])

                    if reply_button:
                        logger.info(f"✅ Found reply button on retry for comment {comment_data['id']}")
                        comment_data['reply_button'] = reply_button
                    else:
                        logger.warning(f"❌ Still no reply button found for comment {comment_data['id']} after retry")
                        self.db.log_event('no_reply_button', f"Comment: {comment_data['id']}", 'warning')
                        return False

                # Improved check for own comment (case-insensitive, checks user_id too)
                if self.is_own_comment(comment_data.get('author'), comment_data.get('user_id')):
                    logger.info(f"🚫 Skipping own comment by {comment_data.get('author')} (user_id: {comment_data.get('user_id', 'unknown')[:8] if comment_data.get('user_id') else 'unknown'})")

                    # Store bot's user_id if we haven't found it yet
                    if comment_data.get('user_id') and not self.bot_user_id:
                        self.bot_user_id = comment_data['user_id']
                        logger.info(f"✅ Detected bot's own user_id: {comment_data['user_id'][:8]}")

                    self.db.add_processed_comment(
                        comment_data['id'],
                        comment_data['user_id'],
                        comment_data['author'] if comment_data['author'] is not None else '',
                        comment_data['text'],
                        'SKIPPED_OWN_COMMENT',
                        status='skipped',
                        retry_count=0
                    )
                    return False
                logger.info(f"Attempting to reply to comment by {comment_data['author']} (attempt {retry_count + 1}/{max_retries})")
                self.db.log_event('reply_attempt', json.dumps({
                    'comment_id': comment_data['id'],
                    'user_id': comment_data['user_id'],
                    'user_name': comment_data['author'],
                    'attempt': retry_count + 1
                }), 'info')

                # Ensure browser is alive before proceeding
                if not self.ensure_browser_alive():
                    logger.error("❌ Cannot proceed: Browser is not available")
                    if retry_count < max_retries - 1:
                        retry_count += 1
                        time.sleep(5)
                        continue
                    else:
                        return False

                # Re-get the element since page might have reloaded
                try:
                    # Try to use the stored element, but verify it's still valid
                    if not self.is_browser_alive():
                        raise RuntimeError("Browser closed")
                    comment_data['element'].scroll_into_view_if_needed()
                except Exception as e:
                    error_msg = str(e)
                    if 'closed' in error_msg.lower() or 'Target' in error_msg or 'Browser closed' in error_msg:
                        logger.error(f"❌ Browser/page closed during scroll: {e}")
                        if not self.ensure_browser_alive():
                            if retry_count < max_retries - 1:
                                retry_count += 1
                                time.sleep(5)
                                continue
                            return False
                        # After recovery, we need to re-detect the comment
                        logger.warning("⚠️  Comment element lost after browser recovery, marking for re-detection")
                        self.comment_queue['failed'].append(comment_data['id'])
                        return False
                    raise

                self.random_pause(1, 2)

                # Close all other reply boxes first to avoid confusion
                if self.is_browser_alive():
                    try:
                        self.close_all_reply_boxes()
                    except Exception as e:
                        if 'closed' in str(e).lower() or 'Target' in str(e):
                            logger.warning(f"Browser closed during close_all_reply_boxes: {e}")
                            if not self.ensure_browser_alive():
                                if retry_count < max_retries - 1:
                                    retry_count += 1
                                    time.sleep(5)
                                    continue
                                return False
                time.sleep(0.5)

                # Mark this comment element with a unique identifier before clicking
                comment_id = comment_data.get('id', '')
                try:
                    comment_data['element'].evaluate(f"""
                        (el) => {{
                            el.setAttribute('data-bot-replying-to', '{comment_id}');
                            el.setAttribute('data-bot-reply-timestamp', '{datetime.now().timestamp()}');
                        }}
                    """)
                except:
                    pass

                # Check browser before clicking
                if not self.is_browser_alive():
                    logger.error("❌ Browser closed before clicking reply button")
                    if not self.ensure_browser_alive():
                        if retry_count < max_retries - 1:
                            retry_count += 1
                            time.sleep(5)
                            continue
                        return False

                try:
                    comment_data['reply_button'].scroll_into_view_if_needed()
                    time.sleep(0.5)
                    self.human_mouse_jiggle(comment_data['reply_button'], moves=1)
                    comment_data['reply_button'].click(timeout=3000)
                    logger.info('✅ Clicked reply button')
                except Exception as e:
                    error_msg = str(e)
                    if 'closed' in error_msg.lower() or 'Target' in error_msg:
                        logger.error(f"❌ Browser/page closed during click: {e}")
                        if not self.ensure_browser_alive():
                            if retry_count < max_retries - 1:
                                retry_count += 1
                                time.sleep(5)
                                continue
                            return False
                        # Continue to next attempt
                        continue
                    logger.info(f'Regular click failed ({e}), using JavaScript click')
                    try:
                        comment_data['reply_button'].evaluate('el => el.click()')
                    except Exception as js_e:
                        if 'closed' in str(js_e).lower():
                            logger.error(f"❌ Browser closed during JS click: {js_e}")
                            if not self.ensure_browser_alive():
                                if retry_count < max_retries - 1:
                                    retry_count += 1
                                    time.sleep(5)
                                    continue
                                return False
                            continue
                        raise

                logger.info('Waiting for reply box to appear...')
                self.random_pause(2, 3)

                # Check browser before proceeding
                if not self.is_browser_alive():
                    logger.error("❌ Browser closed while waiting for reply box")
                    if not self.ensure_browser_alive():
                        if retry_count < max_retries - 1:
                            retry_count += 1
                            time.sleep(5)
                            continue
                        return False

                # Use JavaScript to find the reply box that was just opened
                try:
                    if not self.page:
                        raise RuntimeError("Page is None")
                    reply_box_js = self.page.evaluate(f"""
                        (function() {{
                            // Find the comment we just clicked reply on
                            const targetComment = document.querySelector('[data-bot-replying-to="{comment_id}"]');
                            if (!targetComment) return null;

                            // Find all visible reply boxes
                            const allReplyBoxes = Array.from(document.querySelectorAll('p[dir="auto"][class*="xdj266r"], div[contenteditable="true"]'));

                            // Filter to only visible ones
                            const visibleBoxes = allReplyBoxes.filter(box => {{
                                const style = window.getComputedStyle(box);
                                return style.display !== 'none' && style.visibility !== 'hidden' && box.offsetHeight > 20;
                            }});

                            if (visibleBoxes.length === 0) return null;

                            // Get comment position
                            const commentRect = targetComment.getBoundingClientRect();
                            const commentBottom = commentRect.bottom;

                            // Find the reply box closest to and below this comment
                            let bestBox = null;
                            let bestDistance = Infinity;

                            for (const box of visibleBoxes) {{
                                const boxRect = box.getBoundingClientRect();
                                const boxTop = boxRect.top;

                                // Box should be below the comment
                                if (boxTop < commentBottom) continue;

                                // Check if box is within the same article/thread
                                let current = box;
                                let foundComment = false;
                                for (let i = 0; i < 10; i++) {{
                                    current = current.parentElement;
                                    if (!current) break;
                                    if (current === targetComment || current.contains(targetComment)) {{
                                        foundComment = true;
                                        break;
                                    }}
                                }}

                                if (!foundComment) {{
                                    // Check if they're in the same article
                                    const commentArticle = targetComment.closest('[role="article"]');
                                    const boxArticle = box.closest('[role="article"]');
                                    if (commentArticle && boxArticle && commentArticle !== boxArticle) {{
                                        continue;
                                    }}
                                }}

                                const distance = boxTop - commentBottom;
                                if (distance < bestDistance && distance < 500) {{
                                    bestDistance = distance;
                                    bestBox = box;
                                }}
                            }}

                            if (bestBox) {{
                                // Mark this box so we can find it
                                bestBox.setAttribute('data-bot-selected-reply-box', 'true');
                                return true;
                            }}

                            return false;
                        }})();
                    """)

                    if reply_box_js:
                        logger.info('✅ JavaScript found the reply box')
                except Exception as e:
                    logger.debug(f"JavaScript reply box finder failed: {e}")

                comment_author = comment_data.get('author', 'Unknown')
                logger.info(f'Looking for reply box for comment by {comment_author}')

                reply_box = self.find_reply_input_box(comment_data['element'])
                if not reply_box:
                    logger.error(f'Could not find reply input box (attempt {retry_count + 1})')
                    if retry_count < max_retries - 1:
                        logger.info('Retrying...')
                        retry_count += 1

                        try:
                            if self.page:
                                self.page.keyboard.press('Escape')
                            time.sleep(1)
                        except:
                            pass
                        continue
                    else:
                        if self.page:
                            if self.is_browser_alive() and self.page:
                                try:
                                    self.page.screenshot(path=f'debug_no_reply_box_{reply_count}.png')
                                except Exception:
                                    pass
                        self.db.log_event('reply_failed', f"No reply box: {comment_data['id']}", 'error')

                        self.db.add_processed_comment(
                            comment_data['id'],
                            comment_data['user_id'],
                            comment_data['author'] if comment_data['author'] is not None else '',
                            comment_data['text'],
                            'FAILED_NO_REPLY_BOX',
                            status='failed',
                            retry_count=retry_count
                        )
                        return False

                # Final verification that this reply box belongs to the correct comment
                if not self.verify_reply_box_belongs_to_comment(reply_box, comment_data['element']):
                    logger.error(f'❌ Reply box verification failed - this box does not belong to comment by {comment_data.get("author")}')
                    logger.error(f'   Comment text: {comment_data.get("text", "")[:50]}...')
                    if retry_count < max_retries - 1:
                        logger.warning('Retrying with fresh search...')
                        retry_count += 1
                        try:
                            if self.page:
                                self.page.keyboard.press('Escape')
                            time.sleep(1)
                        except:
                            pass
                        continue
                    else:
                        logger.error('Failed to find correct reply box after all retries')
                        return False

                # Check if reply box is empty or has unexpected content (like @mentions)
                try:
                    existing_text = reply_box.evaluate('(el) => el.textContent || el.innerText || el.value || ""')
                    if existing_text and len(existing_text.strip()) > 0:
                        # Check if it contains a mention/tag (starts with @)
                        if existing_text.strip().startswith('@'):
                            logger.warning(f'⚠️  Reply box contains mention/tag: "{existing_text[:50]}..." - clearing it')
                            # Clear the mention
                            reply_box.evaluate('(el) => { el.textContent = ""; el.innerText = ""; if(el.value) el.value = ""; }')
                            time.sleep(0.3)
                        else:
                            logger.info(f'Reply box has existing text: "{existing_text[:50]}..." - will clear before typing')
                except:
                    pass

                # Final check: Verify the reply box is still the correct one
                if not self.verify_reply_box_belongs_to_comment(reply_box, comment_data['element']):
                    logger.error('❌ Reply box verification failed after finding - retrying...')
                    if retry_count < max_retries - 1:
                        retry_count += 1
                        try:
                            if self.page:
                                self.page.keyboard.press('Escape')
                            time.sleep(1)
                        except:
                            pass
                        continue
                    else:
                        logger.error('Failed to find correct reply box after all retries')
                        return False

                logger.info(f'✅ Found and verified reply input box for comment by {comment_data.get("author")}')
                logger.info(f'📝 Comment we\'re replying to: "{comment_data.get("text", "")[:50]}..."')
                logger.info('Found reply input box, generating reply...')

                # Check for dialog before starting
                self.handle_leave_page_dialog()

                # Get post content for context-aware reply generation
                post_content = self.get_post_content()
                reply_text = self.generate_reply(comment_data['text'], post_content)
                logger.info(f'Typing reply: {reply_text[:50]}...')

                try:
                    reply_box.focus()
                    time.sleep(0.5)
                    # Check for dialog after focus
                    self.handle_leave_page_dialog()
                    reply_box.click()
                    time.sleep(0.3)

                    try:
                        reply_box.press('Control+A')
                        time.sleep(0.2)
                        reply_box.press('Delete')
                        time.sleep(0.2)
                    except:
                        pass

                    logger.info('Clearing any pre-filled content...')
                    try:
                        reply_box.evaluate('(el) => { el.textContent = ""; el.innerText = ""; if(el.value) el.value = ""; }')
                        time.sleep(0.2)
                        reply_box.press('Control+A')
                        time.sleep(0.1)
                        reply_box.press('Backspace')
                        time.sleep(0.2)
                    except:
                        pass

                    logger.info(f'Trying to insert reply text: {reply_text[:50]}...')

                    logger.info('Method 1: Using evaluate with textContent...')
                    try:
                        escaped = reply_text.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'").replace('\n', ' ').replace('\r', '')
                        reply_box.evaluate(f'''
                            (el) => {{
                                el.textContent = "{escaped}";
                                el.innerText = "{escaped}";
                                if(el.value !== undefined) el.value = "{escaped}";
                                el.dispatchEvent(new Event("input", {{ bubbles: true, cancelable: true }}));
                                el.dispatchEvent(new Event("keyup", {{ bubbles: true, cancelable: true }}));
                                el.dispatchEvent(new Event("change", {{ bubbles: true, cancelable: true }}));
                            }}
                        ''')
                        time.sleep(0.5)

                        actual = reply_box.evaluate('(el) => el.textContent || el.innerText || el.value || ""')
                        if reply_text.strip().lower() in str(actual).lower() or (len(str(actual).strip()) > len(reply_text.strip()) * 0.5):
                            logger.info(f'✓ Method 1 successful. Text in box: {actual[:50]}...')
                            typing_successful = True
                        else:
                            raise Exception(f'Text not set. Got: {actual[:50]}')
                    except Exception as e1:
                        logger.warning(f'Method 1 failed: {e1}')
                        typing_successful = False

                    if not typing_successful:
                        logger.info('Method 2: Using fill()...')
                        try:
                            reply_box.fill('')
                            time.sleep(0.2)
                            reply_box.fill(reply_text)
                            time.sleep(0.4)
                            reply_box.dispatch_event('input', {'bubbles': True})
                            reply_box.dispatch_event('keyup', {'bubbles': True})
                            reply_box.dispatch_event('change', {'bubbles': True})
                            time.sleep(0.3)

                            actual = reply_box.evaluate('(el) => el.textContent || el.innerText || el.value || ""')
                            if reply_text.strip().lower() in str(actual).lower() or len(str(actual).strip()) > 3:
                                logger.info(f'✓ Method 2 successful. Text in box: {actual[:50]}...')
                                typing_successful = True
                            else:
                                raise Exception(f'Text not set. Got: {actual[:50]}')
                        except Exception as fill_error:
                            logger.warning(f'Method 2 failed: {fill_error}')
                            typing_successful = False

                    if not typing_successful:
                        logger.info('Trying innerHTML manipulation...')
                        try:
                            escaped_text = reply_text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '')
                            reply_box.evaluate(f'(element) => {{ element.textContent = "{escaped_text}"; element.innerText = "{escaped_text}"; element.dispatchEvent(new Event("input", {{ bubbles: true }})); element.dispatchEvent(new Event("keyup", {{ bubbles: true }})); }}')
                            time.sleep(0.5)

                            actual_text = reply_box.evaluate('(el) => el.textContent || el.innerText || ""')
                            if reply_text.lower().strip() in str(actual_text).lower():
                                logger.info('✓ Used innerHTML method successfully')
                                typing_successful = True
                            else:
                                raise Exception('Text not properly set')
                        except Exception as inner_error:
                            logger.warning(f'innerHTML method failed: {inner_error}')
                            typing_successful = False

                    if not typing_successful:
                        logger.info('Trying type() method...')
                        try:
                            reply_box.clear()
                            time.sleep(0.2)
                            reply_box.type(reply_text, delay=30)
                            time.sleep(0.3)

                            reply_box.dispatch_event('input', {'bubbles': True})
                            reply_box.dispatch_event('change', {'bubbles': True})
                            reply_box.dispatch_event('keyup', {'bubbles': True})
                            time.sleep(0.2)

                            actual_text = reply_box.input_value() if hasattr(reply_box, 'input_value') else reply_box.evaluate('(el) => el.textContent || el.innerText || ""')
                            if reply_text.lower().strip() in str(actual_text).lower():
                                logger.info('✓ Used type() method successfully')
                                typing_successful = True
                            else:
                                raise Exception('Text not properly set')
                        except Exception as type_error:
                            logger.warning(f'Type method failed: {type_error}')
                            typing_successful = False

                    if not typing_successful:
                        logger.info('Using character-by-character fallback...')
                        typing_successful = self._type_character_by_character(reply_box, reply_text)

                    # Check for dialog after typing attempts
                    self.handle_leave_page_dialog()

                except Exception as e:
                    logger.warning(f'Could not focus on reply box: {e}')
                    typing_successful = False

                if not typing_successful:
                    logger.error('All typing methods failed')
                    return False

                time.sleep(1.0)
                # Check for dialog before verification
                self.handle_leave_page_dialog()
                try:
                    final_text = reply_box.evaluate('(el) => el.textContent || el.innerText || el.value || ""')
                    final_text_str = str(final_text).strip()

                    if not final_text_str or len(final_text_str) < 3:
                        logger.error(f'Reply text is empty after typing. Expected: {reply_text[:50]}')
                        logger.error(f'Actual content in box: "{final_text_str}"')
                        return False

                    if reply_text.strip().lower() not in final_text_str.lower() and len(final_text_str) < len(reply_text.strip()) * 0.5:
                        logger.warning(f'Reply text mismatch. Expected: {reply_text[:50]}... Got: {final_text_str[:50]}...')
                        logger.warning('Text might have been modified by Facebook, but proceeding...')

                    logger.info(f'✓ Verified reply text in box: {final_text_str[:50]}...')

                    # Final verification: Make sure we're still in the correct reply box
                    if not self.verify_reply_box_belongs_to_comment(reply_box, comment_data['element']):
                        logger.error('❌ Reply box changed during typing! This is the wrong comment.')
                        logger.error(f'   Expected comment: {comment_data.get("author")} - "{comment_data.get("text", "")[:50]}..."')
                        return False
                    logger.info(f'✓ Confirmed this is still the reply box for comment by {comment_data.get("author")}')
                except Exception as e:
                    logger.warning(f'Could not verify reply text: {e}')

                logger.info('Finished typing, waiting 1 second before submit...')
                time.sleep(1.0)
                # Check for dialog before submission
                self.handle_leave_page_dialog()

                submitted = False

                time.sleep(1)

                try:

                    submit_selectors = [
                        'xpath=./ancestor::div[contains(@class, "x1n2onr6")]//div[@role="button" and @aria-label="Comment"]',
                        'xpath=./following-sibling::div[@role="button" and @aria-label="Comment"]',
                        'xpath=./parent::*/following-sibling::*//div[@role="button" and @aria-label="Comment"]',
                        'xpath=./..//div[@role="button" and contains(@class, "x1i10hfl")]'
                    ]

                    submit_button = None
                    for selector in submit_selectors:
                        try:
                            btn = reply_box.locator(selector).first
                            if btn.is_visible():
                                submit_button = btn
                                logger.info(f'Found submit button with selector: {selector}')
                                break
                        except:
                            continue

                    if submit_button:
                        is_disabled = submit_button.get_attribute('aria-disabled')
                        logger.info(f'Submit button disabled status: {is_disabled}')
                        if is_disabled == 'false' or is_disabled is None:
                            logger.info('Submit button is enabled, clicking it directly')
                            submit_button.click()
                            self.random_pause(3, 5)
                            logger.info('✓ Submitted with submit button click')
                            submitted = True
                        else:
                            logger.warning('Submit button still disabled, trying keyboard shortcuts')
                    else:
                        logger.warning('Could not find submit button, will try keyboard methods')
                except Exception as btn_error:
                    logger.debug(f'Could not check submit button: {btn_error}')

                if 'submitted' not in locals():
                    submitted = False

                self.random_pause(1, 2)

                if not submitted:
                    logger.info('Submitting reply...')

                    try:
                        logger.info('Trying Ctrl+Enter...')
                        reply_box.press('Control+Enter')
                        submitted = True
                        logger.info('✓ Submitted with Ctrl+Enter')
                    except Exception as e:
                        logger.warning(f'Ctrl+Enter failed: {e}')

                    if not submitted:
                        try:
                            logger.info('Trying Enter key...')
                            reply_box.press('Enter')
                            submitted = True
                            logger.info('✓ Submitted with Enter')
                        except Exception as e:
                            logger.warning(f'Enter key failed: {e}')

                if not submitted:
                    try:
                        logger.info('Looking for submit button...')

                        reply_container = reply_box.locator('xpath=./ancestor::div[contains(@class, "x1n2onr6") or contains(@role, "textbox") or contains(@contenteditable, "true")]//following-sibling::*[1] | ./ancestor::div[contains(@class, "x1n2onr6") or contains(@role, "textbox")]//parent::*/following-sibling::*[1]').first

                        submit_selectors = [

                            "//div[@role='button' and @aria-label]",
                            "//div[@role='button' and contains(@class, 'x1i10hfl')]",
                            "//div[contains(@style, 'cursor: pointer') and @role='button']",

                            "//div[@role='button' and (contains(text(), 'Post') or contains(text(), 'Reply') or contains(text(), 'Send') or contains(text(), 'Submit'))]",
                            "//button[contains(text(), 'Post') or contains(text(), 'Reply') or contains(text(), 'Send')]",

                            "//div[@aria-label='Post' or @aria-label='Reply' or @aria-label='Send' or @aria-label='Submit' or contains(@aria-label, 'Send') or contains(@aria-label, 'Post')]",
                        ]

                        for selector in submit_selectors:
                            try:
                                if reply_container:
                                    submit_buttons = reply_container.locator(selector).all()
                                else:

                                    submit_buttons = reply_box.locator(f'xpath=./ancestor::div[3]//{selector[2:]}').all()

                                for btn in submit_buttons:
                                    try:
                                        if btn.is_visible():
                                            btn_text = btn.text_content() or ''
                                            btn_label = btn.get_attribute('aria-label') or ''
                                            logger.info(f'Found potential submit button - Text: "{btn_text}", Aria-label: "{btn_label}"')

                                            btn.click(timeout=2000)
                                            submitted = True
                                            logger.info('✓ Submitted with button click')
                                            break
                                    except Exception as btn_error:
                                        logger.debug(f'Button click failed: {btn_error}')
                                        continue
                                if submitted:
                                    break
                            except Exception as selector_error:
                                logger.debug(f'Selector failed: {selector_error}')
                                continue

                        if not submitted:
                            logger.info('Trying broader page search for submit button...')
                            page_submit_buttons = self.page.locator("//div[@role='button' and (@aria-label or contains(@class, 'x1i10hfl'))]").all()
                            for btn in page_submit_buttons:
                                try:
                                    if btn.is_visible():
                                        btn_bbox = btn.bounding_box()
                                        reply_bbox = reply_box.bounding_box()
                                        if btn_bbox and reply_bbox:

                                            distance = abs(btn_bbox['x'] - reply_bbox['x']) + abs(btn_bbox['y'] - reply_bbox['y'])
                                            if distance < 100:
                                                logger.info(f'Found nearby button at distance {distance}px')
                                                btn.click(timeout=2000)
                                                submitted = True
                                                logger.info('✓ Submitted with nearby button click')
                                                break
                                except Exception as page_btn_error:
                                    logger.debug(f'Page button failed: {page_btn_error}')
                                    continue

                    except Exception as e:
                        logger.warning(f'Submit button search failed: {e}')

                if not submitted:
                    logger.warning('All submission methods failed, trying coordinate-based click as final attempt')
                    try:

                        reply_bbox = reply_box.bounding_box()
                        if reply_bbox and self.page:

                            submit_x = reply_bbox['x'] + reply_bbox['width'] + 10
                            submit_y = reply_bbox['y'] + reply_bbox['height'] / 2
                            logger.info(f'Trying coordinate click at ({submit_x}, {submit_y})')
                            self.page.mouse.click(submit_x, submit_y)
                            submitted = True
                            logger.info('✓ Submitted with coordinate click')
                    except Exception as coord_error:
                        logger.warning(f'Coordinate click failed: {coord_error}')

                        try:
                            if self.page:
                                self.page.keyboard.press('Control+Enter')
                        except:
                            pass

                self.random_pause(3, 5)

                # Check for dialog after submission using the dedicated function
                if self.handle_leave_page_dialog():
                    logger.warning("Found 'Stay on Page' dialog after submission - retrying")
                    time.sleep(1)
                    if retry_count < max_retries - 1:
                        retry_count += 1
                        continue
                    else:
                        self.db.log_event('reply_failed', f"Stay dialog: {comment_data['id']}", 'error')
                        return False

                timestamp = datetime.now().isoformat()
                logger.info(f'SUCCESS - Comment Replied')
                logger.info(f"Comment ID: {comment_data['id']}")
                logger.info(f"User ID: {comment_data['user_id']}")
                logger.info(f"User Name: {comment_data['author']}")
                logger.info(f'Timestamp: {timestamp}')
                logger.info(f'Reply: {reply_text}')
                # Add to database with success status
                self.db.add_processed_comment(
                    comment_data['id'],
                    comment_data['user_id'],
                    comment_data['author'] if comment_data['author'] is not None else '',
                    comment_data['text'],
                    reply_text,
                    status='success',
                    retry_count=retry_count
                )

                # Verify it was saved
                saved_reply = self.db.get_reply_for_comment(comment_data['id'])
                if saved_reply:
                    logger.info(f"✅ Reply saved to database for comment {comment_data['id'][:8]}")
                else:
                    logger.warning(f"⚠️  Reply may not have been saved to database for comment {comment_data['id'][:8]}")

                self.db.update_user_stats(comment_data['user_id'], comment_data['author'] if comment_data['author'] is not None else '')
                self.db.add_rate_limit_entry()
                self.db.log_event('reply_success', json.dumps({
                    'comment_id': comment_data['id'],
                    'user_id': comment_data['user_id'],
                    'user_name': comment_data['author'],
                    'reply': reply_text,
                    'timestamp': timestamp,
                    'retry_count': retry_count
                }), 'info')
                logger.info(f"Reply {reply_count} posted successfully")
                return True
            except Exception as e:
                logger.error(f'Error replying to comment (attempt {retry_count + 1}): {e}')
                import traceback
                logger.error(traceback.format_exc())
                self.db.log_event('reply_error', json.dumps({
                    'comment_id': comment_data['id'],
                    'error': str(e),
                    'attempt': retry_count + 1
                }), 'error')
                if retry_count < max_retries - 1:
                    logger.info(f'Retrying in 5 seconds... (attempt {retry_count + 2}/{max_retries})')
                    time.sleep(5)
                    retry_count += 1

                    try:
                        if self.page:
                            self.page.keyboard.press('Escape')
                    except:
                        pass
                else:
                    # Only take screenshot if browser is still alive
                    if self.is_browser_alive() and self.page:
                        try:
                            self.page.screenshot(path=f'error_{reply_count}.png')
                        except Exception:
                            logger.debug("Could not take error screenshot - browser may be closed")

                    self.db.add_processed_comment(
                        comment_data['id'],
                        comment_data['user_id'],
                        comment_data['author'] if comment_data['author'] is not None else '',
                        comment_data['text'],
                        f'FAILED: {str(e)}',
                        status='failed',
                        retry_count=retry_count
                    )
                    return False
        return False
    def run(self):

        self.setup_driver()
        if not self.page:
            logger.error("Page not initialized after setup_driver. Exiting.")
            return
        try:

            self.page.set_default_timeout(30000)

            post_url = self.config.get('POST_URL')
            if not post_url:
                logger.error('POST_URL is not configured. Set POST_URL in .env or config.')
                return
            logger.info(f"Loading Facebook post URL: {post_url}")

            try:
                self.page.goto(
                    post_url,
                    wait_until='domcontentloaded',
                    timeout=30000
                )
                logger.info("✓ Page loaded successfully")
            except Exception as e:
                logger.warning(f"Page load timeout, but continuing: {e}")

            try:
                self.page.wait_for_selector("//div[@role='main']", timeout=10000)
                logger.info("✓ Facebook main content detected")
            except:
                logger.warning("Main content not detected quickly, but continuing...")

            # Setup dialog prevention as soon as page loads
            self.setup_dialog_prevention()

            stats: Statistics = cast(Statistics, self.db.get_statistics())
            logger.info(f"\n{'='*60}")
            logger.info(f"INITIAL STATISTICS")
            logger.info(f"{'='*60}")
            logger.info(f"Total comments processed: {stats.get('total_comments', 0)}")
            logger.info(f"Unique users replied to: {stats.get('unique_users', 0)}")
            logger.info(f"Successful replies: {stats.get('successful', 0)}")
            logger.info(f"Failed replies: {stats.get('failed', 0)}")
            logger.info(f"{'='*60}\n")

            logger.info("Waiting for page to stabilize...")
            self.random_pause(2, 4)

            # Extract and cache post content early
            logger.info("📄 Extracting post content for context-aware replies...")
            post_content = self.get_post_content()
            if post_content:
                logger.info(f"✅ Post content cached ({len(post_content)} characters)")
            else:
                logger.warning("⚠️  Could not extract post content - replies will use comment-only context")

            reply_count = 0
            iteration = 0

            run_continuously = self.config.get('RUN_CONTINUOUSLY', True)
            if run_continuously:
                logger.info('🤖 ADVANCED CONTINUOUS MODE: Intelligent comment detection and priority-based processing')
                logger.info('✨ Features: Smart detection, priority queue, adaptive timing, real-time metrics')
                logger.info('Press Ctrl+C to stop the bot gracefully\n')

            while iteration < self.config['MAX_ITERATIONS']:
                if not run_continuously and reply_count >= self.config['MAX_REPLIES']:
                    logger.info('Max replies reached.')
                    break

                iteration += 1
                logger.info(f"\n🔍 SCAN {iteration}: Advanced Comment Detection")
                logger.info(f"{'='*60}")

                try:
                    rate_ok, rate_msg = self.check_rate_limit()
                    if not rate_ok:
                        logger.warning(f"🚫 {rate_msg}")

                        logger.info('⚠️  Rate limit check failed, continuing in offline mode...')
                except Exception as e:
                    logger.debug(f"Rate limit check failed (offline mode), continuing: {e}")

                new_comments = self.smart_comment_detection()
                logger.info(f"🔍 Detection result: {len(new_comments)} comments detected")

                if new_comments:
                    logger.info(f"📥 Adding {len(new_comments)} comments to queue...")
                    self.add_comments_to_queue(new_comments)
                    logger.info(f"🎯 Found {len(new_comments)} new comments, queue now has {len(self.comment_queue['pending'])} pending")
                else:
                    logger.debug("No new comments detected in this scan")

                max_queue_iterations = 50
                for queue_iter in range(max_queue_iterations):
                    if not self.comment_queue['pending']:
                        if queue_iter > 0:
                            logger.info(f"✅ Queue empty after {queue_iter} processing iterations")
                        break

                    pending_count = len(self.comment_queue['pending'])
                    logger.info(f"📋 Queue processing iteration {queue_iter + 1}/{max_queue_iterations} ({pending_count} pending comments)")
                    processed_any = self.process_comment_queue()

                    if processed_any:

                        completed_count = len(self.comment_queue['completed'])
                        reply_count = completed_count
                        remaining = len(self.comment_queue['pending'])
                        logger.info(f"✅ Processed comments: {completed_count} completed, {remaining} remaining")
                    else:

                        if len(self.comment_queue['pending']) > 0:
                            logger.warning(f"⚠️  No comments processed but {len(self.comment_queue['pending'])} still pending - retrying...")

                            self.random_pause(1, 2)
                        else:
                            break

                    if len(self.comment_queue['pending']) > 0:
                        self.random_pause(0.5, 1.0)

                if random.random() < 0.3:
                    self.simulate_human_activity()

                if iteration % 3 == 0:
                    self.print_queue_status()

                self.update_adaptive_scan_interval()

                if iteration % 5 == 0:
                    self.db.clean_old_rate_limit_entries(self.config['RATE_LIMIT_WINDOW'])

                if len(self.comment_queue['pending']) > 0:

                    wait_time = min(self.adaptive_scan_interval, 10.0)
                    logger.info(f"⏱️  Pending comments detected - short wait: {wait_time:.1f}s")
                else:
                    wait_time = self.adaptive_scan_interval
                    logger.info(f"💤 No pending comments - adaptive wait: {wait_time:.1f}s")

                time.sleep(wait_time)

                if iteration % 15 == 0:
                    logger.info('🔄 Smart page refresh...')
                    if self.page:
                        self.page.reload()
                    self.random_pause(3, 6)

            logger.info(f"\n🏁 FINAL ADVANCED STATISTICS")
            logger.info(f"{'='*60}")
            stats: Statistics = cast(Statistics, self.db.get_statistics())
            logger.info(f"📊 Total comments processed: {stats.get('total_comments', 0)}")
            logger.info(f"👥 Unique users replied to: {stats.get('unique_users', 0)}")
            logger.info(f"✅ Successful replies: {stats.get('successful', 0)}")
            logger.info(f"❌ Failed replies: {stats.get('failed', 0)}")
            logger.info(f"🚀 Session replies: {len(self.comment_queue['completed'])}")

            logger.info(f"\n⚡ PERFORMANCE METRICS")
            logger.info(f"{'='*60}")
            logger.info(f"🔍 Total scans performed: {self.detection_metrics['total_scans']}")
            logger.info(f"📈 Comments detected: {self.detection_metrics['comments_detected']}")
            logger.info(f"⏱️  Average scan time: {self.detection_metrics['processing_time_avg']:.2f}s")

            if self.detection_metrics['total_scans'] > 0:
                detection_rate = self.detection_metrics['new_comments_found'] / self.detection_metrics['total_scans']
                logger.info(f"🎯 Detection efficiency: {detection_rate:.2f} comments/scan")

            logger.info(f"\n📋 FINAL QUEUE STATUS")
            logger.info(f"{'='*60}")
            logger.info(f"📥 Remaining pending: {len(self.comment_queue['pending'])}")
            logger.info(f"✅ Total completed: {len(self.comment_queue['completed'])}")
            logger.info(f"❌ Total failed: {len(self.comment_queue['failed'])}")

            if len(self.comment_queue['completed']) + len(self.comment_queue['failed']) > 0:
                success_rate = len(self.comment_queue['completed']) / (len(self.comment_queue['completed']) + len(self.comment_queue['failed']))
                logger.info(f"📈 Session success rate: {success_rate:.1%}")

            logger.info(f"{'='*60}\n")
        except KeyboardInterrupt:
            logger.info("\n🛑 Bot stopped by user (Ctrl+C)")
            logger.info("📊 Saving final state and cleanup...")

            if len(self.comment_queue['pending']) > 0:
                logger.info(f"💾 {len(self.comment_queue['pending'])} comments remain in queue")

        except Exception as e:
            logger.critical(f'❌ Bot execution failed: {e}')
            import traceback
            logger.critical(traceback.format_exc())
            self.db.log_event('bot_crash', str(e), 'critical')

            logger.error(f"🆘 Emergency status - Pending: {len(self.comment_queue['pending'])}")
            logger.error(f"🆘 Emergency status - Processing: {len(self.comment_queue['processing'])}")

        finally:
            if self.browser:
                logger.info('Closing browser...')
                self.browser.close()
                logger.info('Browser closed.')
            if self.playwright:
                logger.info('Stopping Playwright...')
                self.playwright.stop()
                logger.info('Playwright stopped.')

            self.db.close()
            logger.info('Database connection closed.')
            self.db.log_event('bot_shutdown', 'Bot stopped gracefully', 'info')
def main():

    try:
        bot = FacebookAIReplyBot()
        bot.run()
    except Exception as e:
        logger.critical(f'Bot initialization failed: {e}')
if __name__ == '__main__':
    main()
