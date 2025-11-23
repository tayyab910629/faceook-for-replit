"""
Simple and Reliable Facebook Comment Reply Bot
"""
import os
import time
import logging
from datetime import datetime, timedelta
from playwright.sync_api import sync_playwright, Page
from openai import OpenAI
from database import SupabaseDatabase
from config import get_bot_config, get_supabase_config, get_openai_config
from utils.logger import setup_logger

logger = setup_logger(__name__)

class FacebookCommentBot:
    def __init__(self):
        """Initialize the bot"""
        self.config = get_bot_config()
        supabase_config = get_supabase_config()
        openai_config = get_openai_config()
        
        # Validate config
        if not self.config.get('POST_URL'):
            raise ValueError("POST_URL not configured")
        if not supabase_config.get('URL') or not supabase_config.get('ANON_KEY'):
            raise ValueError("Supabase not configured")
        if not openai_config.get('API_KEY'):
            raise ValueError("OpenAI API key not configured")
        
        self.db = SupabaseDatabase(
            supabase_url=supabase_config['URL'],
            supabase_key=supabase_config['ANON_KEY']
        )
        self.openai_client = OpenAI(api_key=openai_config['API_KEY'])
        self.openai_config = openai_config
        self.post_url = self.config.get('POST_URL')
        self.bot_name = self.config.get('MY_NAME', 'Bot')
        
        self.page: Page | None = None
        self.playwright = None
        self.browser = None
        self.processed_user_ids = set()  # Track users we've replied to in this session
        
        logger.info("✅ Bot initialized successfully")
        self.db.log_event('bot_startup', 'Bot started', 'info')
    
    def setup_browser(self):
        """Start Playwright and open Facebook"""
        try:
            self.playwright = sync_playwright().start()
            
            user_data_dir = os.path.join(os.getcwd(), 'facebook_user_data')
            os.makedirs(user_data_dir, exist_ok=True)
            
            logger.info("🌐 Starting browser...")
            self.browser = self.playwright.chromium.launch_persistent_context(
                user_data_dir,
                headless=self.config.get('HEADLESS', False),
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-popup-blocking',
                    '--no-sandbox',
                ],
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                viewport={'width': 1280, 'height': 720},
            )
            
            self.page = self.browser.new_page()
            logger.info("✅ Browser started")
            
        except Exception as e:
            logger.error(f"❌ Browser setup failed: {e}")
            raise
    
    def go_to_post(self):
        """Navigate to the Facebook post"""
        try:
            logger.info(f"📄 Loading post: {self.post_url}")
            self.page.goto(self.post_url, wait_until='domcontentloaded', timeout=30000)
            self.page.wait_for_load_state('networkidle', timeout=10000)
            time.sleep(2)
            logger.info("✅ Post loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load post: {e}")
            raise
    
    def get_comments(self):
        """Get all visible comments from the post"""
        try:
            # Scroll to load more comments
            for _ in range(3):
                self.page.evaluate('window.scrollBy(0, 500)')
                time.sleep(0.5)
            
            # Get comment elements
            comments = []
            comment_elements = self.page.locator("//div[@role='article']").all()
            
            logger.info(f"🔍 Found {len(comment_elements)} comment elements")
            
            for elem in comment_elements:
                try:
                    # Get author name
                    author_elem = elem.locator("span.x193iq5w[dir='auto']").first
                    if not author_elem or author_elem.count() == 0:
                        continue
                    
                    author = author_elem.text_content().strip()
                    if not author:
                        continue
                    
                    # Skip bot's own comments
                    if author.lower() == self.bot_name.lower():
                        logger.debug(f"⏭️ Skipping own comment by {author}")
                        continue
                    
                    # Get comment text
                    text_elem = elem.locator("div[dir='auto']").first
                    if not text_elem or text_elem.count() == 0:
                        continue
                    
                    comment_text = text_elem.text_content().strip()
                    if not comment_text or len(comment_text) < 2:
                        continue
                    
                    # Create unique comment ID (user + text + date)
                    today = datetime.now().strftime('%Y-%m-%d')
                    comment_id = f"{author}_{comment_text[:50]}_{today}"
                    
                    comments.append({
                        'id': comment_id,
                        'author': author,
                        'text': comment_text,
                        'element': elem,
                    })
                    
                    logger.debug(f"✓ Found comment by {author}: {comment_text[:50]}...")
                    
                except Exception as e:
                    logger.debug(f"Error processing comment: {e}")
                    continue
            
            return comments
        
        except Exception as e:
            logger.error(f"❌ Error getting comments: {e}")
            return []
    
    def should_reply_to_comment(self, comment_id, author):
        """Check if we should reply to this comment"""
        # Check if already replied to in database
        if self.db.has_replied_to_comment(comment_id):
            logger.debug(f"⏭️ Already replied to comment by {author}")
            return False
        
        # Check if we've replied to this user today
        user_reply_count = self.db.get_user_reply_count(author)
        max_per_user = int(self.config.get('MAX_REPLIES_PER_USER', 1))
        if user_reply_count >= max_per_user:
            logger.info(f"⏭️ Already replied to {author} {user_reply_count} time(s) today")
            return False
        
        return True
    
    def generate_reply(self, comment_text):
        """Generate a reply using OpenAI"""
        try:
            logger.debug(f"🤖 Generating reply for: {comment_text[:50]}...")
            
            system_prompt = "You are helpful assistant. Generate a brief, friendly reply to a Facebook comment. Keep it under 100 words. No emojis."
            user_prompt = f"Reply to this comment: {comment_text}"
            
            response = self.openai_client.chat.completions.create(
                model=self.openai_config.get('MODEL', 'gpt-4o-mini'),
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                temperature=0.7,
                max_tokens=100
            )
            
            reply = response.choices[0].message.content.strip()
            logger.info(f"✅ Generated reply: {reply[:50]}...")
            return reply
            
        except Exception as e:
            logger.error(f"❌ Failed to generate reply: {e}")
            return "Thanks for your comment!"
    
    def reply_to_comment(self, comment_data):
        """Reply to a specific comment"""
        author = comment_data['author']
        comment_text = comment_data['text']
        comment_id = comment_data['id']
        elem = comment_data['element']
        
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"💬 Replying to {author}")
            logger.info(f"Comment: {comment_text[:60]}...")
            logger.info(f"{'='*60}")
            
            # Find and click Reply button
            reply_btn = elem.locator("//div[@role='button' and contains(text(), 'Reply')]").first
            if not reply_btn or reply_btn.count() == 0:
                logger.warning(f"⚠️ No Reply button found for {author}")
                return False
            
            elem.scroll_into_view_if_needed()
            time.sleep(1)
            
            logger.debug("Clicking Reply button...")
            reply_btn.click(timeout=5000)
            time.sleep(2)
            
            # Find reply text box
            reply_box = self.page.locator("p[dir='auto'][contenteditable='true']").first
            if not reply_box or reply_box.count() == 0:
                logger.warning("⚠️ Reply box not found")
                return False
            
            logger.debug("Typing reply...")
            reply_text = self.generate_reply(comment_text)
            reply_box.click()
            time.sleep(0.5)
            reply_box.type(reply_text, delay=50)
            time.sleep(1)
            
            # Find and click Send button
            send_btn = self.page.locator("//div[@role='button' and @aria-label='Comment']").first
            if send_btn and send_btn.count() > 0:
                logger.debug("Clicking Send button...")
                send_btn.click(timeout=5000)
            else:
                # Try pressing Enter
                reply_box.press('Enter')
            
            time.sleep(2)
            
            # Record in database
            self.db.add_processed_comment(
                comment_id,
                author,
                author,
                comment_text[:1000],
                reply_text[:1000],
                status='success'
            )
            self.db.update_user_stats(author, author)
            
            logger.info(f"✅ Successfully replied to {author}")
            self.db.log_event('reply_success', f"Replied to {author}", 'info')
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to reply to {author}: {e}")
            self.db.log_event('reply_failed', f"Failed to reply to {author}: {e}", 'error')
            return False
    
    def run_continuously(self):
        """Run the bot continuously"""
        scan_interval = int(self.config.get('SCAN_INTERVAL', 30))
        max_replies = int(self.config.get('MAX_REPLIES', 100))
        replies_made = 0
        
        try:
            self.setup_browser()
            self.go_to_post()
            
            logger.info(f"\n{'='*60}")
            logger.info(f"🤖 FACEBOOK COMMENT BOT STARTED")
            logger.info(f"Checking every {scan_interval} seconds")
            logger.info(f"Post URL: {self.post_url}")
            logger.info(f"{'='*60}\n")
            
            while replies_made < max_replies:
                try:
                    logger.info(f"\n⏰ Checking for new comments... (Scan #{self.processed_user_ids.__len__() + 1})")
                    
                    # Get comments
                    comments = self.get_comments()
                    
                    if not comments:
                        logger.info("📭 No new comments found")
                    else:
                        logger.info(f"📬 Found {len(comments)} comment(s)")
                        
                        # Process each comment
                        for comment_data in comments:
                            author = comment_data['author']
                            comment_id = comment_data['id']
                            
                            if self.should_reply_to_comment(comment_id, author):
                                if self.reply_to_comment(comment_data):
                                    replies_made += 1
                                    time.sleep(2)
                            else:
                                logger.debug(f"⏭️ Skipping comment by {author}")
                            
                            if replies_made >= max_replies:
                                break
                    
                    # Print stats
                    logger.info(f"\n📊 Stats: {replies_made}/{max_replies} replies made")
                    logger.info(f"⏳ Waiting {scan_interval} seconds before next scan...")
                    
                    # Wait before next scan
                    time.sleep(scan_interval)
                    
                    # Reload page
                    self.page.reload(wait_until='domcontentloaded')
                    time.sleep(2)
                    
                except KeyboardInterrupt:
                    logger.info("\n⛔ Bot stopped by user")
                    break
                except Exception as e:
                    logger.error(f"⚠️ Error in main loop: {e}")
                    time.sleep(5)
            
            logger.info(f"\n✅ Completed! Made {replies_made} replies")
            
        except KeyboardInterrupt:
            logger.info("\n⛔ Bot stopped")
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
            logger.info("✅ Browser closed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        self.db.log_event('bot_shutdown', 'Bot stopped', 'info')
        logger.info("👋 Bot shutdown complete")


def main():
    try:
        bot = FacebookCommentBot()
        bot.run_continuously()
    except Exception as e:
        logger.critical(f"Bot failed to start: {e}")
        raise


if __name__ == '__main__':
    main()
