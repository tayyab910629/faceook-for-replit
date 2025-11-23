import logging
from datetime import datetime, timedelta
from typing import Optional, Dict
from supabase import create_client, Client

logger = logging.getLogger(__name__)

class SupabaseDatabase:
    def __init__(self, supabase_url: str, supabase_key: str):
        try:
            self.client: Client = create_client(supabase_url, supabase_key)
            logger.info("✓ Supabase client initialized successfully")
            self._init_tables()
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise

    def _init_tables(self):
        try:
            result = self.client.table('processed_comments').select('count').limit(1).execute()
            logger.info(f"✓ Database tables verified - {result.count if result else 0} processed comments found")
        except Exception as e:
            logger.warning(f"⚠️ Could not verify tables. Please create them via Supabase SQL Editor. Error: {e}")
            logger.info("Run the SQL schema from SUPABASE_SCHEMA.sql to create tables")

    def is_comment_processed(self, comment_id: str) -> bool:
        try:
            result = self.client.table('processed_comments').select('id').eq('comment_id', comment_id).execute()
            return len(result.data) > 0 if result.data else False
        except Exception as e:
            logger.debug(f"Database offline, allowing comment {comment_id[:8]} to be processed: {e}")
            return False

    def add_processed_comment(self, comment_id: str, user_id: str, user_name: str, comment_text: str, response_text: str, status: str = 'success', retry_count: int = 0):
        try:
            data: Dict[str, str | int] = {
                'comment_id': comment_id,
                'user_id': user_id,
                'user_name': user_name,
                'comment_text': comment_text[:1000],
                'response_text': response_text[:1000],
                'status': status,
                'retry_count': retry_count,
                'timestamp': datetime.now().isoformat()
            }
            result = self.client.table('processed_comments').insert(data).execute()
            logger.info(f"✓ Logged processed comment: {comment_id} by {user_name}")
            return result
        except Exception as e:
            logger.warning(f"⚠️  Could not log processed comment (offline mode): {e}")
            return None

    def get_user_reply_count(self, user_id: str) -> int:
        try:
            result = self.client.table('user_stats').select('reply_count').eq('user_id', user_id).execute()
            if result.data and len(result.data) > 0:
                item = result.data[0]
                if isinstance(item, dict):
                    val = item.get('reply_count', 0)
                    if isinstance(val, int):
                        return val
                    elif isinstance(val, str) and val.isdigit():
                        return int(val)
                return 0
            return 0
        except Exception as e:
            logger.debug(f"Database offline, allowing reply to user {user_id[:8]}: {e}")
            return 0

    def get_last_reply_time(self, user_id: str) -> Optional[datetime]:
        try:
            result = self.client.table('user_stats').select('last_reply_time').eq('user_id', user_id).execute()
            if result.data and len(result.data) > 0:
                item = result.data[0]
                if isinstance(item, dict):
                    last_reply = item.get('last_reply_time')
                    if isinstance(last_reply, str):
                        return datetime.fromisoformat(last_reply.replace('Z', '+00:00'))
            return None
        except Exception as e:
            logger.debug(f"Database offline, allowing reply to user {user_id[:8]}: {e}")
            return None

    def update_user_stats(self, user_id: str, user_name: str):
        try:
            now = datetime.now().isoformat()
            existing = self.client.table('user_stats').select('user_id, reply_count').eq('user_id', user_id).execute()
            if existing.data and len(existing.data) > 0:
                item = existing.data[0]
                if isinstance(item, dict):
                    current_count = item.get('reply_count', 0)
                    if isinstance(current_count, int):
                        self.client.table('user_stats').update({
                            'reply_count': current_count + 1,
                            'last_reply_time': now,
                            'user_name': user_name
                        }).eq('user_id', user_id).execute()
            else:
                self.client.table('user_stats').insert({
                    'user_id': user_id,
                    'user_name': user_name,
                    'reply_count': 1,
                    'last_reply_time': now,
                    'first_reply_time': now
                }).execute()
            logger.debug(f"✓ Updated user stats for {user_name}")
        except Exception as e:
            logger.debug(f"⚠️  Could not update user stats (offline mode): {e}")

    def add_rate_limit_entry(self):
        try:
            data = {'timestamp': datetime.now().isoformat()}
            self.client.table('rate_limit_log').insert(data).execute()
        except Exception as e:
            logger.debug(f"⚠️  Could not log rate limit entry (offline mode): {e}")

    def get_recent_reply_count(self, window_seconds: int) -> int:
        try:
            cutoff_time = (datetime.now() - timedelta(seconds=window_seconds)).isoformat()
            result = self.client.table('rate_limit_log').select('id').gte('timestamp', cutoff_time).execute()
            return result.count if result.count is not None else 0
        except Exception as e:
            logger.debug(f"Database offline, rate limiting disabled: {e}")
            return 0

    def log_event(self, event_type: str, event_data: str, level: str = 'info'):
        try:
            data = {
                'event_type': event_type,
                'event_data': event_data[:2000],
                'level': level,
                'timestamp': datetime.now().isoformat()
            }
            self.client.table('event_log').insert(data).execute()
        except Exception as e:
            logger.error(f"Error logging event: {e}")

    def clean_old_rate_limit_entries(self, window_seconds: int):
        try:
            cutoff_time = (datetime.now() - timedelta(seconds=window_seconds * 2)).isoformat()
            self.client.table('rate_limit_log').delete().lte('timestamp', cutoff_time).execute()
            logger.debug(f"✓ Cleaned old rate limit entries before {cutoff_time}")
        except Exception as e:
            logger.error(f"Error cleaning rate limit entries: {e}")

    def get_statistics(self) -> Dict[str, int]:
        try:
            total_result = self.client.table('processed_comments').select('id').execute()
            total_comments = total_result.count if total_result.count is not None else 0
            unique_result = self.client.table('user_stats').select('user_id').execute()
            unique_users = unique_result.count if unique_result.count is not None else 0
            success_result = self.client.table('processed_comments').select('id').eq('status', 'success').execute()
            successful = success_result.count if success_result.count is not None else 0
            failed_result = self.client.table('processed_comments').select('id').eq('status', 'failed').execute()
            failed = failed_result.count if failed_result.count is not None else 0
            return {
                'total_comments': total_comments,
                'unique_users': unique_users,
                'successful': successful,
                'failed': failed
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {'total_comments': 0, 'unique_users': 0, 'successful': 0, 'failed': 0}

    def get_processed_comment(self, comment_id: str) -> Optional[Dict]:
        """Get details about a processed comment including the reply that was sent."""
        try:
            result = self.client.table('processed_comments').select('*').eq('comment_id', comment_id).order('timestamp', desc=True).limit(1).execute()
            if result.data and len(result.data) > 0:
                return result.data[0]
            return None
        except Exception as e:
            logger.debug(f"Error getting processed comment {comment_id[:8]}: {e}")
            return None

    def get_reply_for_comment(self, comment_id: str) -> Optional[str]:
        """Get the reply text that was sent for a specific comment."""
        try:
            result = self.client.table('processed_comments').select('response_text').eq('comment_id', comment_id).eq('status', 'success').order('timestamp', desc=True).limit(1).execute()
            if result.data and len(result.data) > 0:
                return result.data[0].get('response_text')
            return None
        except Exception as e:
            logger.debug(f"Error getting reply for comment {comment_id[:8]}: {e}")
            return None

    def get_all_replied_comment_ids(self, limit: int = 1000) -> list[str]:
        """Get a list of all comment IDs that have been replied to."""
        try:
            result = self.client.table('processed_comments').select('comment_id').eq('status', 'success').order('timestamp', desc=True).limit(limit).execute()
            if result.data:
                return [item.get('comment_id') for item in result.data if item.get('comment_id')]
            return []
        except Exception as e:
            logger.debug(f"Error getting replied comment IDs: {e}")
            return []

    def get_recent_replies(self, hours: int = 24, limit: int = 100) -> list[Dict]:
        """Get recent replies within the specified hours."""
        try:
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            result = self.client.table('processed_comments').select('*').eq('status', 'success').gte('timestamp', cutoff_time).order('timestamp', desc=True).limit(limit).execute()
            if result.data:
                return result.data
            return []
        except Exception as e:
            logger.debug(f"Error getting recent replies: {e}")
            return []

    def get_replies_by_user(self, user_id: str) -> list[Dict]:
        """Get all replies sent to a specific user."""
        try:
            result = self.client.table('processed_comments').select('*').eq('user_id', user_id).eq('status', 'success').order('timestamp', desc=True).execute()
            if result.data:
                return result.data
            return []
        except Exception as e:
            logger.debug(f"Error getting replies for user {user_id[:8]}: {e}")
            return []

    def has_replied_to_comment(self, comment_id: str) -> bool:
        """Check if we have successfully replied to a comment (status = 'success')."""
        try:
            result = self.client.table('processed_comments').select('id').eq('comment_id', comment_id).eq('status', 'success').limit(1).execute()
            return len(result.data) > 0 if result.data else False
        except Exception as e:
            logger.debug(f"Error checking if replied to comment {comment_id[:8]}: {e}")
            return False

    def close(self):
        logger.info("Supabase connection closed (REST client)")
