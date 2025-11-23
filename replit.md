# Facebook AI Comment Reply Bot

## Overview
This is a production-grade Facebook comment reply bot that automatically monitors a Facebook post and responds to comments using AI-generated replies. The bot uses Playwright for browser automation, OpenAI for generating contextual responses, and Supabase for tracking processed comments.

## Current State
- **Language**: Python 3.11
- **Status**: Configured and ready to run
- **Last Updated**: 2025-11-23

## Project Architecture

### Core Components
1. **cmt.py** - Main bot implementation with the FacebookAIReplyBot class
2. **config.py** - Configuration management and environment variable handling
3. **database.py** - Supabase database integration for comment tracking
4. **constants.py** - Application constants and static configuration
5. **utils/** - Utility modules (logger, retry, validators)

### Key Features
- AI-powered contextual replies using OpenAI GPT models
- Database-backed comment tracking to avoid duplicate replies
- Anti-detection mechanisms for browser automation
- Robust error handling and retry logic
- Rate limiting and user reply limits
- Comprehensive logging and monitoring
- Continuous operation mode with adaptive scanning

### Technology Stack
- **Browser Automation**: Playwright (Chromium)
- **AI/LLM**: OpenAI API (GPT-4o-mini default)
- **Database**: Supabase (PostgreSQL)
- **Environment**: Python 3.11 with dotenv

## Configuration

### Required Environment Variables
The bot requires several environment variables to be configured in a `.env` file:

#### Essential Configuration
- `POST_URL` - The Facebook post URL to monitor
- `FB_OUR_NAME` - Your Facebook name (for bot self-identification)
- `OPENAI_API_KEY` - OpenAI API key for generating replies
- `SUPABASE_URL` - Supabase project URL
- `SUPABASE_ANON_KEY` - Supabase anonymous key

#### Optional Configuration
See `.env.example` for all available configuration options including:
- Bot behavior settings (max replies, rate limits, etc.)
- Delay configurations for human-like behavior
- OpenAI model and prompt customization
- Chrome browser settings

### Supabase Database Setup
The bot requires the following tables in Supabase:
- `processed_comments` - Tracks all processed comments
- `user_stats` - Stores per-user reply statistics
- `rate_limit_log` - Rate limiting tracking
- `event_log` - System event logging

Refer to the SQL schema documentation for table creation.

## How to Use

### Initial Setup
1. Copy `.env.example` to `.env`
2. Fill in all required environment variables
3. Set up Supabase database with required tables
4. Ensure you're logged into Facebook (the bot uses persistent browser session)

### Running the Bot
The bot runs continuously when started via the workflow. It will:
1. Open the Facebook post in a browser
2. Monitor for new comments
3. Generate AI replies based on the comment and post context
4. Track processed comments in the database
5. Continue running until manually stopped (Ctrl+C)

### First Run
On the first run, you may need to:
1. Log into Facebook manually when the browser opens
2. The bot will save your login session for future runs
3. Once logged in, the bot will start monitoring automatically

## Project Structure
```
.
├── cmt.py                 # Main bot implementation
├── config.py              # Configuration management
├── database.py            # Supabase database integration
├── constants.py           # Application constants
├── utils/
│   ├── __init__.py
│   ├── logger.py          # Logging configuration
│   ├── retry.py           # Retry logic utilities
│   └── validators.py      # Input validation
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
├── .gitignore            # Git ignore rules
└── README.md             # Project README

Generated Directories:
├── logs/                  # Application logs
└── facebook_user_data/    # Persistent browser session data
```

## Recent Changes
- 2025-11-23: Initial setup in Replit environment
  - Installed Python 3.11 and all dependencies
  - Installed Playwright Chromium browser
  - Configured system dependencies for browser automation
  - Created .env.example template
  - Updated .gitignore for Python project

## User Preferences
- No specific preferences recorded yet

## Notes
- The bot requires active Facebook login to function
- Browser runs in visible mode by default (set `HEADLESS=true` for background operation)
- The bot saves browser session data in `facebook_user_data/` directory
- All logs are stored in `logs/` directory with timestamps
- The bot can be safely stopped with Ctrl+C for graceful shutdown
