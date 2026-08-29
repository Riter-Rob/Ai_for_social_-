"""
Ethiopia/TelegramBot/config.py

Configuration for the Ethiopian Crisis Detection Telegram Bot.
Load settings from environment variables or .env file.

Setup:
  1. Create a bot via @BotFather on Telegram → get your token
  2. Copy .env.example to .env and fill in your token
  3. Run: python Ethiopia/TelegramBot/bot.py
"""

import os

# ─── Load .env file if python-dotenv is available ────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on environment variables


# ─── Bot Token ────────────────────────────────────────────────────────────────
# Get from @BotFather on Telegram
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

if not TELEGRAM_BOT_TOKEN:
    raise EnvironmentError(
        "\n\n"
        "  TELEGRAM_BOT_TOKEN is not set!\n\n"
        "  Steps to fix:\n"
        "  1. Open Telegram and message @BotFather\n"
        "  2. Type /newbot and follow the prompts\n"
        "  3. Copy the token you receive\n"
        "  4. Create a .env file in the project root:\n"
        "       TELEGRAM_BOT_TOKEN=your_token_here\n"
    )

# ─── Detection settings ───────────────────────────────────────────────────────
# Probability threshold above which a message is classified as crisis
CRISIS_THRESHOLD = float(os.environ.get("CRISIS_THRESHOLD", "0.60"))

# Path to the fine-tuned model (relative to project root)
MODEL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "Model", "ethiopian_crisis_model"
)

# ─── Bot behavior ─────────────────────────────────────────────────────────────
# Maximum message length before truncating for inference
MAX_TEXT_LENGTH = 512

# Log all messages for debugging (disable in production)
DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() == "true"

# Bot name displayed in welcome message
BOT_NAME = os.environ.get("BOT_NAME", "የስነ-ልቦና ድጋፍ ቦት")  # "Mental Health Support Bot"
